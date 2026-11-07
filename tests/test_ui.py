"""The pages are static files with no build step, so these read them as text."""

import json
from pathlib import Path
import re

import pytest

from isharded import orchestrator
from isharded.assistant.gateway import page_response

PACKAGE = Path(orchestrator.__file__).parent
ROOT = PACKAGE.parent
PAGES = ["chat.html", "dashboard.html"]
MARKER = "/* isharded:ui.css */"
STYLES = (PACKAGE / "ui.css").read_text(encoding="utf-8")
# An XML namespace is a name, and nothing ever fetches it.
NOT_FETCHED = {"http://www.w3.org/2000/svg"}
# A page may call this before the route lands, because it says so plainly on a 404.
NOT_BUILT_YET = {("POST", "/admin/invites"), ("POST", "/admin/join/{name}/approve"), ("POST", "/admin/join/{name}/deny")}


def source(name):
    return (PACKAGE / name).read_text(encoding="utf-8")


def served(name):
    return orchestrator.page(name).body.decode("utf-8")


def calls(text):
    """Every (method, path) a page sends, from api() and from its one direct fetch."""
    found = set()
    for match in re.finditer(r"api\('(/[^']+)'(?:,\s*\{([^}]*)\})?", text):
        method = re.search(r"method:\s*'(\w+)'", match.group(2) or "")
        found.add((method.group(1) if method else "GET", match.group(1)))
    for match in re.finditer(r"api\(`(/[^`]+)`(?:,\s*\{([^}]*)\})?", text):
        method = re.search(r"method:\s*'(\w+)'", match.group(2) or "")
        found.add((method.group(1) if method else "GET", re.sub(r"\$\{[^}]*\}", "{name}", match.group(1))))
    for match in re.finditer(r"fetch\(base \+ '(/[^']+)',\s*\{\s*method:\s*'(\w+)'", text):
        found.add((match.group(2), match.group(1)))
    return found


def routes():
    return {(method, route.path) for route in orchestrator.app.routes for method in getattr(route, "methods", None) or ()}


def theme_tokens():
    """The light and dark palettes, as each theme block in ui.css declares them."""
    blocks = re.findall(r"^:root(?:\[data-theme=\"dark\"\])? \{(.*?)\n\}", STYLES, re.S | re.M)
    assert len(blocks) == 2
    return [dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6});", block)) for block in blocks]


def contrast(first, second):
    def luminance(color):
        parts = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        red, green, blue = [part / 12.92 if part <= 0.03928 else ((part + 0.055) / 1.055) ** 2.4 for part in parts]
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("name", PAGES)
def test_each_page_asks_for_the_shared_styles_once(name):
    assert source(name).count(MARKER) == 1


@pytest.mark.parametrize("name", PAGES)
def test_serving_a_page_inlines_the_shared_styles(name):
    page = served(name)
    assert MARKER not in page
    assert STYLES in page


def test_the_design_system_ships_with_the_package():
    assert '"ui.css"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", PAGES)
def test_a_page_reaches_for_nothing_outside_its_node(name):
    page = served(name)
    assert set(re.findall(r"https?://[^\s\"'<>)`]+", page)) <= NOT_FETCHED
    assert not re.search(r"<script[^>]+\bsrc=", page)
    assert not re.search(r"<link[^>]+rel=[\"']?stylesheet", page)
    assert not re.search(r"@import|@font-face", page)
    assert not re.search(r"(?:src|href)=[\"']//", page)
    assert not re.search(r"url\((?!#)", page)


@pytest.mark.parametrize("name", PAGES)
def test_every_api_path_a_page_calls_exists(name):
    called = calls(source(name))
    assert called, "the pattern no longer finds this page's calls"
    assert not called - routes() - NOT_BUILT_YET


def test_the_pages_still_call_what_they_are_for():
    called = calls(source("chat.html")) | calls(source("dashboard.html"))
    for call in [
        ("GET", "/admin/status"), ("GET", "/admin/catalog"), ("POST", "/admin/plan"), ("POST", "/admin/load"), ("POST", "/admin/load/cancel"),
        ("POST", "/admin/rebalance"), ("GET", "/admin/inventory"), ("POST", "/admin/invites"), ("POST", "/v1/chat/completions"),
        ("GET", "/join/requests"), ("POST", "/admin/join/{name}/approve"), ("POST", "/admin/join/{name}/deny"),
    ]:
        assert call in called


@pytest.mark.parametrize("name", PAGES)
def test_the_key_stays_where_a_person_left_it(name):
    text = source(name)
    assert text.count("'isharded-token'") >= 3  # read, kept, forgotten
    assert "history.replaceState" in text  # a link's key leaves the address bar
    assert "localStorage.setItem('isharded-token'" not in text.replace("store.set('isharded-token'", "")


@pytest.mark.parametrize("name", PAGES)
def test_what_arrives_from_outside_is_escaped(name):
    text = source(name)
    assert re.search(r"const esc = value =>", text)
    # Each of these carries text a worker, a model or a server chose.
    for name_of_field in ("error.message", "job.error", "quarantine_reason", "lane.error", "event.detail", "tel.last_error"):
        for match in re.finditer(r"\$\{([^{}]*" + re.escape(name_of_field) + r"[^{}]*)\}", text):
            assert "esc(" in match.group(1), match.group(0)


def test_a_key_that_is_not_an_admins_does_not_sign_anyone_out():
    text = source("dashboard.html")
    for call in re.findall(r"api\([`'](?:/join/|/admin/join/)[^\n]*", text):
        assert "admin: true" in call


def test_a_flag_is_checked_first_and_names_its_country():
    mark = re.search(r"const flagMark = .*", source("dashboard.html")).group(0)
    assert mark.index("flagged(code)") < mark.index("#flag-${code}")  # a worker chooses this text
    assert 'role="img"' in mark and "aria-label=" in mark and "data-tip=" in mark


def test_chat_has_one_model_selector():
    text = source("chat.html")
    composer = text[text.index('id="composer"'):text.index("</form>", text.index('id="composer"'))]
    assert "<select" not in composer
    bar = text[text.index('<header class="top">'):text.index("</header>")]
    assert 'id="modelChoice"' in bar


def test_chat_draws_the_shared_route_and_not_its_old_stacks():
    text = source("chat.html")
    assert "drawRouteList(" in text and "drawRail(" in text
    for retired in ("core-stage", "ribbon", "class=\"pill"):
        assert retired not in text and retired not in STYLES


def test_chat_keeps_conversations_in_the_browser_and_forgets_them_on_sign_out():
    text = source("chat.html")
    handler = text[text.index("$('signOut').onclick"):text.index("$('themeButton').onclick")]
    assert "store.drop(CHATS_KEY)" in handler and "state.chats = []" in handler
    assert "'isharded-chats'" in text


def test_chat_sends_tools_only_when_someone_set_them():
    assert "tools: state.tools.length ? state.tools : undefined" in source("chat.html")


def test_a_recovered_reply_comes_from_the_pools_own_count():
    text = source("chat.html")
    assert "recoveries_total" in text and "seamHtml" in text
    assert ".seam" in STYLES


def test_moving_a_reply_still_asks_first():
    text = source("chat.html")
    assert "await askToMove(" in text and 'id="move"' in text


@pytest.mark.parametrize("name", PAGES)
def test_a_page_links_to_the_other_and_has_a_real_shape(name):
    text = source(name)
    assert '<html lang="en">' in text
    assert 'name="viewport"' in text
    assert 'name="color-scheme" content="light dark"' in text
    assert 'href="/chat"' in text and 'href="/dashboard"' in text
    assert 'aria-current="page"' in text
    assert 'class="skip"' in text


def test_the_styles_cover_themes_motion_and_focus():
    for needle in ("prefers-color-scheme: dark", ':root[data-theme="dark"]', ':root:not([data-theme="light"])', "prefers-reduced-motion: reduce", ":focus-visible", "forced-colors: active"):
        assert needle in STYLES


def test_the_two_dark_palettes_stay_identical():
    """Dark is written twice, once for the system and once for the switch. They must not drift."""
    system = re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}', STYLES, re.S).group(1)
    manual = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', STYLES, re.S).group(1)
    assert system.split() == manual.split()


@pytest.mark.parametrize("theme", [0, 1], ids=["light", "dark"])
@pytest.mark.parametrize("text, ground", [
    ("ink", "paper"), ("ink", "plate"), ("ink", "plate-2"), ("ink", "plate-3"), ("ink-2", "paper"), ("ink-2", "plate"), ("ink-2", "plate-2"),
    ("accent-text", "plate"), ("accent-text", "plate-2"), ("on-accent", "accent"), ("good", "plate"), ("warn", "plate"), ("bad", "plate"),
    ("ink", "warn-soft"), ("ink", "bad-soft"), ("bad", "bad-soft"), ("warn", "warn-soft"),
])
def test_text_colours_pass_wcag_aa(theme, text, ground):
    tokens = theme_tokens()[theme]
    assert contrast(tokens[text], tokens[ground]) >= 4.5


@pytest.mark.parametrize("theme", [0, 1], ids=["light", "dark"])
@pytest.mark.parametrize("mark", ["tier-vram", "tier-unified", "tier-ram", "tier-disk", "edge-strong"])
def test_marks_stand_out_from_the_plate(theme, mark):
    tokens = theme_tokens()[theme]
    assert contrast(tokens[mark], tokens["plate"]) >= 3


SCRIPT_MARKER = "/* isharded:ui.js */"
SCRIPT = (PACKAGE / "ui.js").read_text(encoding="utf-8")
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "status").glob("*.json"))


@pytest.mark.parametrize("name", PAGES)
def test_a_page_inlines_the_shared_drawing_once(name):
    assert source(name).count(SCRIPT_MARKER) == 1
    page = served(name)
    assert SCRIPT_MARKER not in page and SCRIPT in page
    assert '"ui.js"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_the_assistant_page_inlines_both_shared_files_once():
    template = (PACKAGE / "assistant" / "assistant.html").read_text(encoding="utf-8")
    assert template.count(SCRIPT_MARKER) == 1 and template.count(MARKER) == 1
    page = page_response().body.decode("utf-8")
    assert SCRIPT_MARKER not in page and SCRIPT in page and STYLES in page


def test_the_assistant_page_keeps_inside_its_policy():
    template = (PACKAGE / "assistant" / "assistant.html").read_text(encoding="utf-8")
    assert 'style="' not in template and "url(" not in template  # style-src refuses a style attribute, even through innerHTML
    assert not re.search(r"#[0-9a-fA-F]{6}", template)
    assert not re.search(r"\bon[a-z]+=", template)  # script-src refuses an inline handler
    assert re.search(r"<script nonce", template) and "<script>" not in template


def test_the_shared_drawing_works_under_the_assistants_policy():
    assert 'style="' not in SCRIPT  # the policy refuses a style attribute, even one set through innerHTML
    assert "fetch(" not in SCRIPT


@pytest.mark.parametrize("theme", [0, 1], ids=["light", "dark"])
def test_the_board_has_its_own_colours_in_both_themes(theme):
    assert {"band", "band-slow", "on-track"} <= set(theme_tokens()[theme])


def test_a_status_answer_is_not_presented_as_coming_from_this_node():
    text = source("dashboard.html")
    assert "data.mesh.node" not in text  # /admin/status is forwarded, so this names the coordinator
    assert "This node <b>" not in text


@pytest.mark.parametrize("name", PAGES + ["ui.js"])
def test_colour_lives_only_in_the_styles(name):
    assert not re.search(r"#[0-9a-fA-F]{6}", source(name))


def test_the_healthy_state_spends_no_colour():
    assert re.search(r"look\('idle', '', 'Healthy'", source("dashboard.html"))


def test_rebalance_asks_first_when_it_would_free_a_quarantine():
    text = source("dashboard.html")
    assert "Keep them out" in text and "another chance" in text


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_a_fixture_is_labelled_and_names_nobody(path):
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["simulated"] is True  # the page says so on its face
    for name, worker in data["workers"].items():
        assert name.startswith("sample-") and worker["country"] is None
        assert ".invalid" in worker["advertised_url"]
    assert not re.search(r"\d{1,3}(?:\.\d{1,3}){3}", text.replace("3.10.5", ""))


def test_every_fixture_the_plan_names_exists():
    names = {path.stem for path in FIXTURES}
    assert names >= {"calm-7", "repair-7", "spill-7", "lost-7", "load-lane-7", "two", "forty", "two-hundred", "fresh-no-times", "no-model", "no-machines"}


def open_dashboard(playwright, fixture, status_change=None):
    """The real page in real Chrome, with a fixture in place of the pool."""
    try:
        chrome = playwright.chromium.launch(channel="chrome")
    except Exception as error:  # no Chrome on this machine
        pytest.skip(f"Chrome is not available: {error}")
    data = json.loads((ROOT / "tests" / "fixtures" / "status" / f"{fixture}.json").read_text(encoding="utf-8"))
    if status_change:
        status_change(data)
    page = chrome.new_context(viewport={"width": 1440, "height": 900}).new_page()
    page.route("**/*", lambda route: route.fulfill(status=404, body="{}", content_type="application/json"))
    page.route("**/dashboard", lambda route: route.fulfill(body=served("dashboard.html"), content_type="text/html"))
    page.route("**/admin/status", lambda route: route.fulfill(body=json.dumps(data), content_type="application/json"))
    page.goto("http://pool.invalid/dashboard")
    page.wait_for_selector("#consoleBody:not([hidden])")
    return chrome, page


def test_many_machines_keep_true_counts_and_search_reaches_a_closed_group():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        chrome, page = open_dashboard(playwright, "two-hundred")
        idle = page.locator("[data-group=Idle]")
        assert idle.get_attribute("aria-expanded") == "false"
        idle_total = sum(worker["state"] == "standby" for worker in page.evaluate("Object.values(state.data.workers)"))
        assert idle.inner_text().split("workers")[0].split()[-1] == str(idle_total)
        page.fill("#nodeSearch", "sample-1")
        assert page.locator("#nodeTable tr.row").count() == 50  # the first page of them, in spite of the closed group
        page.click("[data-action=more-rows]")
        assert page.locator("#nodeTable tr.row").count() > 50
        chrome.close()


def test_a_failure_in_activity_reads_as_a_line():
    def add_failure(data):
        name = next(iter(data["workers"]))
        data["events"].insert(0, {"id": "sample", "at": data["events"][0]["at"], "kind": "failure", "title": f"{name} quarantined", "detail": "reply failed: the link closed"})

    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        chrome, page = open_dashboard(playwright, "calm-7", add_failure)
        assert page.locator("#eventList .event").first.bounding_box()["height"] < 80
        chrome.close()


def test_a_node_that_stops_answering_takes_its_all_clear_with_it():
    text = source("dashboard.html")
    offline = text[text.index("function renderOffline"):text.index("function render(data)")]
    assert "$('headline').textContent = 'This node stopped answering'" in offline
    assert "setWord('work', 'work', 'Checking')" in text and "setWord('work', 'work', 'Checking')" in source("chat.html")


def test_a_group_choice_works_on_a_small_pool():
    assert "Boolean(store.get('isharded-group'))" in source("dashboard.html")


def test_a_reply_that_ends_late_leaves_the_open_conversation_alone():
    assert "if (chat === state.chat) touchChat();" in source("chat.html")


def test_the_chat_word_counts_machines_asking_to_join():
    text = source("chat.html")
    assert "+ state.joins" in text and "/join/requests" in text

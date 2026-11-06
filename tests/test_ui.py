"""The pages are static files with no build step, so these read them as text."""

from pathlib import Path
import re

import pytest

from shardmesh import orchestrator

PACKAGE = Path(orchestrator.__file__).parent
ROOT = PACKAGE.parent
PAGES = ["chat.html", "dashboard.html"]
MARKER = "/* shardmesh:ui.css */"
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
    assert text.count("'shardmesh-token'") >= 3  # read, kept, forgotten
    assert "history.replaceState" in text  # a link's key leaves the address bar
    assert "localStorage.setItem('shardmesh-token'" not in text.replace("store.set('shardmesh-token'", "")


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


def test_chat_has_one_model_selector():
    text = source("chat.html")
    composer = text[text.index('id="composer"'):text.index("</form>", text.index('id="composer"'))]
    assert "<select" not in composer
    bar = text[text.index('<header class="top">'):text.index("</header>")]
    assert 'id="modelChoice"' in bar


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

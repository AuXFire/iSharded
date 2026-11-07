"""The page in a real browser, for what only a browser can show.

Each test here is a bug a person found by using the page. They need Chrome and
Playwright, and are skipped without them. wait_for_function is not used, because it
evals in the page and the page's own policy refuses that, as it should.
"""
from __future__ import annotations

from collections.abc import Iterator
import time
from typing import Any

import pytest

from isharded.assistant.gateway import Users
from test_assistant_system import OWNER, Install, ScriptedModel, free_port, install, model, script_tidy  # noqa: F401

playwright_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as playwright:
        try:
            chrome = playwright.chromium.launch(channel="chrome")
        except playwright_api.Error as error:
            pytest.skip(f"Chrome is not available: {error}")
        yield chrome
        chrome.close()


@pytest.fixture
def page(browser: Any, install: Install, model: ScriptedModel) -> Iterator[Any]:  # noqa: F811
    install.use(model)
    context = browser.new_context()
    tab = context.new_page()
    tab.problems = []
    tab.on("console", lambda message: tab.problems.append(message.text) if message.type == "error" else None)
    tab.on("pageerror", lambda error: tab.problems.append(str(error)))
    tab.goto(install.url)
    yield tab
    context.close()


def until(page: Any, expression: str, seconds: float = 15.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"Still not true after {seconds} s: {expression}")


def sign_in(page: Any, key: str) -> None:
    page.fill("#key", key)
    page.click("#sign-in-form button[type=submit]")
    until(page, "() => document.getElementById('who').textContent !== ''")


def say(page: Any, text: str) -> None:
    page.fill("#input", text)
    page.press("#input", "Enter")
    until(page, "() => document.querySelector('article.turn.assistant .receipt') !== null")


def asking_for_a_key(page: Any) -> bool:
    return page.locator("#sign-in").evaluate("sheet => sheet.open")


def shown_to_the_next_person(page: Any) -> str:
    """textContent, as innerText skips what is hidden, and hidden text is still on the page."""
    return page.evaluate("() => ['turns', 'panel-approvals', 'panel-tasks', 'panel-history', 'panel-audit', 'panel-manage', 'memory-list', 'who'].map(id => document.getElementById(id).textContent).join(' ')")


def test_escape_cannot_leave_a_person_with_no_way_to_sign_in(page):
    assert asking_for_a_key(page)
    page.keyboard.press("Escape")
    until(page, "() => document.getElementById('sign-in').open")
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")  # Chrome closes a dialog on the second one, however it is asked not to
    until(page, "() => document.getElementById('sign-in').open")


def test_a_wrong_key_is_told_so_and_the_right_one_still_works(page, install):  # noqa: F811
    page.fill("#key", "not-a-key")
    page.click("#sign-in-form button[type=submit]")
    until(page, "() => !document.getElementById('sign-in-error').hidden")
    assert "not accepted" in page.inner_text("#sign-in-error")
    sign_in(page, install.key)
    assert page.inner_text("#who") == f"with {OWNER}"
    assert not asking_for_a_key(page)


def test_signing_out_leaves_nothing_of_that_person_on_the_page(page, install, model):  # noqa: F811
    model.on("my secret plans", "I will keep them to myself.")
    sign_in(page, install.key)
    say(page, "Here are my secret plans")
    page.click("[data-panel=history]")
    until(page, "() => document.querySelectorAll('#panel-history .card').length > 0")
    page.click("[data-panel=audit]")
    until(page, "() => document.querySelectorAll('#panel-audit .audit-line').length > 0")

    page.click("#sign-out")
    assert asking_for_a_key(page)
    left = shown_to_the_next_person(page)
    assert "secret" not in left and "keep them" not in left and "conversation" not in left and OWNER not in left
    assert page.evaluate("() => sessionStorage.getItem('assistant-key') === null && sessionStorage.getItem('assistant-conversation') === null")


def test_a_key_that_stops_working_clears_the_page_for_whoever_comes_next(page, install, model):  # noqa: F811
    model.on("my diary", "Noted.")
    users = Users(install.assistant)
    leaving = users.add("leaving")
    staying = users.add("staying")
    sign_in(page, leaving)
    say(page, "Here is my diary")
    conversation = page.evaluate("() => sessionStorage.getItem('assistant-conversation')")

    users.remove("leaving")
    page.fill("#input", "Are you there?")
    page.press("#input", "Enter")
    until(page, "() => document.getElementById('sign-in').open")
    assert "not accepted" in page.inner_text("#sign-in-error")
    assert "diary" not in shown_to_the_next_person(page)

    sign_in(page, staying)
    assert "diary" not in page.inner_text("#turns")
    assert page.evaluate("() => sessionStorage.getItem('assistant-conversation')") != conversation


def test_the_page_works_inside_its_own_policy(page, install, model):  # noqa: F811
    """A strict policy fails silently: something refused here only shows in the console."""
    sign_in(page, install.key)
    say(page, "Hello")
    for panel in ("memory", "tasks", "history", "audit", "approvals"):
        page.click(f"[data-panel={panel}]")
    page.click("#theme-button")
    assert [problem for problem in page.problems if "401" not in problem] == []
    assert page.evaluate("() => getComputedStyle(document.body).backgroundColor") != "rgba(0, 0, 0, 0)"  # ui.css really arrived


def open_manage(page: Any) -> None:
    page.click("[data-panel=manage]")
    until(page, "() => !document.querySelector('#part-answers .panel').textContent.includes('Checking') && document.querySelector('#part-peers .panel').childElementCount > 0")


def test_manage_works_inside_the_policy_and_says_what_each_part_holds(page, install):  # noqa: F811
    sign_in(page, install.key)
    open_manage(page)
    for title in ("What it may do", "Who answers", "Skills", "Plugins", "Code runs", "Peers"):
        assert title in page.inner_text("#panel-manage")
    assert "Answering" in page.inner_text("#part-answers")
    assert page.locator("#part-answers .route-no").first.inner_text() == "1"  # tried in order, so numbered
    assert [problem for problem in page.problems if "401" not in problem] == []


def test_a_member_has_no_manage_panel(page, install):  # noqa: F811
    sign_in(page, Users(install.assistant).add("member"))
    assert page.locator("[data-panel=manage]").is_hidden()
    assert page.inner_text("#panel-manage") == ""


def test_signing_out_of_manage_leaves_it_closed_for_a_member(page, install):  # noqa: F811
    sign_in(page, install.key)
    open_manage(page)
    page.click("#sign-out")
    assert OWNER not in shown_to_the_next_person(page)
    sign_in(page, Users(install.assistant).add("member"))
    assert page.locator("[data-panel=manage]").is_hidden() and page.locator("#panel-manage").is_hidden()
    assert page.locator("#panel-approvals").is_visible()


def test_a_grant_is_added_and_revoked_from_the_page_and_a_refusal_stays_in_the_sheet(page, install):  # noqa: F811
    sign_in(page, install.key)
    open_manage(page)
    page.click("#part-grants button:text('Add a grant')")
    page.select_option("#ask-fields select", "files.read")
    page.click("#ask-yes")  # a path capability with no scope
    until(page, "() => !document.getElementById('ask-error').hidden")
    assert page.locator("#ask").evaluate("sheet => sheet.open")

    page.select_option("#ask-fields select", "system.info")
    assert page.locator("#ask-fields input").first.is_hidden()  # it takes no scope, so nothing asks for one
    page.click("#ask-yes")
    until(page, "() => document.querySelector('#part-grants .panel').textContent.includes('system.info, for')")
    assert len(install.assistant.broker.grants(OWNER)) == 1

    page.click("#part-grants .card button:text('Revoke')")
    page.click("#ask-yes")
    until(page, "() => !document.querySelector('#part-grants .panel').textContent.includes('system.info, for')")
    assert install.assistant.broker.grants(OWNER) == []


def test_a_provider_that_stops_answering_is_said_in_words(page, install):  # noqa: F811
    install.assistant.providers.add("gone", f"http://127.0.0.1:{free_port()}", reach="pool")
    sign_in(page, install.key)
    page.click("[data-panel=manage]")
    until(page, "() => document.getElementById('part-answers').textContent.includes('Not answering')")
    assert "gone" in page.inner_text("#manage-summary")
    assert "Answering" in page.inner_text("#part-answers")  # the one that is up says so too


def test_a_lost_gateway_is_said_and_sending_comes_back_by_itself(page, install):  # noqa: F811
    sign_in(page, install.key)
    page.route("**/assistant/approvals", lambda route: route.abort())
    page.evaluate("() => poll()")
    until(page, "() => document.getElementById('notices').textContent.includes('stopped answering')")
    assert page.locator("#send").is_disabled()
    assert "Last heard" in page.inner_text("#pill")
    page.unroute("**/assistant/approvals")
    page.evaluate("() => poll()")
    until(page, "() => document.getElementById('notices').textContent === ''")
    assert page.locator("#send").is_enabled()
    assert page.evaluate("() => document.title") != ""


def test_a_failed_reply_stays_in_the_turn_with_a_way_to_send_it_again(page, install):  # noqa: F811
    install.assistant.providers.remove("scripted", by=OWNER)
    install.assistant.providers.add("gone", f"http://127.0.0.1:{free_port()}", reach="pool")
    sign_in(page, install.key)
    page.fill("#input", "Are you there?")
    page.press("#input", "Enter")
    until(page, "() => document.querySelector('article.turn.assistant .failure') !== null", 60)
    assert page.locator("article.turn.assistant .receipt").count() == 1
    page.click("article.turn.assistant .failure button")
    assert page.input_value("#input") == "Are you there?"


def test_an_approval_shows_the_exact_action_and_the_top_bar_counts_it(page, install, model, tmp_path):  # noqa: F811
    folder = tmp_path / "work"
    folder.mkdir()
    install.grant("files.read", str(folder))
    install.grant("files.write", str(folder))
    tidied = script_tidy(model, folder)
    sign_in(page, install.key)
    page.fill("#input", "Please tidy my notes")
    page.press("#input", "Enter")
    until(page, "() => document.querySelector('#panel-approvals .issue') !== null")
    assert "Waiting for you" in page.inner_text("article.turn.assistant .receipt")
    assert page.inner_text("#pill") == "1 needs you"
    assert page.evaluate("() => document.title").startswith("1 needs you")
    assert str(tidied.name) in page.inner_text("#panel-approvals .arguments")
    page.click("#panel-approvals button:text('Approve this exact action')")
    until(page, "() => document.getElementById('panel-approvals').textContent.includes('Nothing is waiting')")
    assert tidied.exists()
    assert page.inner_text("#pill") == "Nothing waiting"


def test_a_models_file_path_is_shown_as_its_name(page):
    named = "text => named(text)"
    assert page.evaluate(named, "pool/C:\\models\\small\\Sample-8B.gguf failed") == "pool/Sample-8B failed"
    assert page.evaluate(named, "/srv/models/Sample-8B.gguf") == "Sample-8B"
    assert page.evaluate(named, "pool cannot be reached") == "pool cannot be reached"

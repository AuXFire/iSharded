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
from test_assistant_system import OWNER, Install, ScriptedModel, install, model  # noqa: F401

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
    return page.evaluate("() => ['turns', 'panel-approvals', 'panel-tasks', 'panel-history', 'panel-audit', 'memory-list', 'who'].map(id => document.getElementById(id).textContent).join(' ')")


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

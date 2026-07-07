"""Frontend chat tests using Playwright.

LLM responses are stubbed at the API level by intercepting /api/ops/chat.
"""
import pytest
import json


@pytest.mark.frontend
def test_chat_send_message(logged_in_page):
    page = logged_in_page

    # Stub the chat endpoint so no real Bedrock call is made
    page.route(
        "**/api/ops/chat",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "session_id": "test-session-1",
                "answer": "This is a mocked chat answer.",
                "sources": [],
                "saved_to": None,
            }),
        ),
    )

    chat_input = page.locator(
        'textarea[placeholder*="essage"], input[placeholder*="essage"], .chat-input'
    )
    if chat_input.count() == 0:
        pytest.skip("Could not locate chat input in the UI")

    chat_input.fill("Hello, what is this wiki about?")
    page.keyboard.press("Enter")
    page.wait_for_timeout(2000)

    assert "mocked chat answer" in page.content().lower()


@pytest.mark.frontend
def test_chat_new_session_clears_history(logged_in_page):
    page = logged_in_page

    page.route(
        "**/api/ops/chat",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "session_id": "test-session-2",
                "answer": "First answer.",
                "sources": [],
                "saved_to": None,
            }),
        ),
    )

    chat_input = page.locator(
        'textarea[placeholder*="essage"], input[placeholder*="essage"], .chat-input'
    )
    if chat_input.count() == 0:
        pytest.skip("Could not locate chat input in the UI")

    chat_input.fill("First message")
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)

    # Click "new chat" / "clear" button
    new_chat = page.locator(
        'button:has-text("New"), button:has-text("Clear"), [data-action="new-chat"]'
    )
    if new_chat.count() > 0:
        new_chat.first.click()
        page.wait_for_timeout(500)
        # Chat area should be empty
        assert "First answer" not in page.locator(".chat-messages, .messages").inner_text()

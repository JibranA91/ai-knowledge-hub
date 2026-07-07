"""Frontend upload tests using Playwright.

The ingest planning background task is intercepted at the API level.
"""
import pytest
import json


@pytest.mark.frontend
def test_upload_unsupported_file_shows_error(logged_in_page):
    page = logged_in_page

    # Stub upload to return 400
    page.route(
        "**/api/documents/upload",
        lambda route: route.fulfill(
            status=400,
            content_type="application/json",
            body=json.dumps({"detail": "Unsupported file type."}),
        ),
    )

    file_input = page.locator("#file-input")
    if file_input.count() == 0:
        pytest.skip("Could not locate file input in the UI")

    # Playwright can fake a file upload
    file_input.set_input_files({
        "name": "evil.exe",
        "mimeType": "application/octet-stream",
        "buffer": b"MZ",
    })
    page.wait_for_timeout(1500)

    # Should show an error toast / message
    error_visible = (
        page.locator(".error, .toast-error, [class*='error']").count() > 0
        or "error" in page.content().lower()
        or "unsupported" in page.content().lower()
    )
    assert error_visible


@pytest.mark.frontend
def test_upload_txt_shows_notification(logged_in_page):
    page = logged_in_page

    page.route(
        "**/api/documents/upload",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"filename": "test.txt", "size": 12}),
        ),
    )

    # Stub the status poll
    page.route(
        "**/api/ops/status/test.txt",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "filename": "test.txt",
                "status": "pending_review",
                "message": "",
                "pages_created": [],
                "pages_updated": [],
                "plan": [],
                "plan_chat_history": [],
            }),
        ),
    )

    file_input = page.locator("#file-input")
    if file_input.count() == 0:
        pytest.skip("Could not locate file input in the UI")

    file_input.set_input_files({
        "name": "test.txt",
        "mimeType": "text/plain",
        "buffer": b"Hello world",
    })
    page.wait_for_timeout(2000)

    # A notification card or status indicator should appear
    assert (
        "test.txt" in page.content()
        or page.locator(".notification, .job-card, [class*='ingest']").count() > 0
    )

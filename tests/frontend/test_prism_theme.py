"""Prism appearance, daylight boundaries, and existing-screen layout regressions.

No model calls or changes to application data. Wiki content is intercepted.
"""
import json
from datetime import datetime, timezone

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.frontend
_KEY = "akh.appearance.v1"
_KYIV = {"latitude": 50.5, "longitude": 30.5}
_ARTICLE = """# A shared knowledge space

Documents become connected ideas, with a clear trail back to their sources.

## Working together

- Upload documents and review the proposed plan.
- Browse related [[concepts/research]] and keep useful context connected.
- Ask questions, or use **AI Edit** to propose a change.

> Review proposals before applying them to your shared wiki.

## A little structure goes a long way

| Topic | Purpose |
| --- | --- |
| Research | Evidence and sources |
| Concepts | Connected knowledge |

```text
knowledge → context → understanding
```
"""


def _set_initial_preference(page, mode="auto", location=None):
    value = json.dumps({"mode": mode, "location": location})
    page.add_init_script(f"localStorage.setItem('{_KEY}', JSON.stringify({value}));")


def _open_appearance(page):
    page.locator("[data-appearance-open]:visible").first.click()
    expect(page.locator("#prism-appearance")).to_be_visible()


def _theme(page, expected):
    expect(page.locator("html")).to_have_attribute("data-theme", expected)


def _no_horizontal_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def _text_contrast_ratios(page, selectors):
    # Measure the settled palette, not interpolated colors during a theme change.
    page.wait_for_function("""selectors => selectors.every(selector => {
        const element = document.querySelector(selector);
        getComputedStyle(element).color;
        return element.getAnimations().every(animation => animation.playState === 'finished');
    })""", arg=selectors)
    return page.evaluate("""selectors => {
        const luminance = rgb => {
            const values = rgb.match(/[\\d.]+/g).slice(0,3).map(Number).map(v => {
                v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
            });
            return values[0]*0.2126 + values[1]*0.7152 + values[2]*0.0722;
        };
        return selectors.map(selector => {
            const s = getComputedStyle(document.querySelector(selector));
            const a = luminance(s.color), b = luminance(s.backgroundColor);
            return (Math.max(a,b)+0.05)/(Math.min(a,b)+0.05);
        });
    }""", selectors)


@pytest.mark.parametrize("instant, expected", [
    ("2013-03-05T04:30:00Z", "dark"),
    ("2013-03-05T04:39:00Z", "light"),
    ("2013-03-05T15:42:00Z", "light"),
    ("2013-03-05T15:51:00Z", "dark"),
])
def test_auto_matches_reference_sunrise_and_sunset(page, instant, expected):
    # SunCalc's published fixture: sunrise 04:34:56Z, sunset 15:46:57Z.
    # Position-based and daily transit approximations differ slightly; bracket by 4 min.
    # https://github.com/mourner/suncalc/blob/v1.9.0/test.js
    _set_initial_preference(page, location=_KYIV)
    page.clock.install(time=datetime.fromisoformat(instant.replace("Z", "+00:00")))
    page.goto("/")
    _theme(page, expected)


@pytest.mark.parametrize("start, minutes, expected", [
    ("2013-03-05T04:30:00Z", 10, "light"),
    ("2013-03-05T15:42:00Z", 10, "dark"),
])
def test_open_page_switches_without_refresh(page, start, minutes, expected):
    _set_initial_preference(page, location=_KYIV)
    page.clock.install(time=datetime.fromisoformat(start.replace("Z", "+00:00")))
    page.goto("/")
    _theme(page, "dark" if expected == "light" else "light")
    page.clock.fast_forward(minutes * 60000)
    _theme(page, expected)


@pytest.mark.parametrize("instant, latitude, longitude, expected", [
    ("2026-06-21T00:00:00Z", 89.9, 0, True),  # polar day, even at midnight
    ("2026-12-21T12:00:00Z", 89.9, 0, False),  # polar night, even at noon
    ("2026-12-21T00:00:00Z", -89.9, 0, True),
    ("2026-06-21T12:00:00Z", -89.9, 0, False),
    ("2026-03-20T00:00:00Z", 0, 179.9, True),  # date-line east and west agree
    ("2026-03-20T00:00:00Z", 0, -179.9, True),
    ("2026-03-20T12:00:00Z", 0, 179.9, False),
    ("2026-03-20T12:00:00Z", 0, 0, True),  # zero is a valid coordinate
])
def test_solar_extremes(page, instant, latitude, longitude, expected):
    page.goto("/")
    assert page.evaluate(
        "([date, lat, lon]) => PrismSolar.isDaylight(new Date(date), lat, lon)",
        [instant, latitude, longitude],
    ) is expected


def test_solar_uses_absolute_time_across_dst(page):
    page.goto("/")
    assert page.evaluate("""() => {
        const daylight = date => PrismSolar.isDaylight(new Date(date), 40.71, -74.01);
        return daylight('2026-03-08T07:00:00Z') === daylight('2026-03-08T03:00:00-04:00')
            && daylight('2026-11-01T06:00:00Z') === daylight('2026-11-01T01:00:00-05:00');
    }""")


def test_no_location_follows_device_without_requesting_permission(page):
    page.add_init_script("navigator.geolocation.getCurrentPosition = () => { throw new Error('Unexpected location request'); };")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.emulate_media(color_scheme="dark")
    page.goto("/")
    _theme(page, "dark")
    _open_appearance(page)
    expect(page.locator("#prism-theme-status")).to_contain_text("No location saved")
    page.emulate_media(color_scheme="light")
    _theme(page, "light")
    assert not errors


@pytest.mark.parametrize("saved", ["bad-json", '{"mode":"invalid","location":{"latitude":91,"longitude":0}}'])
def test_invalid_saved_preferences_fall_back_safely(page, saved):
    page.add_init_script(f"localStorage.setItem('{_KEY}', {json.dumps(saved)});")
    page.emulate_media(color_scheme="light")
    page.goto("/")
    _theme(page, "light")
    expect(page.locator("html")).to_have_attribute("data-theme-mode", "auto")


def test_restricted_storage_does_not_break_appearance(page):
    # The rest of the application expects storage; isolate only the theme's key.
    page.add_init_script(f"""(() => {{
        const get = Storage.prototype.getItem, set = Storage.prototype.setItem;
        Storage.prototype.getItem = function(key) {{ if (key === '{_KEY}') throw new Error('blocked'); return get.call(this, key); }};
        Storage.prototype.setItem = function(key, value) {{ if (key === '{_KEY}') throw new Error('blocked'); return set.call(this, key, value); }};
    }})();""")
    page.goto("/")
    _open_appearance(page)
    page.select_option("#prism-theme-mode", "dark")
    _theme(page, "dark")
    page.click("#prism-theme-close")
    expect(page.locator("#login-form")).to_be_visible()


def test_location_is_optional_validated_and_forgettable(page):
    page.clock.install(time=datetime(2026, 3, 20, 12, tzinfo=timezone.utc))
    page.emulate_media(color_scheme="dark")
    page.goto("/")
    _open_appearance(page)
    page.fill("#prism-latitude", "91")
    page.fill("#prism-longitude", "0")
    page.get_by_role("button", name="Save location", exact=True).click()
    assert page.evaluate(f"localStorage.getItem('{_KEY}')") is None
    page.fill("#prism-latitude", "0")
    page.get_by_role("button", name="Save location", exact=True).click()
    _theme(page, "light")
    saved = page.evaluate(f"JSON.parse(localStorage.getItem('{_KEY}'))")
    assert saved["location"] == {"latitude": 0, "longitude": 0}
    page.click("#prism-clear-location")
    _theme(page, "dark")
    assert page.evaluate(f"JSON.parse(localStorage.getItem('{_KEY}')).location") is None


@pytest.mark.parametrize("granted", [True, False])
def test_device_location_requires_click_and_handles_denial(page, granted):
    callback = "success({coords:{latitude:40.712812,longitude:-74.006023}})" if granted else "failure({code:1})"
    page.add_init_script(f"""window.locationRequests = 0;
        navigator.geolocation.getCurrentPosition = (success, failure) => {{
            window.locationRequests++; {callback};
        }};""")
    page.goto("/")
    assert page.evaluate("window.locationRequests") == 0
    _open_appearance(page)
    page.click("#prism-device-location")
    assert page.evaluate("window.locationRequests") == 1
    if granted:
        saved = page.evaluate(f"JSON.parse(localStorage.getItem('{_KEY}')).location")
        assert saved == {"latitude": 40.71, "longitude": -74.01}
    else:
        expect(page.locator("#prism-location-message")).to_contain_text("Location was not shared")
        assert page.evaluate(f"localStorage.getItem('{_KEY}')") is None


def test_manual_choice_persists_between_main_admin_and_tabs(logged_in_page):
    page = logged_in_page
    _open_appearance(page)
    page.select_option("#prism-theme-mode", "dark")
    page.click("#prism-theme-close")
    page.reload()
    _theme(page, "dark")
    page.goto("/admin")
    _theme(page, "dark")
    other = page.context.new_page()
    try:
        other.goto("/")
        _theme(other, "dark")
        _open_appearance(page)
        page.select_option("#prism-theme-mode", "light")
        _theme(other, "light")
        page.click("#prism-theme-close")
        page.emulate_media(color_scheme="dark")
        _theme(page, "light")
    finally:
        other.close()


def test_resuming_a_sleeping_tab_rechecks_daylight(page):
    _set_initial_preference(page, location=_KYIV)
    page.clock.install(time=datetime(2013, 3, 5, 12, tzinfo=timezone.utc))
    page.goto("/")
    _theme(page, "light")
    page.clock.set_system_time(datetime(2013, 3, 5, 22, tzinfo=timezone.utc))
    page.evaluate("window.dispatchEvent(new Event('focus'))")
    _theme(page, "dark")


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_wiki_and_ai_edit_layouts_remain_usable(logged_in_page, mode, width, tmp_path):
    page = logged_in_page
    page.set_viewport_size({"width": width, "height": 1000})
    _open_appearance(page)
    page.select_option("#prism-theme-mode", mode)
    page.click("#prism-theme-close")
    page.route("**/api/wiki/concepts/prism-demo.md", lambda route: route.fulfill(
        status=200, content_type="text/plain", body=_ARTICLE))
    page.evaluate("loadWikiPage('concepts/prism-demo.md')")
    expect(page.locator("#page-content h1")).to_have_text("A shared knowledge space")
    assert page.locator("#page-content li").count() == 3
    _no_horizontal_overflow(page)
    page.screenshot(path=str(tmp_path / f"wiki-{mode}-{width}.png"), full_page=True)
    page.click("#btn-ai-edit")
    expect(page.locator("#modal-ai-edit")).to_be_visible()
    controls = page.locator("#modal-ai-edit .modal-content").bounding_box()
    assert controls["x"] >= 0 and controls["x"] + controls["width"] <= width + 1
    page.screenshot(path=str(tmp_path / f"ai-edit-{mode}-{width}.png"), full_page=True)
    page.locator('#modal-ai-edit button[data-close="modal-ai-edit"]').click()
    expect(page.locator("#modal-ai-edit")).not_to_be_visible()


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_admin_theme_and_dialog_layout(logged_in_page, mode, tmp_path):
    page = logged_in_page
    page.goto("/admin")
    _open_appearance(page)
    page.select_option("#prism-theme-mode", mode)
    page.click("#prism-theme-close")
    expect(page.locator(".tabs")).to_be_visible()
    _no_horizontal_overflow(page)
    page.screenshot(path=str(tmp_path / f"admin-{mode}.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="+ Add User", exact=True).click()
    expect(page.locator("#cu-email")).to_be_visible()
    _no_horizontal_overflow(page)
    bounds = page.locator("#modal-create-user .modal").bounding_box()
    assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 391
    page.screenshot(path=str(tmp_path / f"admin-dialog-{mode}.png"), full_page=True)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_primary_text_contrast_and_keyboard_access(page, mode):
    _set_initial_preference(page, mode)
    page.goto("/")
    ratios = _text_contrast_ratios(page, ['#login-form .btn-primary', '#login-username'])
    assert all(ratio >= 4.5 for ratio in ratios)
    page.locator(".prism-login-theme").focus()
    page.keyboard.press("Enter")
    expect(page.locator("#prism-appearance")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#prism-appearance")).not_to_be_visible()
    expect(page.locator(".prism-login-theme")).to_be_focused()


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_prism_accents_are_distinct_and_readable(logged_in_page, mode, width, tmp_path):
    page = logged_in_page
    page.set_viewport_size({"width": width, "height": 1000})
    # Keep a transition active long enough to exercise the measurement's wait.
    page.add_style_tag(content="#btn-writer { transition-duration: 1s; }")
    _open_appearance(page)
    page.select_option("#prism-theme-mode", mode)
    page.click("#prism-theme-close")
    page.click("#btn-home")
    expect(page.locator("#welcome-screen")).to_be_visible()
    accents = page.locator(".step-card").evaluate_all(
        "cards => cards.map(card => getComputedStyle(card).borderTopColor)"
    )
    assert len(set(accents)) == 3
    ratios = _text_contrast_ratios(page, [
        '#btn-upload', '#btn-welcome-upload', '#btn-writer',
        '.step-card:first-child .step-num', '.step-card:nth-of-type(3) .step-num',
        '.step-card:last-child .step-num',
    ])
    assert all(ratio >= 4.5 for ratio in ratios), ratios
    assert page.locator('meta[name="theme-color"]').get_attribute("content") == page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()"
    )
    _no_horizontal_overflow(page)
    page.screenshot(path=str(tmp_path / f"prism-spectrum-{mode}-{width}.png"), full_page=True)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_writer_notifications_and_login_surfaces(logged_in_page, mode, width, tmp_path):
    page = logged_in_page
    page.set_viewport_size({"width": width, "height": 1000})
    _open_appearance(page)
    page.select_option("#prism-theme-mode", mode)
    page.click("#prism-theme-close")
    page.evaluate("""draft => {
        state.writerDraft = draft;
        state.writerFilename = 'shared-knowledge.md';
        state.writerDraftReady = true;
        enterWriterView([]);
    }""", _ARTICLE)
    expect(page.locator("#writer-draft-preview h1")).to_have_text("A shared knowledge space")
    expect(page.locator("#btn-writer-ingest")).to_be_enabled()
    _no_horizontal_overflow(page)
    page.screenshot(path=str(tmp_path / f"writer-{mode}-{width}.png"), full_page=True)
    page.evaluate("""() => _applyServerNotifications([
        {id: 99991, type: 'recalib_done', title: 'Recalibration complete', body: 'Your wiki is up to date.'},
        {id: 99992, type: 'pending_review', title: 'Review your document', link: 'sample.pdf'},
    ])""")
    page.click("#btn-bell")
    expect(page.locator("#notif-panel")).not_to_have_class("hidden")
    # Wait for its existing opening transition, then check the complete panel bounds.
    expect(page.locator("#notif-body")).to_contain_text("Recalibration complete")
    page.wait_for_function("""() => {
        const panel = document.querySelector('#notif-panel').getBoundingClientRect();
        const header = document.querySelector('header').getBoundingClientRect();
        return panel.top >= header.bottom && panel.left >= 0 && panel.right <= innerWidth;
    }""")
    page.screenshot(path=str(tmp_path / f"notifications-{mode}-{width}.png"), full_page=True)
    page.click("#btn-bell")
    # Return to login without changing the existing signout/draft workflow.
    page.evaluate("showLoginScreen()")
    _theme(page, mode)
    page.screenshot(path=str(tmp_path / f"login-{mode}-{width}.png"), full_page=True)


@pytest.mark.parametrize("width", [1440, 1000, 390])
@pytest.mark.parametrize("writer", [False, True])
def test_chat_header_is_compact_without_clipping_controls(logged_in_page, width, writer, tmp_path):
    page = logged_in_page
    page.set_viewport_size({"width": width, "height": 1000})
    if writer:
        page.evaluate("startNewWriterDraft()")
    title = page.locator("#chat-panel-title-text")
    expect(title).to_have_text("Writer Conversation" if writer else "Ask the Knowledge Base")
    header = page.locator(".chat-panel-header")
    header.scroll_into_view_if_needed()
    header.screenshot(path=str(tmp_path / f"compact-header-{width}-{'writer' if writer else 'chat'}.png"))
    bounds = header.bounding_box()
    assert bounds["height"] <= 44
    controls = ["#chat-panel-title-text", "#btn-new-chat"]
    if writer:
        controls.append("#writer-mode-badge")
    for selector in controls:
        control = page.locator(selector).bounding_box()
        assert control["x"] >= bounds["x"]
        assert control["x"] + control["width"] <= bounds["x"] + bounds["width"]
        assert control["y"] >= bounds["y"]
        assert control["y"] + control["height"] <= bounds["y"] + bounds["height"]
    button = page.locator("#btn-new-chat")
    expect(button).to_be_enabled()
    button_bounds = button.bounding_box()
    assert button_bounds["width"] >= 24 and button_bounds["height"] >= 24


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_graph_theme_keeps_sandbox_topology_and_navigation(logged_in_page, mode, tmp_path):
    from app.services.graph import WikiGraph

    graph = WikiGraph()
    graph._adj = {"concepts/a.md": {"sources/b.md"}, "sources/b.md": set()}
    graph._meta = {
        "concepts/a.md": {"title": "Connected idea", "tags": [], "entities": []},
        "sources/b.md": {"title": "Source document", "tags": [], "entities": []},
    }
    data = graph.as_dict()
    html = graph.generate_html()
    assert graph.as_dict() == data
    page = logged_in_page
    page.route("**/api/ops/graph", lambda route: route.fulfill(json=data))
    page.route("**/api/ops/graph/html", lambda route: route.fulfill(content_type="text/html", body=html))
    page.route("**/api/wiki/concepts/a.md", lambda route: route.fulfill(content_type="text/plain", body=_ARTICLE))
    _open_appearance(page)
    page.select_option("#prism-theme-mode", mode)
    page.click("#prism-theme-close")
    page.evaluate("openGraphModal()")
    frame_element = page.locator("#graph-frame")
    expect(frame_element).to_have_attribute("sandbox", "allow-scripts")
    frame = frame_element.element_handle().content_frame()
    frame.wait_for_function("typeof network !== 'undefined' && typeof nodes !== 'undefined' && nodes.length === 2")
    expect(frame.locator("html")).to_have_attribute("data-theme", mode)
    assert frame.evaluate("edges.length") == 1
    assert frame.evaluate("nodes.get('concepts/a.md').label") == "Connected idea"
    assert frame.evaluate("nodes.get('concepts/a.md').font.color") == frame.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--text').trim()"
    )
    page.screenshot(path=str(tmp_path / f"graph-{mode}.png"), full_page=True)

    next_mode = "light" if mode == "dark" else "dark"
    page.evaluate("""([key, mode]) => {
        localStorage.setItem(key, JSON.stringify({mode, location:null}));
        window.dispatchEvent(new StorageEvent('storage', {key}));
    }""", [_KEY, next_mode])
    expect(frame.locator("html")).to_have_attribute("data-theme", next_mode)
    assert frame.evaluate("nodes.length") == 2 and frame.evaluate("edges.length") == 1
    frame.evaluate("network.emit('click', {nodes:['concepts/a.md']})")
    expect(page.locator("#modal-graph")).not_to_be_visible()
    expect(page.locator("#page-content h1")).to_have_text("A shared knowledge space")

"""Math rendering uses local fixtures, never model calls or user documents."""
import json

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.frontend


def _render(page, markdown):
    page.evaluate("source => { document.querySelector('#page-content').innerHTML = renderMd(source); }", markdown)
    return page.locator("#page-content")


@pytest.mark.parametrize("formula, display", [
    (r"$E = mc^2$", False),
    (r"\(a_1 + b_2\)", False),
    ("$$\n\\frac{a}{b}\n$$", True),
    (r"\[\sum_{i=1}^{n} i\]", True),
])
def test_wiki_renders_latex_formulas(logged_in_page, formula, display):
    page = logged_in_page
    page.route("**/api/wiki/concepts/math.md", lambda route: route.fulfill(
        content_type="text/plain", body=f"# Formula\n\n{formula}\n"))
    page.evaluate("loadWikiPage('concepts/math.md')")
    expect(page.locator("#page-content .katex")).to_have_count(1)
    expect(page.locator("#page-content .katex-display")).to_have_count(int(display))
    expect(page.locator("#page-content .katex-error")).to_have_count(0)


@pytest.mark.parametrize("markdown, count", [
    (r"Before $a_1 + b_2$ and \(\sqrt{x}\) after.", 2),
    ("Text\n\n$$\n\\begin{aligned}a&=b+c\\\\d&=e\\end{aligned}\n$$\n\nAfter", 1),
    ("```math\n\\frac{1}{2}\n```", 1),
    ("- First $a^2$\n- Second \\(b^2\\)", 2),
    ("| Variable | Value |\n| --- | --- |\n| $x$ | $\\frac{1}{2}$ |", 2),
    (r"$\text{cost: \$5} + x$", 1),
])
def test_math_survives_markdown_tokenization(page, markdown, count):
    page.goto("/")
    content = _render(page, markdown)
    expect(content.locator(".katex")).to_have_count(count)
    expect(content.locator(".math-fallback, .katex em")).to_have_count(0)
    # DOMPurify intentionally drops annotation/semantics; the accessible MathML
    # expression itself must survive, without relaxing its sanitizer policy.
    expect(content.locator("math")).to_have_count(count)
    assert all(content.locator("math").all_text_contents())


@pytest.mark.parametrize("markdown", [
    r"Costs $5 and $10, or $12.50 per month.",
    r"Escaped \$x\$ and \\(x\\) stay literal.",
    "`$x^2$` and `\\(x\\)`\n\n```text\n$$x$$\n```",
    "    $$x$$\n",
    "<code>$x$</code>\n\n<pre>\\(x\\)</pre>",
])
def test_currency_escapes_and_code_are_not_math(page, markdown):
    page.goto("/")
    content = _render(page, markdown)
    expect(content.locator(".katex, .math-fallback")).to_have_count(0)


def test_markdown_and_wiki_links_are_preserved(page):
    page.goto("/")
    content = _render(page, "---\ntitle: Hidden\n---\n# Heading\n\n**Bold** and $x^2$. See [[concepts/test.md|Related]].")
    expect(content.locator("h1")).to_have_text("Heading")
    expect(content.locator("strong")).to_have_text("Bold")
    expect(content.locator(".wiki-link")).to_have_attribute("data-query", "test")
    expect(content.locator(".katex")).to_have_count(1)
    expect(content).not_to_contain_text("title: Hidden")


@pytest.mark.parametrize("formula", [r"$\notARealCommand{x}$", r"$\frac{1}{$", r"$\def\a{\a}\a$"])
def test_invalid_math_falls_back_without_breaking_neighboring_content(page, formula):
    page.goto("/")
    content = _render(page, formula + "\n\n**Still readable** and $x^2$.")
    expect(content.locator(".math-fallback")).to_have_text(formula)
    expect(content.locator("strong")).to_have_text("Still readable")
    expect(content.locator(".katex")).to_have_count(1)


def test_math_cannot_inject_html_or_load_external_resources(page):
    page.goto("/")
    content = _render(page, r"""<img src=x onerror="window.mathAttack=true"><script>window.mathAttack=true</script>

$\href{javascript:alert(1)}{click}$
$\includegraphics{https://invalid.example/track.png}$
$\htmlClass{evil}{x}$
$\unknown{<img src=x onerror=alert(1)>}$
[bad](javascript:alert(1))
""")
    expect(content.locator("script, [onerror], a[href^='javascript:'], .evil")).to_have_count(0)
    expect(content.locator(".katex img, .math-fallback img")).to_have_count(0)
    assert page.evaluate("window.mathAttack !== true")


def test_math_macros_do_not_leak_between_formulas(page):
    page.goto("/")
    content = _render(page, r"$\gdef\custom{x}\custom$ then $\custom$")
    expect(content.locator(".katex")).to_have_count(1)
    expect(content.locator(".math-fallback")).to_have_text(r"$\custom$")


def test_unavailable_math_library_keeps_source_and_markdown_readable(page):
    page.route("**/vendor/katex/katex.min.js", lambda route: route.abort())
    page.goto("/")
    content = _render(page, r"**Formula:** \(\frac{1}{2}\)")
    expect(content.locator(".math-fallback")).to_have_text(r"\(\frac{1}{2}\)")
    expect(content.locator("strong")).to_have_text("Formula:")


def test_unavailable_sanitizer_fails_closed(page):
    page.route("**/dompurify@3/**", lambda route: route.abort())
    page.goto("/")
    content = _render(page, r'<img src=x onerror="alert(1)"> $x^2$')
    expect(content.locator("img, .katex")).to_have_count(0)
    expect(content.locator("pre")).to_contain_text("$x^2$")


def test_partial_stream_can_be_rerendered_without_duplicate_math(page):
    page.goto("/")
    for partial in ["Answer: $", "Answer: $\\frac{1}", "Answer: $\\frac{1}{2}"]:
        content = _render(page, partial)
        expect(content.locator(".katex")).to_have_count(0)
        expect(content).to_contain_text("Answer:")
    for _ in range(3):
        content = _render(page, r"Answer: $\frac{1}{2}$")
        expect(content.locator(".katex")).to_have_count(1)


def test_writer_and_chat_share_formula_rendering(logged_in_page):
    page = logged_in_page
    draft = r"# Draft" + "\n\n" + r"\[\frac{a}{b}\]"
    page.evaluate("draft => { state.writerDraft = draft; enterWriterView([]); }", draft)
    expect(page.locator("#writer-draft-preview .katex-display")).to_have_count(1)
    assert page.evaluate("state.writerDraft") == draft
    page.evaluate("appendChatBubble('assistant', 'Result: $x^2$')")
    expect(page.locator("#chat-messages .katex")).to_have_count(1)


def test_ai_edit_preview_renders_math_but_apply_preserves_markdown(logged_in_page):
    page = logged_in_page
    original = "# Formula\n\n$x$"
    proposed = "# Formula\n\n" + r"\[\frac{a}{b}\]"
    saved = []

    def wiki(route):
        if route.request.method == "PUT":
            saved.append(json.loads(route.request.post_data)["content"])
            route.fulfill(json={"path": "concepts/formula.md", "updated": True})
        else:
            route.fulfill(content_type="text/plain", body=original)

    page.route("**/api/wiki/concepts/formula.md", wiki)
    events = [{"type": "chunk", "text": proposed},
              {"type": "done", "full_content": proposed, "scope": "page", "heading": None}]
    page.route("**/api/ops/wiki/edit/stream", lambda route: route.fulfill(
        content_type="text/event-stream", body="".join(f"data: {json.dumps(event)}\n\n" for event in events)))
    page.evaluate("loadWikiPage('concepts/formula.md')")
    page.click("#btn-ai-edit")
    page.click("#btn-ai-edit-generate")
    expect(page.locator("#btn-ai-edit-apply")).to_be_enabled()
    expect(page.locator("#ai-edit-preview .katex-display")).to_have_count(1)
    page.click("#btn-ai-edit-apply")
    expect(page.locator("#modal-ai-edit")).not_to_be_visible()
    assert saved == [proposed]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_equations_use_local_fonts_and_scroll_without_clipping(logged_in_page, theme, tmp_path):
    page = logged_in_page
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)
    formula = r"\[\frac{-b\pm\sqrt{b^2-4ac}}{2a} = " + " + ".join(["x_i"] * 35) + r"\]"
    page.route("**/api/wiki/concepts/long-math.md", lambda route: route.fulfill(
        content_type="text/plain", body="# Mathematics\n\n" + r"\[x = \frac{-b\pm\sqrt{b^2-4ac}}{2a}\]" + "\n\n## Wide equation\n\n" + formula))
    page.evaluate("loadWikiPage('concepts/long-math.md')")
    math = page.locator("#page-content .katex-display").last
    expect(math).to_be_visible()
    page.evaluate("document.fonts.ready")
    assert page.evaluate("document.fonts.check('16px KaTeX_Main')")
    assert math.evaluate("el => getComputedStyle(el).overflowX") == "auto"
    assert math.evaluate("el => el.scrollWidth > el.clientWidth")
    page.locator("#page-content").screenshot(path=str(tmp_path / f"math-{theme}.png"))
    # Check the first visible glyph rather than depending on KaTeX's internal
    # layout class names, which vary between releases.
    assert math.evaluate("""el => {
        const text = document.createTreeWalker(el.querySelector('.katex-html'), NodeFilter.SHOW_TEXT).nextNode();
        const range = document.createRange();
        range.selectNodeContents(text);
        return range.getBoundingClientRect().left >= el.getBoundingClientRect().left - 1;
    }""")
    assert math.evaluate("el => { el.scrollLeft = 100; return el.scrollLeft > 0; }")
    math.evaluate("el => { el.scrollLeft = 0; }")

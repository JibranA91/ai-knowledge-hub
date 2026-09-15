/* Recognize math before Markdown can consume LaTeX escapes and underscores. */
(() => {
  if (typeof marked === 'undefined') return;

  function escapedAt(source, index) {
    let slashes = 0;
    while (index > 0 && source[--index] === '\\') slashes++;
    return slashes % 2 === 1;
  }

  function mathToken(source, block) {
    const offset = block ? (source.match(/^ {0,3}/)?.[0].length || 0) : 0;
    const text = source.slice(offset);
    const open = ['$$', '\\[', '\\(', '$'].find(delimiter => text.startsWith(delimiter));
    if (!open || (block && open !== '$$' && open !== '\\[')) return;
    if (open === '$' && /\s/.test(text[1] || ' ')) return;
    const close = {'$$': '$$', '\\[': '\\]', '\\(': '\\)', '$': '$'}[open];
    let end = text.indexOf(close, open.length);
    while (end !== -1 && escapedAt(text, end)) end = text.indexOf(close, end + close.length);
    if (end === -1) return;
    const formula = text.slice(open.length, end);
    // Single-dollar math follows the usual no-edge-whitespace convention, and
    // a closing dollar followed by a digit is currency ("$5 and $10"), not math.
    if (open === '$' && (/\n|\s$/.test(formula) || /\d/.test(text[end + 1] || ''))) return;
    if (!formula.trim()) return;
    return {
      type: block ? 'mathBlock' : 'mathInline',
      raw: source.slice(0, offset + end + close.length),
      text: formula,
      display: open === '$$' || open === '\\[',
    };
  }

  function renderMath(token) {
    try {
      if (typeof katex === 'undefined' || token.text.length > 16000) throw new Error('Math unavailable');
      return katex.renderToString(token.text, {
        displayMode: token.display, output: 'htmlAndMathml', throwOnError: true,
        trust: false, maxSize: 20, maxExpand: 1000, macros: {},
      });
    } catch {
      // Never insert a KaTeX error message: it may contain unescaped source.
      const fallback = document.createElement('span');
      fallback.className = 'math-fallback';
      fallback.textContent = token.raw;
      return fallback.outerHTML;
    }
  }

  const parser = new marked.Marked({extensions: [
    {
      name: 'mathBlock', level: 'block',
      start: source => source.search(/(?:^|\n) {0,3}(?:\$\$|\\\[)/),
      tokenizer: source => mathToken(source, true), renderer: renderMath,
    },
    {
      name: 'mathInline', level: 'inline',
      start: source => source.search(/\$|\\(?:\(|\[)/),
      tokenizer(source) {
        if (!this.lexer.state.inRawBlock) return mathToken(source, false);
      },
      renderer: renderMath,
    },
    {
      name: 'code',
      renderer: token => token.lang === 'math'
        ? renderMath({text: token.text, raw: token.text, display: true}) : false,
    },
  ]});
  window.renderMarkdownWithMath = source => parser.parse(source);
})();

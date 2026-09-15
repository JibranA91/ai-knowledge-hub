# KaTeX browser runtime

Vendored from the `katex` npm package, version **0.18.7** (MIT; see `LICENSE`).
The download was verified against the npm registry's SHA-512 integrity value.
Only the minified JavaScript/CSS, matching fonts, and license are included.

Source: https://github.com/KaTeX/KaTeX/tree/v0.18.7
Package: https://registry.npmjs.org/katex/0.18.7

To update, choose a reviewed version, verify the registry integrity hash, and
replace the JavaScript, CSS, and fonts together. Run the Markdown-math browser
tests, including unsafe input and unavailable-library cases. Keep `trust: false`
and bounded expansion/size settings; output still passes through DOMPurify.

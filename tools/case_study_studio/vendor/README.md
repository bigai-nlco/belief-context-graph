# Local Markdown parser

`marked.mjs` is the unmodified ESM build of **Marked 18.0.5**, copied from the locally installed `marked` package (`lib/marked.esm.js`). Its original license is in `marked.LICENSE`.

Source: https://github.com/markedjs/marked

The app loads this file from its own static server, not a CDN. No build or npm install is needed to run the Studio. Application-specific HTML escaping, link restrictions and image handling live in `../markdown.mjs`; do not use the parser's default raw-HTML renderer on imported reports.

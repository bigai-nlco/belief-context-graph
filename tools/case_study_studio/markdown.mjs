import {Marked, Renderer} from './vendor/marked.mjs';

const escape = value => String(value ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));

function safeHref(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:', 'mailto:'].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

// Imported reports are untrusted. Never pass raw HTML through the renderer.
// Only explicit, safe links may leave the workbench; images do not auto-load.
const markdown = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    html({text}) { return escape(text); },
    link({href, title, tokens}) {
      const text = this.parser.parseInline(tokens);
      const url = safeHref(href);
      return url
        ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer"${title ? ` title="${escape(title)}"` : ''}>${text}</a>`
        : text;
    },
    image({href, text}) {
      const label = escape(text || '图片');
      const url = safeHref(href);
      return url
        ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
        : label;
    },
    table(token) {
      return `<div class="markdown-table" tabindex="0" role="region" aria-label="结果表格">${Renderer.prototype.table.call(this, token)}</div>`;
    },
  },
});

/** Render for display only; saved reports, exports and source IDs stay unchanged. */
export function renderMarkdown(value) {
  const text = String(value ?? '');
  try { return markdown.parse(text); }
  catch { return `<pre class="markdown-fallback">${escape(text)}</pre>`; }
}

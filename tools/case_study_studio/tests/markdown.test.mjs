import assert from 'node:assert/strict';
import {test} from 'node:test';
import {renderMarkdown} from '../markdown.mjs';

test('renders headings, nested lists, emphasis, quotes and fenced code', () => {
  const html = renderMarkdown([
    '# 报告', '', '## 结论', '', '**重点**、*说明*、~~旧信息~~与 `node_id`。',
    '', '> 引用证据', '', '1. 第一项', '   - 子项', '2. 第二项',
    '', '```js', 'if (a < b) console.log("<script>");', '```',
  ].join('\n'));
  for (const tag of ['h1', 'h2', 'strong', 'em', 'del', 'blockquote', 'ol', 'ul']) {
    assert.match(html, new RegExp(`<${tag}>`));
  }
  assert.match(html, /<code>node_id<\/code>/);
  assert.match(html, /<pre><code class="language-js">/);
  assert.match(html, /a &lt; b/);
  assert.doesNotMatch(html, /<script>/);
});

test('renders GFM tables, alignment and escaped pipes in a scrollable region', () => {
  const html = renderMarkdown('| 指标 | 值 |\n| :--- | ---: |\n| a\\|b | **42** |');
  assert.match(html, /class="markdown-table" tabindex="0" role="region"/);
  assert.match(html, /<th align="left">指标<\/th>/);
  assert.match(html, /<td align="right"><strong>42<\/strong><\/td>/);
  assert.match(html, /a\|b/);
});

test('raw HTML and active attributes remain literal text', () => {
  const html = renderMarkdown([
    '<script>window.__xss=true</script>', '',
    '<img src=x onerror="window.__xss=true">', '',
    '<iframe src="https://example.test"></iframe>', '',
    '<svg onload=alert(1)><foreignObject><img src=x></foreignObject></svg>', '',
    'Text <a data-action="delete">danger</a>.',
  ].join('\n'));
  assert.doesNotMatch(html, /<(script|img|iframe|svg|foreignObject)\b/i);
  assert.doesNotMatch(html, /<a data-action=/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /&lt;img src=x onerror=&quot;/);
});

test('only explicit http, https and mailto links are clickable', () => {
  for (const href of ['javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'jav&#x61;script:alert(1)',
    'data:text/html,payload', 'file:///etc/passwd', '//example.test/pixel', './delete', '#app']) {
    assert.equal(renderMarkdown(`[visible](${href})`).trim(), '<p>visible</p>', href);
  }
  const html = renderMarkdown('[**来源**](https://example.test/?a=1&b=2 "source") [邮件](mailto:test@example.test)');
  assert.match(html, /href="https:\/\/example.test\/\?a=1&amp;b=2"/);
  assert.match(html, /target="_blank" rel="noopener noreferrer" title="source"/);
  assert.match(html, /<strong>来源<\/strong>/);
  assert.match(html, /href="mailto:test@example.test"/);
});

test('URL and title characters cannot introduce new attributes', () => {
  const html = renderMarkdown('[source](<https://example.test/"><svg/onload=alert(1)>>)');
  assert.doesNotMatch(html, /<svg\b|<[^>]+\sonload=/);
  const title = renderMarkdown('[source](https://example.test/ "\' onmouseover=\'alert(1)")');
  assert.match(title, /title="&#39; onmouseover=&#39;alert\(1\)"/);
});

test('images become links and never issue automatic resource requests', () => {
  const html = renderMarkdown('![图示](https://example.test/pixel.png)\n\n![blocked](data:image/svg+xml,malicious)');
  assert.doesNotMatch(html, /<img|src=|data:image/);
  assert.match(html, />图示<\/a>/);
  assert.match(html, /<p>blocked<\/p>/);
});

test('reference links, autolinks and task lists render without editable controls', () => {
  const html = renderMarkdown('[资料][source]\n\n[source]: https://example.test\n\n<https://example.test/other>\n\n- [x] 已核实\n- [ ] 待核实');
  assert.match(html, /href="https:\/\/example.test\/"/);
  assert.match(html, /href="https:\/\/example.test\/other"/);
  assert.equal((html.match(/disabled="" type="checkbox"/g) || []).length, 2);
});

test('plain text, empty reports and encoded HTML remain readable', () => {
  assert.equal(renderMarkdown(null), '');
  assert.equal(renderMarkdown('结论：42。').trim(), '<p>结论：42。</p>');
  assert.equal(renderMarkdown('&lt;script&gt;').trim(), '<p>&lt;script&gt;</p>');
});

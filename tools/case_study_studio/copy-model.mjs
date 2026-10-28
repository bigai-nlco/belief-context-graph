// Text-only presentation overrides. Annotation values and evidence never change.
export const COPY_VERSION = 1;
export const KINDS = ['text', 'placeholder', 'title', 'aria-label'];
export const copyKey = entry => JSON.stringify([entry.kind, entry.original]);
export const emptyPublished = () => ({format:'bcg-ui-copy', version:COPY_VERSION, revision:0, entries:[]});
export const emptyDraft = revision => ({format:'bcg-ui-copy-draft', version:COPY_VERSION, baseRevision:revision, entries:[]});
export const slots = text => [...text.matchAll(/\{\{n\d+\}\}/g)].map(x=>x[0]).sort();

export function validateEntry(entry) {
  if (!entry || !KINDS.includes(entry.kind) ||
      typeof entry.original !== 'string' || !entry.original.trim() ||
      entry.original.length > 12000 || typeof entry.value !== 'string' ||
      entry.value.length > 12000) throw new Error('文案条目无效或长度超过限制。');
  if (JSON.stringify(slots(entry.original)) !== JSON.stringify(slots(entry.value))) {
    throw new Error('请保留动态数字占位符，例如 {{n1}}，数量与编号都不能改变。');
  }
  return {kind:entry.kind, original:entry.original, value:entry.value};
}

export function validateCopy(document, draft = false) {
  const revision = draft ? 'baseRevision' : 'revision';
  if (document?.format !== (draft ? 'bcg-ui-copy-draft' : 'bcg-ui-copy') ||
      document.version !== COPY_VERSION || !Number.isSafeInteger(document[revision]) ||
      document[revision] < 0 || !Array.isArray(document.entries) ||
      document.entries.length > 4000) throw new Error('不支持的文案文件格式。');
  const seen = new Set();
  const entries = document.entries.map(raw => {
    const e = validateEntry(raw), key = copyKey(e);
    if (seen.has(key)) throw new Error('文案文件包含重复条目。');
    seen.add(key); return e;
  });
  return {format:document.format, version:COPY_VERSION, [revision]:document[revision], entries};
}

export function mergedCopy(published, draft) {
  published = validateCopy(published);
  if (draft) {
    draft = validateCopy(draft, true);
    if (draft.baseRevision !== published.revision) throw new Error('正式文案已更新。请先导出当前草稿，再基于最新版本核对修改。');
  }
  const values = new Map(published.entries.map(e => [copyKey(e), e]));
  for (const e of draft?.entries || []) values.set(copyKey(e), e);
  return values;
}

export function publishDraft(published, draft) {
  const entries = [...mergedCopy(published, draft).values()]
    .filter(e=>e.original!==e.value).sort((a,b)=>copyKey(a).localeCompare(copyKey(b)));
  return {format:'bcg-ui-copy', version:COPY_VERSION, revision:published.revision+1, entries};
}

export function describeText(text, dynamic = false) {
  const [, before, body, after] = text.match(/^(\s*)([\s\S]*?)(\s*)$/);
  const values = [];
  const original = dynamic ? body.replace(/\d+(?:,\d{3})*(?:\.\d+)?/g, value => {
    values.push(value); return `{{n${values.length}}}`;
  }) : body;
  return {original, before, after, values};
}

export function renderCopy(entry, description) {
  const text = entry?.value ?? description.original;
  return description.before + text.replace(/\{\{n(\d+)\}\}/g, (_, i)=>description.values[Number(i)-1] ?? `{{n${i}}}`) + description.after;
}

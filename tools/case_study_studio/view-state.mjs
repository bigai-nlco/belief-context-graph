// Browser navigation only. Kept separate from annotations and workspace exports.
export const VIEW_STORAGE_KEY = 'bcg-case-study-view-v1';
export const DEFAULT_VIEW = Object.freeze({
  form: 'basic', tab: 'trajectory', source: '', context: '', graphView: 'full',
  graphFull: '', graphTurn: '', event: '', node: '', query: '', filter: '', page: 0,
  evidenceScroll: 0, formScroll: 0,
});

export function normalizeView(value) {
  const input = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const view = {...DEFAULT_VIEW};
  for (const [key, allowed] of Object.entries({
    form: ['basic', 'events', 'dimensions', 'conclusion'],
    tab: ['trajectory', 'context', 'graph', 'report'], graphView: ['full', 'turns'],
  })) if (allowed.includes(input[key])) view[key] = input[key];
  for (const key of ['source', 'context', 'graphFull', 'graphTurn', 'event', 'node', 'query', 'filter']) {
    if (typeof input[key] === 'string') view[key] = input[key].slice(0, 4000);
  }
  for (const key of ['page', 'evidenceScroll', 'formScroll']) {
    if (Number.isFinite(input[key]) && input[key] >= 0) view[key] = Math.min(Math.floor(input[key]), 10000000);
  }
  return view;
}

export function readViewState(raw, caseIds) {
  const result = {version: 1, active: caseIds[0] || '', cases: Object.create(null)};
  try {
    const saved = JSON.parse(raw);
    if (saved?.version !== 1) return result;
    if (caseIds.includes(saved.active)) result.active = saved.active;
    for (const id of caseIds) {
      if (saved.cases && Object.hasOwn(saved.cases, id)) result.cases[id] = normalizeView(saved.cases[id]);
    }
  } catch { /* A missing or invalid navigation snapshot does not affect case data. */ }
  return result;
}

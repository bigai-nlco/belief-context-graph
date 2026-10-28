import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_VIEW, normalizeView, readViewState} from '../view-state.mjs';

test('new and legacy cases start at basic information and the trajectory', () => {
  assert.equal(DEFAULT_VIEW.form, 'basic');
  assert.equal(DEFAULT_VIEW.tab, 'trajectory');
  for (const value of [null, undefined, '', false, [], 'invalid']) assert.deepEqual(normalizeView(value), {...DEFAULT_VIEW});
  const saved = readViewState(null, ['newest', 'older']);
  assert.equal(saved.active, 'newest');
  assert.deepEqual(Object.keys(saved.cases), []);
});

test('restores the last active case, not the one most recently annotated', () => {
  const raw = JSON.stringify({version: 1, active: 'older', cases: {
    older: {form: 'conclusion', tab: 'context', context: 'call-5', formScroll: 240, evidenceScroll: 125},
    newest: {form: 'events', tab: 'graph', graphView: 'turns', graphTurn: 'turn-9', event: 'E-uuid'},
  }});
  const saved = readViewState(raw, ['newest', 'older']);
  assert.equal(saved.active, 'older');
  assert.equal(saved.cases.older.form, 'conclusion');
  assert.equal(saved.cases.older.context, 'call-5');
  assert.equal(saved.cases.older.formScroll, 240);
  assert.equal(saved.cases.newest.event, 'E-uuid');
  assert.equal(saved.cases.newest.graphTurn, 'turn-9');
});

test('unknown, corrupt and deleted navigation data falls back without affecting cases', () => {
  for (const raw of ['broken', 'null', '{"version":99}', '[]']) {
    assert.equal(readViewState(raw, ['exists']).active, 'exists');
  }
  const saved = readViewState(JSON.stringify({version: 1, active: 'deleted', cases: {deleted: {form: 'conclusion'}, exists: {form: 'invalid', tab: 'script', page: -4}}}), ['exists']);
  assert.equal(saved.active, 'exists');
  assert.deepEqual(Object.keys(saved.cases), ['exists']);
  assert.deepEqual(saved.cases.exists, {...DEFAULT_VIEW});
  assert.equal(readViewState(null, []).active, '');
});

test('navigation snapshots whitelist fields and do not copy annotations or unsafe object keys', () => {
  const input = {...DEFAULT_VIEW, form: 'events', query: 'Europa', source: 'researcher', page: 2.8, formScroll: Infinity,
    busy: true, annotation: {reviewer: 'Private reviewer'}, cases: [], extra: 'ignored'};
  const view = normalizeView(input);
  assert.equal(view.page, 2);
  assert.equal(view.formScroll, 0);
  assert.equal(view.query, 'Europa');
  assert.equal(view.source, 'researcher');
  assert.equal(Object.hasOwn(view, 'annotation'), false);
  assert.equal(Object.hasOwn(view, 'busy'), false);
  const saved = readViewState('{"version":1,"active":"__proto__","cases":{"__proto__":{"form":"events"}}}', ['__proto__']);
  assert.equal(Object.getPrototypeOf(saved.cases), null);
  assert.equal(saved.cases.__proto__.form, 'events');
});

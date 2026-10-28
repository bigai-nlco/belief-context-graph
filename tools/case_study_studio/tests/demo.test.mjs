import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {buildDemoCase, demoLabel, REAL_DEMO_ID} from '../demo.mjs';
import {completion, tokenMetrics, graphViews, agentTrajectories} from '../model.mjs';

const bundle = JSON.parse(await readFile(new URL('../demo-case.json', import.meta.url), 'utf8'));

test('bundled Demo is a complete real execution with initially empty ratings', async () => {
  const c = await buildDemoCase(bundle);
  assert.equal(c.taskId, REAL_DEMO_ID);
  assert.equal(demoLabel(c), '真实运行 Demo');
  assert.equal(c.metrics.status, 'completed');
  assert.equal(c.metrics.usage.agent_model_calls, 22);
  assert.equal(c.metrics.usage.graph_fallback_calls, 0);
  assert.equal(c.graphs.length, 47);
  assert.equal(graphViews(c).full.length, 1);
  assert.equal(graphViews(c).turns.length, 46);
  assert.equal(graphViews(c).full[0].label, 'agent/bcg/state.json');
  assert.equal(c.contexts.length, 22);
  assert.equal(agentTrajectories(c).length, 1);
  assert.equal(agentTrajectories(c)[0].name, 'stateful_react');
  assert.equal(agentTrajectories(c)[0].records.find(r=>r.index===12).raw.turn, 4);
  assert.equal(c.annotation.result, '');
  assert.equal(c.annotation.events.length, 0);
  assert.equal(completion(c).completed, 0);
  assert.match(c.report, /Europa Clipper/);
  assert.match(c.report, /https:\/\/.*esa.int/);
  assert.deepEqual(tokenMetrics(c), {
    agent: 123949, graph: 38753, combined: 162702,
    graphScope: '含最终建图（源文件统计口径）',
  });
});

test('tutorial references point to the recorded injection, beliefs and action', async () => {
  const c = await buildDemoCase(bundle);
  const ctx = c.contexts.find(x => x.call === 4);
  for (const id of [3, 10, 11, 12]) assert(ctx.nodeIds.includes(id));
  assert.equal(ctx.raw.messages_before, 13);
  assert.equal(ctx.raw.messages_after, 7);
  const trace = c.sources.find(x => x.label.endsWith('react_agent.jsonl'));
  const action = trace.records.find(x => x.index === 12);
  assert.equal(action.raw.turn, 4);
  assert.match(action.tools, /Europa Clipper launched October 14 2024/);
  const final = c.graphs.find(x => x.label === 'agent/bcg/state.json');
  assert.equal(final.nodes.length, 74);
  assert.equal(final.relations.length, 53);
  assert(final.relations.some(e => e.from_id === 10 && e.to_id === 3));
});

test('each snapshot has its own historical values, complete final evidence remains available', async () => {
  const c = await buildDemoCase(bundle);
  const timeline = c.graphs.filter(x => x.index);
  assert.equal(new Set(timeline.map(x => x.id)).size, timeline.length);
  assert.equal(timeline[0].nodes.length, 0);
  assert.equal(timeline.at(-1).nodes.length, 74);
  const final = c.graphs.find(x => x.label === 'agent/bcg/state.json');
  assert(final.nodes.some(n => n.supporting_excerpts?.length));
  assert.equal(c.sources.filter(x => x.kind === 'graph_snapshots').length, 1);
});

test('Demo rejects tampered source content and incomplete runs', async () => {
  const bad = structuredClone(bundle);
  bad.files[0].text += ' ';
  await assert.rejects(buildDemoCase(bad), /校验失败/);
  await assert.rejects(buildDemoCase({...bundle, provenance: {status:'error'}}), /不完整/);
  const path = structuredClone(bundle);
  path.files[0].path = '../.env';
  await assert.rejects(buildDemoCase(path), /来源文件无效/);
});

test('old fictional demos retain their distinct identity and label', () => {
  assert.equal(demoLabel({demo:true,taskId:'DEMO-01'}), '虚构教学示例');
  assert.equal(demoLabel({mode:'default'}), 'default');
  assert.notEqual(REAL_DEMO_ID, 'DEMO-01');
});

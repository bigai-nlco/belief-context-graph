import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ingestFiles, contextTurn, contextBeliefText, eventBeliefContext, eventEvidencePresentation, eventFromTrajectory, recordPresentation, workspaceExport, restoreWorkspace} from '../model.mjs';

const file = (path, rows) => ({path, digest: path, text: rows.map(JSON.stringify).join('\n')});
const audit = (call, text = 'Injected B12') => ({call, system_memory: text, selected_node_ids: [12]});
const trace = (agent, turn = 4) => [
  {t: 'start', agent_id: agent, user_message: 'Task'},
  {t: 'llm', turn: turn - 1, content: 'Previous answer'},
  {t: 'result', turn: turn - 1, name: 'lookup', result: 'Previous result'},
  {t: 'llm', turn, content: '**Checking the evidence.**', thinking: 'Use B12.', tool_calls: [{name: 'web_search', args: {q: 'targeted query'}}]},
  {t: 'result', turn, name: 'web_search', result: 'First result'},
  {t: 'result', turn, name: 'web_search', result: 'Second result'},
  {t: 'llm', turn: turn + 1, content: 'Final answer'},
];

test('shows only recorded beliefs and relations without rewriting or reconstructing them', () => {
  const graph = '## Beliefs\n- [B12] A known claim. (confidence 0.8)\n\n## Relations\n- [B12] depends_on [B11]';
  const ctx = {text: '<context_blocks_guide>Fixed guide</context_blocks_guide>\n<belief_graph>\n' + graph + '\n</belief_graph>'};
  const before = structuredClone(ctx);
  assert.equal(contextBeliefText(ctx), graph);
  assert.deepEqual(ctx, before);
  assert.equal(contextBeliefText({text: 'Legacy unwrapped input'}), null);
  assert.equal(contextBeliefText({text: '<context_blocks_guide>Fixed guide only</context_blocks_guide>'}), null);
  assert.equal(contextBeliefText({text: '<belief_graph>incomplete'}), null);
  assert.equal(contextBeliefText({text: '<belief_graph></belief_graph>'}), '');
  assert.equal(contextBeliefText({text: '<belief_graph>A</belief_graph><belief_graph>B</belief_graph>'}), null);
});

test('joins actual call numbers rather than the audit row or adjacent model calls', () => {
  const c = ingestFiles([file('agent/bcg/requests.jsonl', [audit(4)]), file('agent/trajectories/react_agent.jsonl', trace('main'))]);
  const before = structuredClone(c), paired = contextTurn(c, c.contexts[0]);
  assert.equal(paired.status, 'matched');
  assert.equal(paired.call, '4');
  assert.equal(paired.assistant.index, 4);
  assert.equal(recordPresentation(paired.assistant).thinking, 'Use B12.');
  assert.equal(recordPresentation(paired.assistant).toolCalls[0].arguments.q, 'targeted query');
  assert.deepEqual(paired.results.map(r => r.text), ['First result', 'Second result']);
  assert.deepEqual(c, before, 'Pairing must not alter evidence, IDs, annotations, or saved navigation');
});

test('pairs early calls with empty memory and final answers with no tool results', () => {
  const c = ingestFiles([
    file('bcg/requests.jsonl', [{call: 1, graph_chars: 0, selected_node_ids: []}, audit(2)]),
    file('trajectories/main.jsonl', [{t: 'llm', turn: 1, tool_calls: [{name: 'lookup', args: {id: 1}}]},
      {t: 'result', turn: 1, result: 'Evidence'}, {t: 'llm', turn: 2, content: '# Final answer'}]),
  ]);
  assert.equal(c.contexts[0].text, '');
  assert.equal(contextTurn(c, c.contexts[0]).assistant.call, 1);
  const last = contextTurn(c, c.contexts[1]);
  assert.equal(last.assistant.text, '# Final answer');
  assert.deepEqual(last.results, []);
});

test('keeps matching outputs visible when graph injection falls back', () => {
  const c = ingestFiles([file('requests.jsonl', [{call: 4, selected_node_ids: [], fallback: 'graph unavailable'}]), file('trace.jsonl', trace('main'))]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'matched');
  assert.equal(c.contexts[0].fallback, 'graph unavailable');
});

test('matches the same Agent directory when multiple Agents use call 4', () => {
  const c = ingestFiles([
    file('researcher-a/bcg/requests.jsonl', [audit(4)]),
    file('researcher-b/trajectories/main.jsonl', trace('B')),
    file('researcher-a/trajectories/main.jsonl', trace('A')),
  ]);
  const paired = contextTurn(c, c.contexts[0]);
  assert.equal(paired.status, 'matched');
  assert.equal(paired.agent.name, 'A');
  assert.equal(paired.assistant.id, c.sources[2].records[3].id);
});

test('does not join an audit to the only imported but differently owned trajectory', () => {
  const c = ingestFiles([file('A/bcg/requests.jsonl', [audit(4)]), file('B/trajectories/main.jsonl', trace('B'))]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'unmatched-agent');
});

test('uses explicit Agent metadata and does not resolve ambiguity from coincidental call availability', () => {
  const c = ingestFiles([
    file('requests.jsonl', [audit(4), {...audit(4), agent_id: 'B'}, {...audit(4), agent_id: 'missing'}]),
    file('a.jsonl', trace('A', 8)), file('b.jsonl', trace('B', 4)),
  ]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'ambiguous-agent');
  assert.equal(contextTurn(c, c.contexts[1]).agent.name, 'B');
  assert.equal(contextTurn(c, c.contexts[2]).status, 'unmatched-agent');
});

test('separates duplicate call numbers within one trajectory rather than assuming an attempt', () => {
  const c = ingestFiles([file('requests.jsonl', [audit(4)]), file('trace.jsonl', [...trace('main'), {t: 'llm', turn: 4, content: 'Another attempt'}])]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'ambiguous-call');
});

test('JSON/JSONL alternative exports are not duplicate Assistant outputs', () => {
  const rows = trace('main');
  const c = ingestFiles([file('requests.jsonl', [audit(4)]), file('trace.jsonl', rows), {path: 'trace.json', digest: 'json', text: JSON.stringify(rows)}]);
  const paired = contextTurn(c, c.contexts[0]);
  assert.equal(paired.status, 'matched');
  assert.equal(paired.results.length, 2);
});

test('never equates Graph turn or row indices with Agent calls', () => {
  const c = ingestFiles([
    file('graph-contexts/log.jsonl', [{view: 'compact', text: 'B12', streamTurnIndex: 4, call: 4},
      {view: 'compact', text: 'B12', streamTurnIndex: 12, local_call: 4, agent_id: 'main'}]),
    file('requests.jsonl', [{system_memory: 'No call field'}, audit(9)]), file('trace.jsonl', trace('main')),
  ]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'missing-call');
  assert.equal(contextTurn(c, c.contexts[1]).status, 'matched');
  assert.equal(contextTurn(c, c.contexts[2]).status, 'missing-call');
  assert.equal(contextTurn(c, c.contexts[3]).status, 'missing-output');
});

test('reports missing output instead of presenting a tool result as the Assistant', () => {
  const c = ingestFiles([file('requests.jsonl', [audit(4)]), file('trace.jsonl', [{t: 'result', turn: 4, result: 'Tool-only evidence'}])]);
  assert.equal(contextTurn(c, c.contexts[0]).status, 'missing-output');
  const noTrace = ingestFiles([file('requests.jsonl', [audit(4)])]);
  assert.equal(contextTurn(noTrace, noTrace.contexts[0]).status, 'missing-trajectory');
});

test('a full model request can use its explicit call id to match the same trajectory', () => {
  const c = ingestFiles([file('model-io.jsonl', [{call_id: '4', payload: {messages: [{role: 'system', content: 'Memory'}]}}]), file('trace.jsonl', trace('main'))]);
  assert.equal(c.contexts[0].kind, 'model_request');
  assert.equal(contextTurn(c, c.contexts[0]).assistant.call, 4);
});

test('all 22 calls of the real Demo match, including no-graph calls and the final answer', () => {
  const bundle = JSON.parse(readFileSync(new URL('../demo-case.json', import.meta.url), 'utf8'));
  const c = ingestFiles(bundle.files);
  const before = JSON.stringify(c);
  for (const ctx of c.contexts) {
    const paired = contextTurn(c, ctx);
    assert.equal(paired.status, 'matched', ctx.label);
    assert.equal(Number(paired.assistant.call), Number(ctx.call));
    assert(paired.results.every(r => Number(r.call) === Number(ctx.call)));
  }
  assert.equal(c.contexts.length, 22);
  assert.equal(contextTurn(c, c.contexts.at(-1)).results.length, 0);
  assert.match(contextTurn(c, c.contexts.at(-1)).assistant.text, /Europa Clipper/);
  assert.equal(JSON.stringify(c), before);
  const restored = restoreWorkspace(workspaceExport([c]))[0];
  assert.equal(contextTurn(restored, restored.contexts[3]).assistant.id, contextTurn(c, c.contexts[3]).assistant.id);
});

const graphInput = text => '<context_blocks_guide>Fixed guide</context_blocks_guide>\n<belief_graph>' + text + '</belief_graph>';

test('critical-process Assistant and tool citations use their producing call injection', () => {
  const c = ingestFiles([
    file('agent/bcg/requests.jsonl', [audit(4, graphInput('B12: Known evidence.')), audit(5, graphInput('B99: Later evidence.'))]),
    file('agent/trajectories/main.jsonl', trace('main')),
  ]);
  const paired = contextTurn(c, c.contexts[0]);
  for (const record of [paired.assistant, ...paired.results]) {
    const event = eventFromTrajectory(c, 'record:' + record.id);
    const beforeCase = JSON.stringify(c), beforeEvent = JSON.stringify(event);
    const result = eventBeliefContext(c, event);
    assert.equal(result.status, 'matched');
    assert.equal(result.call, '4');
    assert.equal(result.text, 'B12: Known evidence.');
    assert.equal(result.context.id, c.contexts[0].id);
    assert.doesNotMatch(result.text, /context_blocks_guide|B99/);
    assert.equal(JSON.stringify(c), beforeCase);
    assert.equal(JSON.stringify(event), beforeEvent);
  }
});

test('selected and older excerpts still get the same injection without expanding their quote', () => {
  const c = ingestFiles([file('requests.jsonl', [audit(4, graphInput('B12: Evidence.'))]), file('trace.jsonl', trace('main'))]);
  const record = contextTurn(c, c.contexts[0]).assistant;
  const event = eventFromTrajectory(c, 'record:' + record.id, 'Checking the evidence.');
  c.annotation.events.push(event);
  assert.deepEqual(eventEvidencePresentation(c, event), {kind:'excerpt', text:'Checking the evidence.'});
  assert.equal(eventBeliefContext(c, event).text, 'B12: Evidence.');
  const restored = restoreWorkspace(workspaceExport([c]))[0];
  assert.equal(eventBeliefContext(restored, restored.annotation.events[0]).text, 'B12: Evidence.');
  assert.deepEqual(restored.annotation, c.annotation);
  assert.equal(eventBeliefContext(c, {...event, sourceRef:'record:missing'}).status, 'missing-record');
  assert.equal(eventBeliefContext(c, {...event, sourceRef:''}).status, 'missing-record');
  assert.equal(eventBeliefContext(null, undefined).status, 'missing-record');
});

test('critical-process evidence never borrows the same numbered call from another Agent', () => {
  const c = ingestFiles([
    file('A/bcg/requests.jsonl', [audit(4, graphInput('A evidence'))]),
    file('B/bcg/requests.jsonl', [audit(4, graphInput('B evidence'))]),
    file('A/trajectories/main.jsonl', trace('A')), file('B/trajectories/main.jsonl', trace('B')),
  ]);
  for (const context of c.contexts) {
    const paired = contextTurn(c, context);
    const event = eventFromTrajectory(c, 'record:' + paired.assistant.id);
    const result = eventBeliefContext(c, event);
    assert.equal(result.agent.id, paired.agent.id);
    assert.equal(result.text, contextBeliefText(context));
  }
  c.contexts = c.contexts.slice(0, 1);
  const b = c.sources.find(s => s.records.some(r => r.raw?.agent_id === 'B')).records.find(r => r.role === 'assistant' && r.call === 4);
  assert.equal(eventBeliefContext(c, {sourceRef:'record:' + b.id}).status, 'missing-context');
});

test('duplicate copies of injected beliefs are deduplicated but conflicting versions are not guessed', () => {
  const c = ingestFiles([
    file('requests.jsonl', [audit(4, graphInput('B12'))]),
    file('model-io.jsonl', [{call_id:4,payload:{messages:[{role:'system',content:graphInput('B12')}]}}]),
    file('trace.jsonl', trace('main')),
  ]);
  const event = {sourceRef:'record:' + contextTurn(c, c.contexts[0]).assistant.id};
  assert.equal(eventBeliefContext(c, event).status, 'matched');
  c.contexts[1].text = graphInput('B13');
  assert.equal(eventBeliefContext(c, event).status, 'ambiguous-context');
  assert.equal(eventBeliefContext(c, event).text, null);
});

test('missing and empty event injections do not use guides or final graph snapshots', () => {
  const c = ingestFiles([
    file('requests.jsonl', [{call:3,graph_chars:0,selected_node_ids:[]}, audit(4,'<context_blocks_guide>Guide only</context_blocks_guide>')]),
    file('trace.jsonl', trace('main')),
    {path:'state.json',digest:'state',text:JSON.stringify({nodes:[{id:999,belief:'Final graph is not injected evidence'}],relations:[]})},
  ]);
  const event = call => ({sourceRef:'record:' + c.sources[1].records.find(r => r.role === 'assistant' && r.call === call).id});
  assert.equal(eventBeliefContext(c, event(3)).status, 'empty');
  assert.equal(eventBeliefContext(c, event(4)).status, 'missing-graph');
  assert.equal(eventBeliefContext(c, event(4)).text, null);
  assert.equal(eventBeliefContext(c, event(5)).status, 'missing-context');
  assert.equal(eventBeliefContext(c, {sourceRef:'record:' + c.sources[1].records[0].id}).status, 'not-agent-call');
});

test('a cited call does not use a coincidentally numbered Graph turn', () => {
  const c = ingestFiles([
    file('graph-contexts/log.jsonl', [{view:'compact',text:graphInput('B12'),streamTurnIndex:4,call:4}]),
    file('trace.jsonl', trace('main')),
  ]);
  const event = {sourceRef:'record:' + c.sources[1].records[3].id};
  assert.equal(eventBeliefContext(c, event).status, 'missing-context');
});

test('real Demo process citations recover exactly the graph displayed by the per-call view', () => {
  const bundle = JSON.parse(readFileSync(new URL('../demo-case.json', import.meta.url), 'utf8'));
  const c = ingestFiles(bundle.files), before = JSON.stringify(c);
  for (const context of c.contexts) {
    const paired = contextTurn(c, context);
    for (const record of [paired.assistant, ...paired.results]) {
      const event = {sourceRef:'record:' + record.id, quote:'Saved excerpt'};
      const result = eventBeliefContext(c, event);
      assert.equal(result.context.id, context.id);
      assert.equal(result.call, String(record.call));
      assert.equal(result.text, contextBeliefText(context));
      assert.equal(event.quote, 'Saved excerpt');
    }
  }
  assert.equal(JSON.stringify(c), before);
});

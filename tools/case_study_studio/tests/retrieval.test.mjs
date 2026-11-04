import test from 'node:test';
import assert from 'node:assert/strict';
import {ingestFiles} from '../model.mjs';
import {DEFAULT_VIEW, normalizeView} from '../view-state.mjs';
import {retrievalRounds, retrievalSnapshot, retrievalRows, renderRetrieval, graphLayout} from '../retrieval.mjs';

const file=(path,rows)=>({path,digest:path,text:rows.map(JSON.stringify).join('\n')});
const snapshot=(call=4,turns=7,graphId='g',confidence=.2)=>({call,problem_id:graphId,snapshot_scope:'acknowledged_update',snapshot:{problem_id:graphId,n_turns_ingested:turns,nodes:[
  {id:1,belief:'An earlier claim',confidence}, {id:2,belief:'A bridge',confidence:0}, {id:3,belief:'Unselected evidence',confidence:.9},
],relations:[{id:8,from_id:1,to_id:2,type:'depends_on'},{id:9,from_id:2,to_id:3,type:'supplements'}]}});
const audit=(call=4,turns=7,graphId='g')=>({call,problem_id:graphId,graph_n_turns_ingested:turns,graph_chars:123,selected_node_ids:[1,2],selected_relation_ids:[8],selector_result:{strategy:'query_pcst',selected_prizes:{1:.1,2:0},node_scores:{1:{confidence:.2,relevance:.4,prize:.1},2:{confidence:0,relevance:0,prize:0},3:{confidence:.9,relevance:.3,prize:.276}}}});
const load=(audits=[audit()],snapshots=[snapshot()])=>ingestFiles([file('agent/bcg/requests.jsonl',audits),file('agent/bcg/graph_snapshots.jsonl',snapshots)]);

test('large snapshot raw evidence is preserved without persisting an additional pretty-printed copy',()=>{
  const original=snapshot(),c=load([audit()],[original]);
  const record=c.sources.find(source=>source.kind==='graph_snapshots').records[0];
  assert.deepEqual(record.raw,original);
  assert.equal(record.text,'');
  assert.deepEqual(c.graphs[0].nodes,original.snapshot.nodes);
});

test('matches exact acknowledged snapshot and preserves original confidence instead of final values',()=>{
  const c=load([audit()],[snapshot(),snapshot(5,9,'g',.95)]),ctx=c.contexts[0], before=JSON.stringify(c);
  const {graph,status}=retrievalSnapshot(c,ctx);
  assert.equal(status,'matched');assert.equal(graph.nodes[0].confidence,.2);
  const rows=retrievalRows(graph,ctx);
  assert.equal(rows.length,3);assert.equal(rows[2].selected,false);assert.equal(rows[2].prize,.276);
  assert.equal(rows[1].prize,0);assert.equal(rows[1].confidence,0);
  assert.equal(JSON.stringify(c),before);
});

test('reuses unchanged snapshot on later calls but rejects future or later final flush snapshots',()=>{
  const c=load([audit(5)],[snapshot(),snapshot(5,10)]);
  assert.equal(retrievalSnapshot(c,c.contexts[0]).graph.nTurns,7);
  const future=load([audit(3)],[snapshot()]);
  assert.equal(retrievalSnapshot(future,future.contexts[0]).status,'missing');
});

test('never substitutes the final graph or uses a snapshot from a different graph generation',()=>{
  const c=ingestFiles([file('agent/bcg/requests.jsonl',[audit()]),{path:'agent/bcg/state.json',digest:'s',text:JSON.stringify(snapshot())}]);
  assert.equal(retrievalSnapshot(c,c.contexts[0]).status,'missing');
  const reset=load([audit(5,7,'new')],[snapshot(4,7,'old')]);
  assert.equal(retrievalSnapshot(reset,reset.contexts[0]).status,'missing');
});

test('early legacy calls reuse only explicitly unchanged acknowledged graphs',()=>{
  const early=[1,2,3].map(call=>({call,problem_id:'g',graph_turns_written:call===1?2:0,selected_node_ids:[],selected_relation_ids:[]}));
  const c=load(early,[snapshot(1,2)]);
  assert.equal(retrievalSnapshot(c,c.contexts[2]).status,'matched');
  c.contexts[1].raw.graph_turns_written=1;
  assert.equal(retrievalSnapshot(c,c.contexts[2]).status,'missing');
  c.contexts.splice(1,1);
  assert.equal(retrievalSnapshot(c,c.contexts[1]).status,'missing');
});

test('does not mix Agent graphs with identical call and node numbers',()=>{
  const c=ingestFiles([
    file('a/bcg/requests.jsonl',[audit(4,7,'a')]),file('b/bcg/requests.jsonl',[audit(4,7,'b')]),
    file('a/bcg/graph_snapshots.jsonl',[snapshot(4,7,'a',.2)]),file('b/bcg/graph_snapshots.jsonl',[snapshot(4,7,'b',.9)]),
  ]);
  assert.equal(retrievalSnapshot(c,c.contexts[0]).graph.nodes[0].confidence,.2);
  assert.equal(retrievalSnapshot(c,c.contexts[1]).graph.nodes[0].confidence,.9);
});

test('reports ambiguous histories and corrupt selection IDs rather than highlighting a guessed graph',()=>{
  const c=load([audit()],[snapshot(),snapshot(4,7,'g',.99)]);
  assert.equal(retrievalSnapshot(c,c.contexts[0]).status,'ambiguous');
  const bad=load([{...audit(),selected_node_ids:[999]}]);
  assert.equal(retrievalSnapshot(bad,bad.contexts[0]).status,'mismatch');
  const duplicate=load([audit()],[snapshot(),snapshot()]);
  assert.equal(retrievalSnapshot(duplicate,duplicate.contexts[0]).status,'matched');
});

test('legacy audits show recorded prizes without reconstructing absent relevance or rejected prizes',()=>{
  const a=audit();delete a.selector_result.node_scores;
  const c=load([a]),ctx=c.contexts[0],rows=retrievalRows(retrievalSnapshot(c,ctx).graph,ctx);
  assert.equal(rows[0].prize,.1);assert.equal(rows[0].relevance,undefined);
  assert.equal(rows[1].prize,0);assert.equal(rows[2].prize,undefined);
  assert.match(renderRetrieval(c,{...DEFAULT_VIEW}),/旧记录未保存全部候选评分/);
});

test('replay fills missing scores with labels and preserves original prizes, selections and evidence',()=>{
  const a=audit();delete a.selector_result.node_scores;
  a.selector_result.score_replay={kind:'retrospective_recomputation',node_scores:{
    1:{confidence:.2,relevance:.4,prize:.15},2:{confidence:0,relevance:0,prize:0},3:{confidence:.9,relevance:.3,prize:.276}
  },comparison:{matches_within_tolerance:false}};
  const c=load([a]),ctx=c.contexts[0],before=JSON.stringify(c);
  const rows=retrievalRows(retrievalSnapshot(c,ctx).graph,ctx);
  assert.equal(rows[0].prize,.1);assert.equal(rows[0].prizeOrigin,'recorded');
  assert.equal(rows[0].relevance,.4);assert.equal(rows[0].relevanceOrigin,'replay');
  assert.equal(rows[1].prize,0);assert.equal(rows[1].prizeOrigin,'recorded');
  assert.equal(rows[2].prize,.276);assert.equal(rows[2].prizeOrigin,'replay');
  assert.equal(rows[2].selected,false);
  const html=renderRetrieval(c,{...DEFAULT_VIEW,retrievalNode:'3'});
  assert.match(html,/0\.276（补算）/);assert.match(html,/补算 prize 与部分原记录有差异/);
  assert.equal(JSON.stringify(c),before);
});

test('unscored calls and confidence-only retrieval do not claim missing query scores',()=>{
  const early=load([{call:4,problem_id:'g',graph_n_turns_ingested:7,selected_node_ids:[],selected_relation_ids:[]}]);
  assert.match(renderRetrieval(early,{...DEFAULT_VIEW}),/未计算/);
  const a=audit();a.selector_result.strategy='confidence_pcst';
  for(const score of Object.values(a.selector_result.node_scores))score.relevance=null;
  assert.match(renderRetrieval(load([a]),{...DEFAULT_VIEW}),/不适用/);
});

test('full-graph ablation shows an uncapped graph rather than missing retrieval scores',()=>{
  const a={...audit(),retrieval_strategy:'all',graph_chars_budget:null,selected_node_ids:[1,2,3],selected_relation_ids:[8,9],
    selector_result:{strategy:'all',node_ids:[1,2,3],relation_ids:[8,9]}};
  const c=load([a]),html=renderRetrieval(c,{...DEFAULT_VIEW});
  assert.match(html,/BCG w\/o Retrieval/);
  assert.match(html,/跳过检索与候选评分/);
  assert.match(html,/无上限/);
  assert.doesNotMatch(html,/旧记录未保存全部候选评分/);
  assert.equal((html.match(/class="retrieval-node is-selected/g)||[]).length,3);
  assert.equal((html.match(/class="retrieval-edge is-selected/g)||[]).length,2);
  assert(retrievalRows(retrievalSnapshot(c,c.contexts[0]).graph,c.contexts[0]).every(row=>row.scoringSkipped));
});

test('highlight uses actual injected IDs rather than an attempted selection on fallback',()=>{
  const a={...audit(),selected_node_ids:[],selected_relation_ids:[],fallback:'selection_exceeds_render_budget'};
  a.selector_result.node_ids=[1,2];a.selector_result.relation_ids=[8];
  const c=load([a]),html=renderRetrieval(c,{...DEFAULT_VIEW});
  assert.doesNotMatch(html,/retrieval-node is-selected/);
  assert.doesNotMatch(html,/retrieval-edge is-selected/);
  assert.match(html,/回退到原始上下文/);
});

test('renders full graph or selected subgraph with exact edges and escapes untrusted contents',()=>{
  const s=snapshot();s.snapshot.nodes[0].belief='<img src=x onerror=alert(1)>';
  const c=load([audit()],[s]);
  const full=renderRetrieval(c,{...DEFAULT_VIEW});
  assert.equal((full.match(/class="retrieval-node /g)||[]).length,3);
  assert.equal((full.match(/class="retrieval-node is-selected/g)||[]).length,2);
  assert.match(full,/&lt;img/);assert.doesNotMatch(full,/<img src/);
  const selected=renderRetrieval(c,{...DEFAULT_VIEW,retrievalScope:'selected'});
  assert.equal((selected.match(/class="retrieval-node /g)||[]).length,2);
  assert.equal((selected.match(/data-retrieval-edge=/g)||[]).length,1);
});

test('rounds exclude final flush records and keep early no-injection calls',()=>{
  const c=load([{call:1,selected_node_ids:[],graph_chars:0},audit(),{kind:'final_graph_flush',status:'ok'}]);
  assert.deepEqual(retrievalRounds(c).map(ctx=>ctx.call),[1,4]);
});

test('layout preserves all nodes, including isolated nodes, with finite positions',()=>{
  const graph=snapshot().snapshot;graph.nodes.push({id:77,belief:'isolated'});
  const positions=graphLayout(graph,[1,2,3,77]);
  assert.equal(positions.size,4);
  assert([...positions.values()].every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
  assert.equal(graphLayout(graph,[1,2,3,77]),positions);
});

test('restores retrieval navigation without storing evidence as view state',()=>{
  const view=normalizeView({tab:'retrieval',context:'call-4',retrievalScope:'selected',retrievalFilter:'unselected',retrievalLabels:'on',retrievalNode:'3',retrievalQuery:'evidence'});
  assert.equal(view.tab,'retrieval');assert.equal(view.retrievalNode,'3');assert.equal(view.retrievalScope,'selected');
});

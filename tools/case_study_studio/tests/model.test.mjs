import test from 'node:test';
import assert from 'node:assert/strict';
import {DIMENSIONS, IMPROVEMENT_STAGES, DEMO_TITLE, updateDemoTitle, parseFile, ingestFiles, graphViews, agentTrajectories, recordPresentation, eventEvidencePresentation, eventFromTrajectory, newEvent, selectedDimensions, hasNoApplicableDimensions, setDimensionSelected, confirmNoApplicableDimensions, linkedEvents, setDimensionEvent, removeEvent, eventLabel, conclusionObservations, setConclusionFocus, textOf, tokenMetrics, completion, validateReview, workspaceExport, restoreWorkspace, csvSummary, markdownReport} from '../model.mjs';

const file = (path, data) => ({path, digest: `${path}:${JSON.stringify(data).length}`, text: typeof data === 'string' ? data : JSON.stringify(data)});
const result = {benchmark:'test-bench', task_id:'42', mode:'bcg', question:'Find the year.', final_response:'2024', usage:{total:100}, graph_usage:{total:20}};
const graph = {all_nodes:[{id:0,belief:'The project entered operation.',confidence:.8,source:{turn_id:2}},{id:1,belief:'The year was 2024.'}],relations:[{id:0,from_id:1,to_id:0,type:'depends_on'}]};

test('dimension guidance uses statements and retains stable annotation IDs', () => {
  assert.deepEqual(DIMENSIONS.map(d => d.id), ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']);
  for (const d of DIMENSIONS) {
    assert.ok(d.question.length);
    assert.doesNotMatch(d.question, /[?？]|是否/);
  }
});

test('demo title update preserves all evidence, annotations, dates, and review status', () => {
  const c = ingestFiles([file('demo.json', {...result, task_id:'DEMO-01'})]);
  c.demo = true; c.title = '试运行 ≠ 正式运营：一次由 Graph 引导的验证';
  c.annotation.reviewer = 'Reviewer'; c.annotation.status = 'reviewed';
  c.annotation.events.push(newEvent({behavior:'Saved analysis', reference:'call 15'}));
  const before = structuredClone(c);
  assert.equal(updateDemoTitle(c), true);
  assert.deepEqual(c, {...before, title:DEMO_TITLE});
  assert.equal(updateDemoTitle(c), false);

  const restored = restoreWorkspace(workspaceExport([before]));
  assert.equal(restored[0].title, DEMO_TITLE);
  assert.deepEqual(restored[0].annotation, before.annotation);
  assert.equal(before.title, '试运行 ≠ 正式运营：一次由 Graph 引导的验证');
});

test('title update leaves imported cases and customized demo names unchanged', () => {
  for (const c of [
    {demo:false, taskId:'DEMO-01', title:'试运行 ≠ 正式运营：一次由 Graph 引导的验证'},
    {demo:true, taskId:'42', title:'试运行 ≠ 正式运营：一次由 Graph 引导的验证'},
    {demo:true, taskId:'DEMO-01', title:'My custom title'},
  ]) {
    const before = structuredClone(c);
    assert.equal(updateDemoTitle(c), false);
    assert.deepEqual(c, before);
  }
});

test('parses BOM JSON and reports broken JSONL lines without renumbering evidence', () => {
  assert.equal(parseFile('a.json','\uFEFF{"task_id":1}').data.task_id,1);
  const p=parseFile('a.jsonl','{"role":"user"}\nnot-json\n\n{"role":"assistant"}');
  assert.deepEqual(p.data.map(r=>r.line),[1,4]);assert.equal(p.warnings.length,1);
  assert.throws(()=>parseFile('a.jsonl','broken'),/没有可读取/);
});

test('imports metadata, graph, transcript and context as separate evidence', () => {
  const c=ingestFiles([file('42.json',result),file('graph/result.json',graph),file('trajectory.json',{messages:[{role:'user',content:'Find the year.'},{role:'assistant',content:'2024'}]}),file('requests.jsonl',JSON.stringify({call:5,system_memory:'<belief_graph>B0</belief_graph>',selected_node_ids:[0],selected_relation_ids:[],fallback:null}))]);
  assert.equal(c.taskId,'42');assert.equal(c.taskIdVerified,true);assert.equal(c.graphs.length,1);assert.equal(c.contexts.length,1);assert.equal(c.report,'2024');
  assert.deepEqual(c.contexts[0].nodeIds,[0]);assert.equal(c.graphs[0].kind,'uploaded_snapshot');
  assert.deepEqual(tokenMetrics(c),{agent:100,graph:20,combined:120,graphScope:'源文件 graph_usage（请核对是否仅在线）'});
  assert.equal(c.annotation.result,'');assert.equal(completion(c).completed,0);
});

test('current native Agent trajectory JSONL preserves thinking, tools and line positions', () => {
  const data=[{t:'start',task_id:'native',model_name:'model',user_message:'Question'}, {t:'llm',turn:1,content:'Checking',thinking:'Need evidence',tool_calls:[{name:'web_search',args:{query:'Q'}}]}, {t:'result',turn:1,name:'web_search',tool_call_id:'x',result:'Evidence'}];
  const c=ingestFiles([file('native.jsonl',data.map(JSON.stringify).join('\n'))]);
  assert.equal(c.question,'Question');assert.equal(c.model,'model');
  const r=c.sources[0].records;assert.equal(r[1].role,'assistant');assert.equal(r[1].thinking,'Need evidence');assert.match(r[1].tools,/web_search/);assert.equal(r[2].role,'tool');assert.equal(r[2].text,'Evidence');assert.equal(r[2].index,3);
  assert.equal(r[0].title,'任务开始');assert.equal(r[0].text,'Question');assert.equal(r[0].raw.model_name,'model');
});

test('tool-only Assistant records have no synthetic JSON body', () => {
  const raw = {t:'llm', turn:1, content:'', tool_calls:[
    {name:'web_search', args:{q:'first query'}}, {name:'web_search', args:{q:'second query'}},
  ], usage:{prompt_tokens:123}, finish_reason:'tool_calls'};
  const c = ingestFiles([file('tool-only.jsonl', JSON.stringify(raw))]);
  const r = c.sources[0].records[0];
  assert.equal(r.text, ''); assert.equal(r.thinking, '');
  assert.deepEqual(r.toolCalls, [
    {name:'web_search',arguments:{q:'first query'}}, {name:'web_search',arguments:{q:'second query'}},
  ]);
  assert.deepEqual(r.raw, raw);
  assert.doesNotMatch(r.tools, /prompt_tokens|finish_reason/);
});

test('Assistant text is preserved even if it is itself JSON', () => {
  const c = ingestFiles([file('text.json', {messages:[
    {role:'assistant',content:'{"answer": 42}',tool_calls:[]},
    {role:'assistant',content:'I will verify the date.',tool_calls:[{type:'function',function:{name:'web_search',arguments:'{"query":"launch date","limit":5}'}}]},
  ]})]);
  const [final, next] = c.sources[0].records;
  assert.equal(final.text, '{"answer": 42}');
  assert.equal(final.tools, ''); assert.deepEqual(final.toolCalls, []);
  assert.equal(next.text, 'I will verify the date.');
  assert.deepEqual(next.toolCalls, [{name:'web_search',arguments:{query:'launch date',limit:5}}]);
});

test('content blocks separate text, visible reasoning and tool calls', () => {
  const raw = {type:'message_end',message:{role:'assistant',content:[
    {type:'thinking',thinking:'Check the official source.'},
    {type:'text',text:'Checking the date.'},
    {type:'toolCall',name:'web_search',arguments:{query:'official date'}},
    {type:'tool_use',name:'web_fetch',input:{url:'https://example.test'}},
    {type:'function_call',name:'lookup',arguments:'{"id":42}'},
  ]}};
  const c = ingestFiles([file('blocks.jsonl', JSON.stringify(raw))]), r = c.sources[0].records[0];
  assert.equal(r.text, 'Checking the date.');
  assert.equal(r.thinking, 'Check the official source.');
  assert.deepEqual(r.toolCalls.map(c=>c.name), ['web_search','web_fetch','lookup']);
  assert.deepEqual(r.toolCalls[2].arguments, {id:42});
  assert.deepEqual(r.raw, raw);
});

test('duplicate tool representations are not displayed twice, encrypted reasoning is not text', () => {
  const call = {type:'function_call',name:'lookup',arguments:'{"id":42}'};
  const c = ingestFiles([file('mixed.json', {messages:[{
    role:'assistant',content:[call,{type:'reasoning',encrypted_content:'private-bytes',summary:[]}],
    tool_calls:[call],reasoning_content:'Visible reasoning',
  }]})]);
  const r = c.sources[0].records[0];
  assert.equal(r.toolCalls.length, 1); assert.equal(r.text, '');
  assert.equal(r.thinking, 'Visible reasoning');
  assert(!r.text.includes('private-bytes') && !r.thinking.includes('private-bytes'));
});

test('older saved JSON fallbacks are re-presented without changing evidence or annotations', () => {
  const raw = {t:'llm',turn:2,content:'',tool_calls:[{name:'web_search',args:{q:'Q'}}],usage:{total_tokens:123}};
  const c = ingestFiles([file('old.jsonl',JSON.stringify(raw))]), r = c.sources[0].records[0];
  r.text = JSON.stringify(raw, null, 2); delete r.toolCalls;
  c.annotation.events.push(newEvent({sourceRef:'record:'+r.id,quote:'Previously saved quotation'}));
  const before = structuredClone(c);
  const view = recordPresentation(r);
  assert.equal(view.id, r.id); assert.equal(view.index, 1); assert.equal(view.call, 2);
  assert.equal(view.text, ''); assert.equal(view.toolCalls[0].arguments.q, 'Q');
  assert.equal(view.raw, r.raw);
  assert.deepEqual(c, before);
  const restored = restoreWorkspace(workspaceExport([c]))[0];
  assert.equal(recordPresentation(restored.sources[0].records[0]).text, '');
  assert.deepEqual(restored.annotation.events, before.annotation.events);
});

test('empty Assistant output stays empty; argument values are not discarded or invented', () => {
  const c = ingestFiles([file('args.json', {messages:[
    {role:'assistant',content:null,tool_calls:[]},
    {role:'assistant',tool_calls:[
      {name:'broken',arguments:'{"incomplete":'}, {name:'batch',args:['A','B']},
      {name:'empty',args:{}}, {name:'flag',args:false}, {name:'no_args'},
    ]},
  ]})]);
  const [empty, calls] = c.sources[0].records;
  assert.equal(empty.text, ''); assert.equal(empty.thinking, ''); assert.equal(empty.tools, '');
  assert.deepEqual(calls.toolCalls.map(c=>c.arguments), ['{"incomplete":',['A','B'],{},false,undefined]);
});

test('trajectory view excludes reports, graph inputs, graph snapshots and request audits', () => {
  const files=[
    file('result.json',result),file('report.md','Report'),
    file('bcg/state.json',{acknowledged:[{role:'user',content:'Graph-only input'}],snapshot:graph}),
    file('belief_graph.jsonl',JSON.stringify(graph)),
    file('requests.jsonl',JSON.stringify({call:1,system_memory:'Memory'})),
    file('model-io.jsonl',JSON.stringify({type:'request',payload:{messages:[{role:'user',content:'Request-only input'}]}})),
    file('unrelated.jsonl',JSON.stringify({status:'completed'})),
  ];
  assert.deepEqual(agentTrajectories(ingestFiles(files)),[]);
  const c=ingestFiles([...files,file('agent/trace.jsonl',[
    {t:'start',role_id:'researcher',task_id:'job-1',user_message:'Original question'},
    {t:'llm',turn:1,content:'Finding evidence'},
    {t:'result',turn:1,name:'web_search',result:'Evidence'},
    {call:1,system_memory:'Interleaved injection audit'},
  ].map(JSON.stringify).join('\n'))]);
  const agents=agentTrajectories(c);
  assert.equal(agents.length,1);assert.equal(agents[0].name,'researcher');
  assert.equal(agents[0].records.length,3);
  assert.equal(agents[0].records[1],c.sources.at(-1).records[1]);
});

test('critical events may only originate from displayed Agent trajectory records', () => {
  const c = ingestFiles([
    file('trace.jsonl', [
      {t:'llm',turn:2,content:'Check the date.',thinking:'Need an official source',tool_calls:[{name:'web_search',args:{q:'launch date'}}]},
      {call:2,system_memory:'Interleaved audit',selected_node_ids:[]},
    ].map(JSON.stringify).join('\n')),
    file('bcg/state.json',{acknowledged:[{role:'assistant',content:'Graph input only'}],snapshot:graph}),
    file('requests.jsonl',JSON.stringify({call:2,system_memory:'Injected memory'})),
    file('report.md','Final report'),
  ]);
  const record = agentTrajectories(c)[0].records[0], ref = 'record:' + record.id;
  const before = structuredClone(c), event = eventFromTrajectory(c,ref);
  assert.equal(event.sourceRef,ref); assert.equal(event.label,'记录 1 · assistant');
  assert.equal(event.agent, agentTrajectories(c)[0].name);
  for (const key of ['need', 'behavior', 'impact', 'fault', 'strength']) assert.equal(event[key], '', 'attribution must remain the reviewer\'s judgment');
  assert.match(event.reference,/trace.jsonl.*记录 1 · call 2/);
  assert.match(event.quote,/Need an official source[\s\S]+Check the date[\s\S]+web_search/);
  assert.deepEqual(c,before,'creation helper must not mutate the case or existing annotations');
  for (const invalid of ['',undefined,'report','context:'+c.contexts[0].id,'node:'+c.graphs[0].id+':0','record:unknown',
    ...c.sources.flatMap(s=>s.records).filter(r=>r.id!==record.id).map(r=>'record:'+r.id)]) {
    assert.equal(eventFromTrajectory(c,invalid),null,invalid);
  }
  assert.equal(eventFromTrajectory(null,ref),null);
});

test('trajectory citations retain source identity, selected excerpts and bounded full quotes', () => {
  const c = ingestFiles([file('trace.json',{messages:[
    {role:'assistant',content:'The launch date is verified.'},
    {role:'assistant',content:'x'.repeat(13000)},
  ]})]);
  const [short,long] = agentTrajectories(c)[0].records;
  assert.equal(eventFromTrajectory(c,'record:'+short.id,'launch date').quote,'launch date');
  assert.equal(eventFromTrajectory(c,'record:'+short.id,'unrelated text').quote,short.text);
  assert.match(eventFromTrajectory(c,'record:'+long.id).quote,/\[摘录截断；完整内容见来源记录\]$/);
});

test('legacy non-trajectory events survive restore and exports unchanged', () => {
  const c = ingestFiles([file('result.json',result)]);
  c.annotation.events.push(newEvent({sourceRef:'report',reference:'Original report',quote:'Old saved quotation',behavior:'Saved review'}));
  c.annotation.events.push(newEvent({reference:'Handwritten source',behavior:'Manual review'}));
  const before = structuredClone(c.annotation.events);
  assert.equal(eventFromTrajectory(c,'report'),null);
  assert.deepEqual(restoreWorkspace(workspaceExport([c]))[0].annotation.events,before);
  assert.match(markdownReport(c),/Old saved quotation/);
});

test('whole-record event evidence presents text, reasoning and tool calls without rewriting the citation', () => {
  const c = ingestFiles([file('trace.jsonl', JSON.stringify({t:'llm', turn:3, content:'## Findings\n\n**A claim**', thinking:'Check **the date**.', tool_calls:[{name:'web_search',args:{q:'year'}}],usage:{prompt_tokens:987654}}))]);
  const r = agentTrajectories(c)[0].records[0], e = eventFromTrajectory(c, 'record:' + r.id);
  const beforeCase = structuredClone(c), beforeEvent = structuredClone(e);
  const view = eventEvidencePresentation(c, e);
  assert.equal(view.kind, 'record');
  assert.equal(view.record.id, r.id);
  assert.equal(view.record.text, '## Findings\n\n**A claim**');
  assert.equal(view.record.thinking, 'Check **the date**.');
  assert.deepEqual(view.record.toolCalls, [{name:'web_search',arguments:{q:'year'}}]);
  assert(!view.record.text.includes('987654'));
  assert.deepEqual(c, beforeCase); assert.deepEqual(e, beforeEvent);
});

test('selected, truncated and detached evidence stays within the saved excerpt', () => {
  const c = ingestFiles([file('trace.json',{messages:[
    {role:'assistant',content:'## Heading\n\n**Selected claim**\n\nOther material'},
    {role:'assistant',content:'x'.repeat(13000)},
  ]})]);
  const [short, long] = agentTrajectories(c)[0].records;
  const selected = eventFromTrajectory(c, 'record:' + short.id, '**Selected claim**');
  assert.deepEqual(eventEvidencePresentation(c, selected), {kind:'excerpt',text:'**Selected claim**'});
  const truncated = eventFromTrajectory(c, 'record:' + long.id);
  assert.deepEqual(eventEvidencePresentation(c, truncated), {kind:'excerpt',text:truncated.quote});
  assert.deepEqual(eventEvidencePresentation(c, {...selected,sourceRef:'record:missing'}), {kind:'excerpt',text:selected.quote});
  assert.deepEqual(eventEvidencePresentation(c, {...selected,quote:''}), {kind:'excerpt',text:''});
  assert.deepEqual(eventEvidencePresentation(null, undefined), {kind:'excerpt',text:''});
});

test('legacy whole JSON citations can use structured rendering while retaining their raw quote', () => {
  const c = ingestFiles([file('trace.json',{messages:[{role:'assistant',content:'A visible claim',tool_calls:[{name:'web_fetch',args:{url:'https://example.test'}}]}]})]);
  const r = agentTrajectories(c)[0].records[0];
  const e = newEvent({sourceRef:'record:' + r.id,quote:JSON.stringify(r.raw,null,2)});
  const saved = structuredClone(e);
  assert.equal(eventEvidencePresentation(c, e).record.text, 'A visible claim');
  assert.deepEqual(e, saved);
});

test('concise process review works without optional details and exports the three core judgments', () => {
  const c = ingestFiles([file('result.json', result), file('agent/trajectory.jsonl', JSON.stringify({role:'assistant',content:'Search again.'}))]);
  const record = agentTrajectories(c)[0].records[0];
  const event = eventFromTrajectory(c, 'record:' + record.id);
  Object.assign(event, {need:'Reuse the known source.', behavior:'Searched for the same fact.', impact:'No new evidence; check retrieval before searching.', fault:'Agent 使用', strength:'明确可见'});
  c.annotation.events.push(event);
  c.annotation.reviewer = 'Reviewer'; c.annotation.result = '部分正确 / 部分达标';
  c.annotation.conclusion.problem = 'Known evidence was not reused.'; c.annotation.conclusion.evidence = '明确可见';
  setDimensionSelected(c, 'A2', true);
  setDimensionEvent(c, 'A2', event.id, true);
  Object.assign(c.annotation.dimensions.A2, {utilization:'未利用', effect:'负向', note:'The same evidence was queried again.'});
  assert.deepEqual(validateReview(c), []);
  assert.equal(completion(c).percent, 100);
  assert.deepEqual(restoreWorkspace(workspaceExport([c]))[0].annotation, c.annotation);
  const md = markdownReport(c);
  assert.match(md, /BCG 应发挥的作用[\s\S]+Reuse the known source/);
  assert.match(md, /实际发生的情况[\s\S]+Searched for the same fact/);
  assert.match(md, /影响与改进线索[\s\S]+No new evidence/);
  assert.doesNotMatch(md, /补充证据检查|\*\*入图情况\*\*|\*\*其他来源与影响因素\*\*/);
});

test('detailed legacy process notes remain unchanged and appear in supplemental exports', () => {
  const c = ingestFiles([file('result.json', result)]);
  const event = newEvent({label:'Legacy review', need:'Old information need', objective:'Old subproblem', original:'Old source', ingestion:'Old graph notes', visible:'Old injection notes', alternatives:'Old external limits', behavior:'Observed action', impact:'Observed impact', strength:'有较强迹象'});
  c.annotation.events.push(event);
  const before = structuredClone(c);
  const restored = restoreWorkspace(workspaceExport([c]))[0];
  const md = markdownReport(restored);
  assert.deepEqual(restored, before);
  assert.deepEqual(c, before);
  for (const key of ['need','objective','original','ingestion','visible','alternatives','behavior','impact']) assert(md.includes(event[key]));
  assert.match(md, /#### 补充证据检查/);
});

test('alternate JSON/JSONL transcripts choose one complete representation and retain citations', () => {
  const records=[{role:'user',content:'Q'},{role:'assistant',content:'A'}];
  const c=ingestFiles([file('agent/trace.json',{messages:records}),file('agent/trace.jsonl',records.map(JSON.stringify).join('\n'))]);
  const before=structuredClone(c);
  assert.equal(agentTrajectories(c).length,1);
  assert.deepEqual(agentTrajectories(c)[0].records,c.sources[1].records);
  assert.deepEqual(c,before);
  const extended=ingestFiles([file('agent/trace.json',{messages:[...records,{role:'tool',content:'More evidence'}]}),file('agent/trace.jsonl',records.map(JSON.stringify).join('\n'))]);
  assert.equal(agentTrajectories(extended)[0].records.length,3);
});

test('multi-agent view uses metadata and does not merge different Agents with matching filenames', () => {
  const docs=[
    file('result.json',result),
    file('leader.json',{task_id:'lead-job',agent_id:'lead',agent_name:'Lead Agent',messages:[{role:'assistant',content:'Delegate'}]}),
    file('child/trace.json',{task_id:'child-job',agent_id:'child',agent_name:'Research Agent',messages:[{role:'assistant',content:'Investigate'}]}),
    file('child/trace.jsonl',[{t:'start',task_id:'child-job',agent_id:'child',agent_name:'Research Agent',user_message:'Research question'},{t:'llm',content:'Investigate'}].map(JSON.stringify).join('\n')),
    file('child/continuation.json',{task_id:'child-job',agent_id:'child',agent_name:'Research Agent',messages:[{role:'assistant',content:'Report findings'}]}),
  ];
  const c=ingestFiles(docs),agents=agentTrajectories(c);
  assert.deepEqual(agents.map(a=>a.name),['Lead Agent','Research Agent']);
  assert.equal(agents[1].records.length,3);
  assert.equal(agents[1].records.at(-1),c.sources.at(-1).records[0]);
  const different=ingestFiles([
    file('trace.json',{agent_id:'a',messages:[{role:'assistant',content:'A'}]}),
    file('trace.jsonl',[{t:'start',agent_id:'b'},{t:'llm',content:'B'}].map(JSON.stringify).join('\n')),
  ]);
  assert.equal(agentTrajectories(different).length,2);
});

test('existing workspaces recover native Agent metadata without rewriting source records', () => {
  const c=ingestFiles([file('private-log-name.jsonl',[{t:'start',agent_id:'worker',role_id:'researcher',task_id:'child'},{t:'llm',content:'Found'}].map(JSON.stringify).join('\n'))]);
  const expected=agentTrajectories(c);
  delete c.sources[0].agent;
  const before=structuredClone(c);
  assert.deepEqual(agentTrajectories(c),expected);assert.deepEqual(c,before);
  const unnamed=ingestFiles([file('secret-file.json',{messages:[{role:'assistant',content:'A'}]}),file('another-private.json',{messages:[{role:'assistant',content:'B'}]})]);
  assert.deepEqual(agentTrajectories(unnamed).map(a=>a.name),['Agent 1','Agent 2']);
});

test('current state.json graph input is not misrepresented as a model request', () => {
  const c=ingestFiles([file('bcg/state.json',{problem_id:'p',acknowledged:[{role:'user',content:'Q'}],snapshot:{nodes:graph.all_nodes,relations:graph.relations}})]);
  assert.equal(c.graphs.length,1);assert.equal(c.contexts.length,0);assert.equal(c.sources[0].kind,'graph_trajectory');assert.equal(c.question,'Q');
});

test('complete graphs and historical snapshots remain separate without changing evidence', () => {
  const c=ingestFiles([
    file('agent/state.json',{snapshot:graph}),
    file('belief_graph.jsonl',[{stream_turn_index:-1,nodes:[],relations:[]},{stream_turn_index:0,...graph}].map(JSON.stringify).join('\n')),
    file('other-agent/state.json',{snapshot:graph}),
  ]);
  c.annotation.events.push(newEvent({sourceRef:c.graphs[2].id,quote:'Saved snapshot citation'}));
  const before=structuredClone(c),views=graphViews(c);
  assert.equal(views.full.length,2);assert.equal(views.turns.length,2);
  assert.equal(views.turns[0].nodes.length,0);
  assert.equal(views.turns[1],c.graphs[2]);
  assert(views.full.every(g=>g.scope==='full'));
  assert(views.turns.every(g=>g.scope==='turns'));
  assert.deepEqual(c,before);
});

test('saved workspaces without scope retain the same graph groups and citation IDs', () => {
  const c=ingestFiles([file('state.json',graph),file('belief_graph.jsonl',JSON.stringify(graph))]);
  const expected=graphViews(c);
  c.graphs.forEach(g=>delete g.scope);
  const before=structuredClone(c);
  const restored=graphViews(restoreWorkspace(workspaceExport([c]))[0]);
  assert.deepEqual(restored.full.map(g=>g.id),expected.full.map(g=>g.id));
  assert.deepEqual(restored.turns.map(g=>g.id),expected.turns.map(g=>g.id));
  assert.deepEqual(c,before);
  assert.equal(graphViews({graphs:[c.graphs[0]]}).turns.length,0);
  assert.equal(graphViews({graphs:[c.graphs[1]]}).full.length,0);
});

test('final graph alone cannot create historical contexts or online usage', () => {
  const c=ingestFiles([file('result.json',{...graph,token_usage:{totals:{total_tokens:999}},mode:'unified'})]);
  assert.equal(c.contexts.length,0);assert.equal(c.mode,'');assert.equal(tokenMetrics(c).graph,null);
});

test('reads actual Chat and Responses system inputs, keeping separate source IDs', () => {
  const logs=[{type:'request',call_id:9,payload:{messages:[{role:'system',content:'S'},{role:'user',content:'U'}]}},{type:'request',call_id:10,payload:{input:[{role:'developer',content:[{type:'input_text',text:'D'}]}]}}];
  const c=ingestFiles([file('model-io.jsonl',logs.map(JSON.stringify).join('\n'))]);
  assert.equal(c.contexts[0].text,'S');assert.equal(c.contexts[1].text,'D');assert.equal(c.contexts[0].kind,'model_request');assert.notEqual(c.contexts[0].id,c.contexts[1].id);
});

test('legacy streaming updates are not counted as extra complete messages', () => {
  const logs=[{type:'message_start',message:{role:'assistant',content:'A'}},{type:'message_update',message:{role:'assistant',content:'AB'}},{type:'message_end',message:{role:'assistant',content:'ABC'}},{type:'turn_end',message:{role:'assistant',content:'ABC'}}];
  const c=ingestFiles([file('trace.jsonl',logs.map(JSON.stringify).join('\n'))]);
  assert.equal(c.sources[0].records.length,1);assert.equal(c.sources[0].records[0].index,3);
});

test('keeps fallback audit without inventing a graph', () => {
  const c=ingestFiles([file('requests.jsonl',JSON.stringify({call:3,fallback:'graph_timeout',selected_node_ids:[]}))]);
  assert.equal(c.contexts[0].fallback,'graph_timeout');assert.equal(c.contexts[0].text,'');assert.equal(c.graphs.length,0);
});

test('prevents accidental cross-task or cross-mode merging', () => {
  assert.throws(()=>ingestFiles([file('1.json',result),file('2.json',{...result,task_id:'43'})]),/多道题/);
  assert.throws(()=>ingestFiles([file('1.json',result),file('2.json',{...result,mode:'default'})]),/不同模式/);
  const c=ingestFiles([file('42.json',result)]);
  assert.throws(()=>ingestFiles([file('43.json',{...result,task_id:'43'})],c),/不一致/);
  assert.throws(()=>ingestFiles([file('default.json',{...result,mode:'default'})],c),/不同模式/);
});

test('attaches task metadata after logs, avoids duplicate identical files, keeps annotations', () => {
  let c=ingestFiles([file('requests.jsonl',JSON.stringify({call:1,system_memory:'G'}))]);c.annotation.reviewer='Reader';
  c=ingestFiles([file('42.json',result)],c);
  assert.equal(c.taskId,'42');assert.equal(c.annotation.reviewer,'Reader');
  const n=c.files.length;c=ingestFiles([file('42.json',result)],c);assert.equal(c.files.length,n);
});

test('native task metrics retain explicit final-flush scope and missing is not zero', () => {
  const c=ingestFiles([file('result.json',{...result,usage:{agent:{total_tokens:100},graph:{total_tokens:20},combined:{total_tokens:120}}})]);
  assert.equal(tokenMetrics(c).combined,120);assert.match(tokenMetrics(c).graphScope,/最终建图/);
  c.metrics.usage.graph.total_tokens=null;assert.equal(tokenMetrics(c).combined,null);
});

test('multi-agent native child transcript does not replace primary benchmark identity', () => {
  const c=ingestFiles([file('result.json',result),file('agent/researcher.json',{task_id:'child-job',role_id:'researcher',messages:[{role:'assistant',content:'Finding'}]})]);
  assert.equal(c.taskId,'42');assert.equal(c.metrics.usage.total,100);
});

test('metadata provenance is stable regardless of graph file order', () => {
  const c=ingestFiles([file('42.json',result),file('graph/result.json',{...graph,mode:'streaming'})]);assert.equal(c.mode,'bcg');
});

test('completion counts selected dimensions only and requires process evidence', () => {
  const c=ingestFiles([file('r.json',result)]);
  assert.deepEqual(completion(c),{completed:0,total:0,percent:0});
  setDimensionSelected(c,'A1',true);assert.deepEqual(completion(c),{completed:0,total:1,percent:0});
  Object.assign(c.annotation.dimensions.A1,{utilization:'部分利用',effect:'正向',evidence:'E01 free text',note:'Used B1'});assert.equal(completion(c).completed,0);
  const e=newEvent({reference:'trace line 3'});c.annotation.events.push(e);
  setDimensionEvent(c,'A1',e.id,true);assert.deepEqual(completion(c),{completed:1,total:1,percent:100});
  setDimensionSelected(c,'A2',false);assert.deepEqual(completion(c),{completed:1,total:1,percent:100});
  Object.assign(c.annotation.dimensions.A3,{opportunity:'有',utilization:'不适用',effect:'不适用',note:'inconsistent'});assert.deepEqual(completion(c),{completed:1,total:2,percent:50});
  setDimensionEvent(c,'A1',e.id,false);assert.equal(completion(c).completed,0);
});

test('dimension selection does not imply utilization and deselection preserves drafts', () => {
  const c=ingestFiles([file('r.json',result)]),e=newEvent({reference:'trace line 3'});
  c.annotation.events.push(e);
  setDimensionSelected(c,'A1',true);
  Object.assign(c.annotation.dimensions.A1,{utilization:'未利用',effect:'负向',note:'Retain this draft'});
  setDimensionEvent(c,'A1',e.id,true);
  const draft=structuredClone(c.annotation.dimensions.A1);
  assert.equal(completion(c).completed,1,'failure to use Graph is a valid evaluation');
  setDimensionSelected(c,'A1',false);
  assert.deepEqual(selectedDimensions(c),[]);
  assert.deepEqual(c.annotation.dimensions.A1,{...draft,opportunity:'无'});
  assert.equal(hasNoApplicableDimensions(c),false,'deselecting is not confirmation that the trajectory had no relevant dimension');
  assert.doesNotMatch(markdownReport(c),/Retain this draft/);
  assert.doesNotMatch(csvSummary([c]),/Retain this draft/);
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored,c);
  setDimensionSelected(restored,'A1',true);
  assert.deepEqual(restored.annotation.dimensions.A1,draft);
  assert.match(markdownReport(restored),/Retain this draft/);
  assert.equal(setDimensionSelected(c,'unknown',true),false);
});

test('no relevant dimensions requires explicit trajectory review confirmation', () => {
  const c=ingestFiles([file('r.json',result)]);
  Object.assign(c.annotation,{reviewer:'R',result:'尚未评判'});
  Object.assign(c.annotation.conclusion,{problem:'No relevant behavior in the available trajectory',evidence:'证据不足'});
  assert(validateReview(c).some(v=>v.includes('选择维度')));
  assert.equal(confirmNoApplicableDimensions(c,true),true);
  assert.equal(hasNoApplicableDimensions(c),true);
  assert.deepEqual(completion(c),{completed:0,total:0,percent:100});
  assert.deepEqual(validateReview(c),[]);
  assert.match(markdownReport(c),/已查看轨迹，未发现相关维度/);
  assert.equal(csvSummary([c]).split('\r\n').length,2,'retain a case-level CSV row');
  setDimensionSelected(c,'A2',true);
  assert.equal(hasNoApplicableDimensions(c),false);
  assert.equal(c.annotation.noApplicableDimensions,false);
  assert.equal(confirmNoApplicableDimensions(c,true),false);
});

test('legacy uncertain dimensions remain visible and need evidence-based confirmation', () => {
  const c=ingestFiles([file('r.json',result)]),e=newEvent({reference:'trace line 3'});
  delete c.annotation.noApplicableDimensions;
  c.annotation.events.push(e);
  Object.assign(c.annotation.dimensions.A1,{opportunity:'无法判断',utilization:'无法判断',effect:'待验证',note:'Earlier judgment'});
  setDimensionEvent(c,'A1',e.id,true);
  const before=structuredClone(c);
  assert.deepEqual(selectedDimensions(c).map(d=>d.id),['A1']);
  assert.equal(completion(c).completed,0);
  assert(validateReview(c).some(v=>v.includes('A1 根据关键过程确认影响可能')));
  assert.deepEqual(c,before);
  setDimensionSelected(c,'A1',true);
  assert.equal(completion(c).completed,1);
  assert.equal(c.annotation.dimensions.A1.note,'Earlier judgment');
  assert.deepEqual(c.annotation.dimensions.A1.eventIds,[e.id]);
  const oldNone=ingestFiles([file('r.json',result)]);
  delete oldNone.annotation.noApplicableDimensions;
  for(const r of Object.values(oldNone.annotation.dimensions))r.opportunity='无';
  assert.equal(hasNoApplicableDimensions(oldNone),true);
});

test('new selections clear legacy N/A ratings and invalid selection state is rejected', () => {
  const c=ingestFiles([file('r.json',result)]);
  Object.assign(c.annotation.dimensions.A1,{opportunity:'无',utilization:'不适用',effect:'不适用'});
  setDimensionSelected(c,'A1',true);
  assert.equal(c.annotation.dimensions.A1.utilization,'');
  assert.equal(c.annotation.dimensions.A1.effect,'');
  for(const value of ['yes',true]){
    const bad=structuredClone(c);bad.annotation.noApplicableDimensions=value;
    assert.throws(()=>restoreWorkspace(workspaceExport([bad])),/维度选择状态无效/);
  }
});

test('review cannot complete without evidence for positive/negative attribution', () => {
  const c=ingestFiles([file('r.json',result)]);const a=c.annotation;
  assert(validateReview(c).includes('标注人'));
  Object.assign(a,{reviewer:'R',result:'正确 / 达标'});
  assert.equal(a.criteria,'');
  assert.equal(a.settings,'');
  confirmNoApplicableDimensions(c,true);
  Object.assign(a.conclusion,{benefit:'No opportunity',evidence:'证据不足'});
  assert.deepEqual(validateReview(c),[]);
  setDimensionSelected(c,'A1',true);
  Object.assign(a.dimensions.A1,{utilization:'充分利用',effect:'正向',note:'Used B1'});
  assert(validateReview(c).some(x=>x.includes('关键过程')));
  const complete=newEvent({reference:'trace line 3',behavior:'Targeted search',impact:'Found answer',strength:'明确可见'});
  const incomplete=newEvent({reference:'trace line 4'});
  a.events.push(complete,incomplete);
  assert(validateReview(c).includes('A1 关联的关键过程'),'an unrelated process cannot satisfy a dimension');
  setDimensionEvent(c,'A1',incomplete.id,true);
  assert(validateReview(c).some(x=>x.includes('A1')&&x.includes('实际行为')));
  setDimensionEvent(c,'A1',complete.id,true);
  assert.deepEqual(validateReview(c),[]);
});

test('conclusion recap uses selected annotations and current links without inferring conclusions', () => {
  const c=ingestFiles([file('r.json',result)]),a=c.annotation;
  const first=newEvent({label:'First',fault:'Agent 使用'}),second=newEvent({label:'Second'});
  a.events.push(first,second);
  setDimensionSelected(c,'A2',true);
  Object.assign(a.dimensions.A2,{utilization:'未利用',effect:'负向',note:'Evidence from the trajectory'});
  setDimensionEvent(c,'A2',first.id,true);
  Object.assign(a.dimensions.A4,{opportunity:'无',note:'Inactive draft'});
  setDimensionEvent(c,'A4',second.id,true);
  const before=structuredClone(c);
  const observations=conclusionObservations(c);
  assert.deepEqual(observations.map(d=>d.id),['A2']);
  assert.equal(observations[0].effect,'负向');
  assert.equal(observations[0].note,'Evidence from the trajectory');
  assert.deepEqual(observations[0].events,[first]);
  assert.deepEqual(c,before,'recap never fills or changes the conclusion');
  removeEvent(c,first.id);
  assert.deepEqual(conclusionObservations(c)[0].events,[]);
  setDimensionSelected(c,'A2',false);
  assert.deepEqual(conclusionObservations(c),[]);
});

test('improvement stages and verification preserve existing text and round-trip in exports', () => {
  const c=ingestFiles([file('r.json',result)]),conclusion=c.annotation.conclusion;
  Object.assign(conclusion,{benefit:'Benefit',problem:'Problem',improvement:'Existing proposal',comparison:'Existing comparison',verification:'Check repeated queries and answer quality'});
  setConclusionFocus(c,'Agent 使用',true);
  setConclusionFocus(c,'建图',true);
  setConclusionFocus(c,'建图',true);
  assert.deepEqual(conclusion.focus,['建图','Agent 使用']);
  setConclusionFocus(c,'建图',false);
  assert.deepEqual(conclusion.focus,['Agent 使用']);
  assert.equal(setConclusionFocus(c,'无明显失效',true),false);
  assert.equal(setConclusionFocus(c,'made-up',true),false);
  assert.equal(setConclusionFocus(c,IMPROVEMENT_STAGES[0],'yes'),false);
  assert.equal(conclusion.improvement,'Existing proposal');
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored,c);
  const md=markdownReport(c);
  assert.match(md,/优先改进环节：Agent 使用/);
  assert.match(md,/具体改进方案：Existing proposal/);
  assert.match(md,/验证方式与预期变化：Check repeated queries and answer quality/);
  assert.match(md,/对照条件与其他影响因素：Existing comparison/);
});

test('old conclusion fields remain unchanged and malformed new fields are rejected', () => {
  const c=ingestFiles([file('r.json',result)]);
  delete c.annotation.conclusion.focus;
  delete c.annotation.conclusion.verification;
  c.annotation.conclusion.improvement='Legacy proposal and verification';
  assert.deepEqual(restoreWorkspace(workspaceExport([c]))[0],c);
  assert.match(markdownReport(c),/Legacy proposal and verification/);
  setConclusionFocus(c,'Graph 选择',true);
  assert.deepEqual(c.annotation.conclusion.focus,['Graph 选择']);
  for(const focus of ['建图',['bad'],['建图','建图'],[null]]){
    const bad=structuredClone(c);bad.annotation.conclusion.focus=focus;
    assert.throws(()=>restoreWorkspace(workspaceExport([bad])),/改进环节无效/);
  }
  const bad=structuredClone(c);bad.annotation.conclusion.verification={text:'bad'};
  assert.throws(()=>restoreWorkspace(workspaceExport([bad])),/验证方式无效/);
});

test('process links are multi-select, case-local and stable across renaming and reordering', () => {
  const c=ingestFiles([file('r.json',result)]),a=c.annotation;
  const first=newEvent({label:'First'}),second=newEvent({label:'Second'});
  a.events.push(first,second);
  assert.equal(setDimensionEvent(c,'A1',first.id,true),true);
  setDimensionEvent(c,'A1',first.id,true);
  setDimensionEvent(c,'A1',second.id,true);
  assert.deepEqual(a.dimensions.A1.eventIds,[first.id,second.id]);
  assert.equal(setDimensionEvent(c,'A1','another-case-id',true),false);
  assert.equal(setDimensionEvent(c,'not-a-dimension',first.id,true),false);
  first.label='Renamed';
  a.events.reverse();
  assert.deepEqual(linkedEvents(c,'A1').map(e=>e.id),[second.id,first.id]);
  assert.equal(eventLabel(c,first),'E02 · Renamed');
  setDimensionEvent(c,'A1',second.id,false);
  assert.deepEqual(linkedEvents(c,'A1').map(e=>e.id),[first.id]);
});

test('deleting a process removes its links without retargeting or erasing other annotations', () => {
  const c=ingestFiles([file('r.json',result)]),a=c.annotation;
  const first=newEvent(),second=newEvent();a.events.push(first,second);
  setDimensionEvent(c,'A1',first.id,true);setDimensionEvent(c,'A1',second.id,true);
  setDimensionEvent(c,'A2',first.id,true);
  a.dimensions.A2.note='Keep this judgment';
  a.dimensions.A2.evidence='Keep this old evidence text';
  assert.equal(removeEvent(c,first.id),true);
  assert.deepEqual(a.events,[second]);
  assert.deepEqual(a.dimensions.A1.eventIds,[second.id]);
  assert.deepEqual(a.dimensions.A2.eventIds,[]);
  assert.equal(a.dimensions.A2.note,'Keep this judgment');
  assert.equal(a.dimensions.A2.evidence,'Keep this old evidence text');
  assert.equal(removeEvent(c,first.id),false);
});

test('legacy evidence stays intact and never silently becomes a process link', () => {
  const c=ingestFiles([file('r.json',result)]);
  for(const row of Object.values(c.annotation.dimensions))delete row.eventIds;
  c.annotation.events.push(newEvent({label:'Existing process'}));
  c.annotation.dimensions.A1.evidence='E01 / call 4 / B10';
  setDimensionSelected(c,'A1',true);
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored,c);
  assert.deepEqual(linkedEvents(restored,'A1'),[]);
  assert.match(markdownReport(restored),/原有证据备注：E01 \/ call 4 \/ B10/);
  assert.match(csvSummary([restored]),/E01 \/ call 4 \/ B10/);
  setDimensionEvent(restored,'A1',restored.annotation.events[0].id,true);
  assert.equal(linkedEvents(restored,'A1').length,1);
  assert.equal(restored.annotation.dimensions.A1.evidence,'E01 / call 4 / B10');
});

test('process links survive JSON restore and appear in Markdown and CSV exports', () => {
  const c=ingestFiles([file('r.json',result)]);
  const e=newEvent({label:'Inspect source',reference:'trace line 12'});c.annotation.events.push(e);
  setDimensionSelected(c,'A3',true);
  setDimensionEvent(c,'A3',e.id,true);
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored,c);
  assert.match(markdownReport(c),/\[E01 · Inspect source\]\(#process-1\)/);
  assert.match(markdownReport(c),/<a id="process-1"><\/a>/);
  assert.match(csvSummary([c]),/E01 · Inspect source · trace line 12/);
  assert(csvSummary([c]).includes(e.id));
});

test('workspace restore rejects malformed, duplicate and foreign process links', () => {
  const c=ingestFiles([file('r.json',result)]),e=newEvent();c.annotation.events.push(e);
  for(const value of ['E01',null,[42],['missing'],[e.id,e.id]]){
    const altered=structuredClone(c);altered.annotation.dimensions.A1.eventIds=value;
    assert.throws(()=>restoreWorkspace(workspaceExport([altered])),/关联无效/);
  }
  const altered=structuredClone(c);altered.annotation.events.push({...altered.annotation.events[0]});
  assert.throws(()=>restoreWorkspace(workspaceExport([altered])),/ID 重复/);
});

test('exports and restores without losing evidence or annotation, rejects bad versions', () => {
  const c=ingestFiles([file('r.json',result)]);c.annotation.events.push(newEvent({reference:'trace',quote:'Q',behavior:'Used source'}));
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))));assert.deepEqual(restored,[c]);
  assert.throws(()=>restoreWorkspace({format:'bcg-case-study-workspace',version:2,cases:[]}),/版本/);
  const bad=workspaceExport([structuredClone(c)]);delete bad.cases[0].annotation.dimensions.A1;assert.throws(()=>restoreWorkspace(bad),/不完整/);
});

test('removed basic fields are omitted when empty and preserved in legacy exports', () => {
  const c=ingestFiles([file('r.json',result)]);
  assert.doesNotMatch(markdownReport(c),/成功标准|## 运行配置/);
  c.annotation.criteria='Previously recorded criteria';
  c.annotation.settings='Previously recorded configuration';
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))));
  assert.deepEqual(restored,[c]);
  assert.match(markdownReport(restored[0]),/Previously recorded criteria/);
  assert.match(markdownReport(restored[0]),/Previously recorded configuration/);
});

test('AI task assessment survives export with rationale and separate task/report evidence',()=>{
  const c=ingestFiles([file('r.json',result)]);
  c.annotation.result='正确 / 达标';
  c.annotation.resultAssessment={source:'ai',result:c.annotation.result,rationale:'The report answers the requested year.',
    task_quotes:['Find the year.'],report_quotes:['2024'],limitations:['The original source was not independently verified.']};
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored.annotation.resultAssessment,c.annotation.resultAssessment);
  const md=markdownReport(restored);
  for(const value of ['## 任务结果 · AI 评估','非 Benchmark 官方评分','The report answers the requested year.','> Find the year.','> 2024','not independently verified'])assert(md.includes(value));
  delete restored.annotation.resultAssessment;
  assert.doesNotMatch(markdownReport(restored),/## 任务结果 · AI 评估/);
});

test('CSV protects formula injection and Markdown includes original source hashes', () => {
  const c=ingestFiles([file('r.json',{...result,task_id:'=HYPERLINK("x")'})]);c.annotation.reviewer='@danger';
  setDimensionSelected(c,'A8',true);
  assert.match(csvSummary([c]),/"'=HYPERLINK/);assert.match(csvSummary([c]),/"'@danger"/);assert.equal(csvSummary([c]).split('\r\n').length,2);
  assert.match(markdownReport(c),/SHA-256/);assert.match(markdownReport(c),/A8/);
  assert.match(markdownReport(c),/\| 维度 \| 是否利用 \| 是否产生效果 \|/);
  assert.match(markdownReport(c),/## 关键过程/);
  assert.doesNotMatch(markdownReport(c),/## 关键事件/);
});

test('AI search rubric and grouped original turns survive workspace and Markdown export',()=>{
  const c=ingestFiles([file('r.json',result)]);
  const e=newEvent({label:'Targeted verification',behavior:'Fetch the original source.',impact:'Scope was preserved.',strength:'有较强迹象'});
  e.searchAssessment={verdict:'合理核验',information_need:'Original study population',graph_sufficiency:'部分',verification_reason:'原文细节',expected_gain:'Study scope',basis:'Only a summary was injected.',observed_gain:'有新增证据',result_quote:'The study included 200 participants.',unavailable_quote:''};
  e.groupReason='同一信息缺口的核验过程。';
  e.relatedOccurrences=[{sourceLocator:{path:'agent/log.jsonl',digest:'abc',index:4,call:2},agent:'agent',call:2,quote:'Need the source scope.',graphEvidence:{text:'B2: Only the abstract is available.',call:2,status:'matched'}}];
  c.annotation.events.push(e);
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored.annotation.events[0].searchAssessment,e.searchAssessment);
  assert.deepEqual(restored.annotation.events[0].relatedOccurrences,e.relatedOccurrences);
  const md=markdownReport(restored);
  for(const expected of ['搜索判断 · 合理核验','同类过程与关联轮次','agent · call 2','B2: Only the abstract is available.','Need the source scope.'])assert(md.includes(expected));
});

test('malicious HTML remains text and primitive array entries do not crash import', () => {
  const c=ingestFiles([file('r.json',[{role:'assistant',content:'<img src=x onerror=alert(1)>'},null,4])]);
  assert.equal(c.sources[0].records[0].text,'<img src=x onerror=alert(1)>');assert.equal(c.sources[0].records.length,3);
});

test('native tool blocks and Responses reasoning summaries retain readable content', () => {
  assert.match(textOf({type:'toolCall',name:'web_search',arguments:{query:'X'}}),/web_search/);
  assert.equal(textOf({type:'reasoning',summary:[{type:'summary_text',text:'Need evidence'}]}),'Need evidence');
});

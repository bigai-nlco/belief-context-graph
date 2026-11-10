import assert from 'node:assert/strict';
import {test} from 'node:test';
import {benchmarkLabel,contextModeLabel,remoteLocation,caseLink,defaultRunTitle,bindAIAnnotation,saveSharedCase,taskAnnotationStatus,canAnnotateTask,annotationProgress,reportScoreLabel,runDisplayStatus,runScoreSummary} from '../shared.mjs';

test('full-graph ablation is displayed separately from BCG and Default',()=>{
  assert.equal(contextModeLabel('bcg'),'BCG');
  assert.equal(contextModeLabel('bcg_no_retrieval'),'BCG w/o Retrieval');
  assert.equal(contextModeLabel('default'),'Default');
  assert.equal(contextModeLabel(undefined),'BCG');
});

test('run mode labels consistently include enabled task state',()=>{
  assert.equal(contextModeLabel('bcg',true),'BCG + 任务状态');
  assert.equal(contextModeLabel('bcg_no_retrieval',true),'BCG w/o Retrieval + 任务状态');
  assert.equal(contextModeLabel('bcg',false),'BCG');
  assert.equal(contextModeLabel('bcg_no_retrieval',false),'BCG w/o Retrieval');
  assert.equal(contextModeLabel('default',false),'Default');
  assert.equal(contextModeLabel('default',true),'Default');
  assert.equal(contextModeLabel('bcg',undefined),'BCG');
});

test('run total sums only recorded Judge scores, with zero distinct from unscored',()=>{
  const scored=official_score=>({scoring_status:'official_protocol_scored',official_score});
  assert.deepEqual(runScoreSummary([]),{scored:0,total:0,points:null,average:null});
  assert.deepEqual(runScoreSummary([scored(0)]),{scored:1,total:1,points:0,average:0});
  const pending=[{status:'running'}, {scoring_status:'judge_error',official_score:.5},
    {scoring_status:'not_judged',official_score:1}, {scoring_status:'official_pending',official_score:1},
    ...[null,undefined,NaN,Infinity,'0.5',-.2,1.2].map(scored)];
  assert.deepEqual(runScoreSummary(pending),{scored:0,total:pending.length,points:null,average:null});
  assert.deepEqual(runScoreSummary([scored(.25),scored(.5),scored(0),...pending]),
    {scored:3,total:pending.length+3,points:75,average:25});
  // Aggregate original scores, not rounded per-task labels.
  const summary=runScoreSummary([scored(1/3),scored(1/3),scored(1/3)]);
  assert.equal(summary.points.toFixed(1),'100.0');
  assert.equal(summary.average.toFixed(1),'33.3');
});

test('report judging is separate from report generation and annotation',()=>{
  assert.equal(runDisplayStatus({status:'running',phase:'judging'}),'judging');
  assert.equal(runDisplayStatus({status:'cancelled',phase:'judging'}),'cancelled');
  assert.equal(runDisplayStatus({status:'finished_with_judge_errors',phase:'judging'}),'finished_with_judge_errors');
  assert.equal(reportScoreLabel({status:'completed'}),'待评分');
  assert.equal(reportScoreLabel({status:'running'}),'—');
  assert.equal(reportScoreLabel({scoring_status:'official_protocol_scored',official_score:0}),'0.0 / 100');
  assert.equal(reportScoreLabel({scoring_status:'official_protocol_scored',official_score:0.735}),'73.5 / 100');
  assert.equal(reportScoreLabel({scoring_status:'official_protocol_scored',official_score:1}),'100.0 / 100');
  assert.equal(reportScoreLabel({scoring_status:'official_protocol_scored',official_score:null,status:'completed'}),'待评分');
  assert.equal(reportScoreLabel({scoring_status:'official_protocol_scored',official_score:NaN,status:'completed'}),'待评分');
  assert.equal(reportScoreLabel({scoring_status:'official_pending',official_score:1,status:'completed'}),'待评分');
  assert.equal(reportScoreLabel({scoring_status:'judge_error'}),'评分失败');
  assert.equal(reportScoreLabel({scoring_status:'not_judged'}),'未评分');
  assert.equal(taskAnnotationStatus({scoring_status:'official_protocol_scored',official_score:1}).completed,false);
});

test('annotation progress counts eligible tasks once and excludes failures and drafts',()=>{
  const task=(reviews=[],ai='not_started',changes={})=>({status:'completed',annotatable:true,reviews:reviews.map(status=>({status})),ai_review:{status:ai},...changes});
  assert.deepEqual(annotationProgress([]),{completed:0,total:0});
  assert.deepEqual(annotationProgress([task(['reviewed']),task([],'completed'),task(['reviewed'],'completed'),task(['draft'],'running')]),{completed:3,total:4});
  assert.deepEqual(annotationProgress([task(['reviewed'],'completed',{status:'timeout'}),task([],'completed',{status:'agent_error'}),task([],'not_started',{status:'running'}),task([],undefined,{annotatable:false}),task(['draft'])]),{completed:0,total:1});
  assert.deepEqual(annotationProgress([task([],undefined,{status:'not_run'})]),{completed:0,total:0});
});

test('annotation is only available for successfully completed tasks',()=>{
  assert.equal(canAnnotateTask({status:'completed',annotatable:true}),true);
  assert.equal(canAnnotateTask({status:'completed',annotatable:false}),false);
  assert.equal(canAnnotateTask(undefined),false);
  for(const status of ['failed','timeout','agent_error','worker_error','empty_report','score_error','graph_fallback','interrupted','cancelled','queued','running','not_run','unknown']){
    assert.equal(canAnnotateTask({status,annotatable:true,report:true,ai_review:{status:'completed'}}),false,status);
  }
});

test('either completed human or AI annotation marks the task annotated, never drafts or running jobs',()=>{
  const status=(human,ai)=>taskAnnotationStatus({reviews:human.map(s=>({status:s})),ai_review:{status:ai}});
  assert.deepEqual(status([],'not_started'),{completed:false,inProgress:false,label:'未标注',sources:[]});
  for(const state of ['queued','running'])assert.equal(status([],state).label,'标注中');
  assert.equal(status(['draft'],'failed').completed,false);
  assert.equal(status(['draft'],'not_started').label,'标注中');
  assert.deepEqual(status([],'completed'),{completed:true,inProgress:false,label:'AI 已标注',sources:['AI']});
  assert.deepEqual(status(['reviewed'],'not_started').sources,['人工']);
  assert.equal(status(['reviewed'],'running').label,'人工已标注');
  assert.deepEqual(status(['draft'],'completed').sources,['AI']);
  assert.deepEqual(status(['reviewed','draft'],'completed').sources,['人工','AI']);
  assert.equal(status(['reviewed','draft'],'completed').label,'AI 和人工已标注');
  assert.equal(status(['draft'],'completed').label,'AI 已标注');
});

test('shared routes and labels retain only DeepResearch Bench II',()=>{
  assert.equal(benchmarkLabel('drb2'),'DeepResearch Bench II');
  assert.equal(remoteLocation(''),null);
  assert.deepEqual(remoteLocation('?run=5&benchmark=drb2&task=30'),{run:5,benchmark:'drb2',task:'30'});
  assert.equal(caseLink(5,'drb2','30'),'./index.html?run=5&benchmark=drb2&task=30');
  assert.throws(()=>remoteLocation('?run=5&benchmark=drbench&task=DR0001'),/DeepResearch Bench II/);
  assert.throws(()=>remoteLocation('?run=5&benchmark=drb2&task=..%2Fsecret'),/无效/);
});

test('AI route is separate from personal annotation and read-only',async()=>{
  assert.deepEqual(remoteLocation('?run=1&benchmark=drb2&task=2&review=ai'),{run:1,benchmark:'drb2',task:'2',review:'ai'});
  assert.equal(caseLink(1,'drb2','2','ai'),'./index.html?run=1&benchmark=drb2&task=2&review=ai');
  assert.throws(()=>remoteLocation('?run=1&benchmark=drb2&task=2&review=other'),/类型/);
  await assert.rejects(saveSharedCase({remote:{readOnly:true}}),/只读/);
});

test('AI source locators bind to fresh browser record IDs using path, digest, line and call',()=>{
  const c={files:[{id:'fresh-file',path:'agent/trajectories/react_agent.jsonl',digest:'abc'}],
    sources:[{fileId:'fresh-file',records:[{id:'fresh-record',index:2,call:1,role:'assistant'}]}]};
  const annotation={events:[{sourceRef:'old-random-id',sourceLocator:{path:c.files[0].path,digest:'abc',index:2,call:1}}]};
  assert.equal(bindAIAnnotation(c,annotation).events[0].sourceRef,'record:fresh-record');
  assert.equal(annotation.events[0].sourceRef,'old-random-id','saved annotation is not mutated');
  for(const field of ['path','digest','index','call']){
    const changed=structuredClone(annotation);changed.events[0].sourceLocator[field]='incorrect';
    assert.throws(()=>bindAIAnnotation(c,changed),/不匹配/);
  }
  c.sources[0].records.push({...c.sources[0].records[0],id:'ambiguous-record'});
  assert.throws(()=>bindAIAnnotation(c,annotation),/不匹配/);
});

test('AI reviewer uses only the configured model name, including legacy annotations',()=>{
  const annotation={reviewer:'AI · GPT-6.1-Sol',events:[]};
  for(const model of ['gpt-6.1-sol','another-review-model']){
    assert.equal(bindAIAnnotation({},annotation,model).reviewer,model);
  }
  assert.equal(annotation.reviewer,'AI · GPT-6.1-Sol','stored annotation is not mutated');
});

test('grouped AI occurrences also require exact source binding',()=>{
  const source={path:'agent/trajectories/react_agent.jsonl',digest:'abc',index:2,call:1};
  const other={...source,index:4,call:2};
  const c={files:[{id:'file',path:source.path,digest:'abc'}],sources:[{fileId:'file',records:[
    {id:'one',role:'assistant',index:2,call:1},{id:'two',role:'assistant',index:4,call:2}]}]};
  const annotation={events:[{sourceLocator:source,relatedOccurrences:[{sourceLocator:other}]}]};
  assert.equal(bindAIAnnotation(c,annotation).events[0].relatedOccurrences[0].sourceRef,'record:two');
  annotation.events[0].relatedOccurrences[0].sourceLocator.digest='wrong';
  assert.throws(()=>bindAIAnnotation(c,annotation),/不匹配/);
});

test('new run titles include local date and minute, without seconds',()=>{
  assert.equal(defaultRunTitle(new Date(2026,9,26,15,8,47)), 'DeepResearch Bench II 测试 · 2026-10-26 15:08');
  assert.equal(defaultRunTitle(new Date(2026,0,2,0,3,59)), 'DeepResearch Bench II 测试 · 2026-01-02 00:03');
  assert.equal(defaultRunTitle(new Date(2027,0,1,0,0,0)), 'DeepResearch Bench II 测试 · 2027-01-01 00:00');
});

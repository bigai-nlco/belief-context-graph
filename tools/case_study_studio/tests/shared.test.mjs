import assert from 'node:assert/strict';
import {test} from 'node:test';
import {benchmarkLabel,remoteLocation,caseLink,defaultRunTitle,bindAIAnnotation,saveSharedCase,taskAnnotationStatus,canAnnotateTask,annotationProgress} from '../shared.mjs';

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

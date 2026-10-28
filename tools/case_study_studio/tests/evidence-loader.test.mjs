import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {sha256Hex} from '../browser_crypto.mjs';
import {BUNDLE_FORMAT} from '../bundles.mjs';
import {newAnnotation} from '../model.mjs';
import {loadSharedCase,loadFullSharedEvidence,bindAIAnnotation} from '../shared.mjs';
import {parseEvidence} from '../evidence-loader.mjs';

const remote={run:1,benchmark:'drb2',task:'2',review:'ai'},sha='a'.repeat(64);
async function fixture(benchmark='deepresearch_bench_ii',task='2'){
  async function file(path,value){
    const text=typeof value==='string'?value:JSON.stringify(value),bytes=new TextEncoder().encode(text);
    return {path,text,size:bytes.length,digest:await sha256Hex(bytes)};
  }
  const files=[
    await file('case.json',{task_id:task,benchmark,mode:'bcg',question:'Question',final_response:'Answer'}),
    await file('agent/trajectories/react_agent.jsonl','{"t":"start","user_message":"Question"}\n{"t":"llm","turn":1,"content":"Evidence used"}\n'),
    await file('agent/bcg/requests.jsonl','{"call":1,"system_memory":"<belief_graph>B1: known fact</belief_graph>"}\n'),
    await file('agent/bcg/state.json',{snapshot:{nodes:[{id:1,belief:'known fact'}],relations:[]}}),
  ];
  const full={format:BUNDLE_FORMAT,version:1,cases:[{id:'b'.repeat(64),task_id:task,benchmark,mode:'bcg',warnings:[],files}]};
  const preview={...full,cases:[{...full.cases[0],files:files.slice(0,3)}],projection:{
    kind:'review-preview',version:1,source_sha256:sha,omitted_files:[{path:files[3].path,size:files[3].size}]}};
  const annotation=newAnnotation();
  annotation.events=[{id:'e1',sourceLocator:{path:files[1].path,digest:files[1].digest,index:2,call:1},label:'Observed use'}];
  return {full,preview,review:{status:'completed',evidence_sha:sha,annotation,model:'test-model'}};
}
function mockAPI(t,fixture,changes={}){
  const fetch=globalThis.fetch,storage=globalThis.localStorage,requests=[];
  globalThis.localStorage={getItem:()=>''};
  globalThis.fetch=async url=>{
    requests.push(url);
    if(url.endsWith('/session'))return Response.json({user:{id:9,name:'Reviewer'}});
    if(url==='/api/runs/1')return Response.json({id:1,tasks:[]});
    if(url.endsWith('/ai-review'))return Response.json(fixture.review);
    if(url.endsWith('/annotation'))return Response.json({revision:1,annotation:null});
    const preview=url.endsWith('/evidence-preview');
    const body=preview?fixture.preview:fixture.full;
    return new Response(gzipSync(JSON.stringify(body)),{headers:{'X-Evidence-SHA256':changes.sha||sha}});
  };
  t.after(()=>{globalThis.fetch=fetch;globalThis.localStorage=storage;});
  return requests;
}

test('AI initial load uses preview, binds exact sources and hydrates full evidence only on demand',async t=>{
  const data=await fixture(),requests=mockAPI(t,data);
  const {c}=await loadSharedCase(remote,()=>{throw new Error('AI cannot read personal drafts');});
  assert.equal(c.benchmark,'deepresearch_bench_ii','canonical result metadata is not rewritten to the route name');
  assert.equal(c.graphs.length,0);assert.equal(c.contexts.length,1);assert.equal(c.remote.deferredEvidence,true);
  assert.equal(c.annotation.reviewer,'test-model');
  assert.match(c.annotation.events[0].sourceRef,/^record:/);
  assert.equal(requests.filter(p=>p.endsWith('/evidence')).length,0);
  assert.equal(requests.filter(p=>p.endsWith('/evidence-preview')).length,1);
  const before=structuredClone(c),full=await loadFullSharedEvidence(c);
  assert.deepEqual(c,before,'original view remains usable until all verification succeeds');
  assert.equal(full.graphs.length,1);assert.equal(full.remote.deferredEvidence,false);assert.equal(full.id,c.id);
  assert.equal(full.annotation.events[0].label,'Observed use');
  assert.notEqual(full.annotation.events[0].sourceRef,c.annotation.events[0].sourceRef);
  assert.deepEqual(bindAIAnnotation(full,c.annotation,'test-model'),full.annotation);
  assert.equal(requests.filter(p=>p.endsWith('/evidence')).length,1);
  await loadFullSharedEvidence(full);
  assert.equal(requests.filter(p=>p.endsWith('/evidence')).length,1,'no duplicate full downloads');
});

test('legacy short benchmark names still load through the same shared route',async t=>{
  const data=await fixture('drb2');mockAPI(t,data);
  const {c}=await loadSharedCase(remote);
  assert.equal(c.benchmark,'drb2');
  assert.equal((await loadFullSharedEvidence(c)).graphs.length,1);
});

for(const [benchmark,task] of [['deepresearch_bench_ii','5'],['deepresearch_bench','2'],['mmlu','2'],['','2']]){
  test(`both readers reject genuinely different task identity: ${benchmark}/${task}`,async t=>{
    const data=await fixture(benchmark,task);mockAPI(t,data);
    for(const review of ['ai',undefined]){
      await assert.rejects(loadSharedCase({...remote,review}),/证据与当前任务不匹配/);
    }
  });
}

test('preview and full hydration reject mixed evidence revisions',async t=>{
  const data=await fixture();mockAPI(t,data);
  const {c}=await loadSharedCase(remote);
  data.preview.projection.source_sha256='f'.repeat(64);
  await assert.rejects(loadSharedCase(remote),/版本/);
  data.full.cases[0].files[1]={...data.full.cases[0].files[1],text:'changed'};
  await assert.rejects(loadFullSharedEvidence(c));
  assert.equal(c.remote.deferredEvidence,true);
  assert.equal(c.annotation.events[0].label,'Observed use');
});

test('wrong source binding is rejected before publishing the preview',async t=>{
  const data=await fixture();mockAPI(t,data);
  data.review.annotation.events[0].sourceLocator.call=2;
  await assert.rejects(loadSharedCase(remote),/不匹配/);
});

test('human pages load only the current user draft and preserve conflict detection',async t=>{
  const data=await fixture(),requests=mockAPI(t,data),human={...remote};delete human.review;
  const {c}=await loadSharedCase(human,id=>{assert.equal(id,'shared-1-drb2-2-9');return null;});
  assert.equal(c.graphs.length,1);assert.equal(c.annotation.reviewer,'Reviewer');
  assert.equal(requests.some(p=>p.endsWith('/evidence-preview')),false);
  c.remote.dirty=true;c.remote.revision=0;c.annotation.goal='unsaved personal note';
  const restored=await loadSharedCase(human,id=>{assert.equal(id,c.id);return c;});
  assert.equal(restored.c.annotation.goal,'unsaved personal note');assert.equal(restored.c.remote.conflict,true);
});

test('shared parser still verifies retained files when skipping the wrapper hash',async()=>{
  const data=await fixture();data.preview.cases[0].files[1].digest='0'.repeat(64);
  await assert.rejects(parseEvidence(new Blob([gzipSync(JSON.stringify(data.preview))])));
});

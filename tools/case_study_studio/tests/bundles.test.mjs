import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {sha256Hex} from '../browser_crypto.mjs';
import {importBundles, readImportFile, BUNDLE_FORMAT} from '../bundles.mjs';
import {agentTrajectories, graphViews, workspaceExport, restoreWorkspace} from '../model.mjs';

const encode = text => new TextEncoder().encode(text);
async function file(path, data) {
  const text=typeof data==='string' ? data : JSON.stringify(data);
  return {path,text,size:encode(text).length,digest:await sha256Hex(encode(text))};
}
async function entry(task='42', mode='bcg', attempt='first') {
  return {
    id:await sha256Hex(encode(task+mode+attempt)), task_id:task,benchmark:'offline',mode,warnings:[],
    files:[
      await file('case.json',{task_id:task,benchmark:'offline',mode,question:`Question ${task}`,final_response:`Answer ${task}`}),
      await file('agent/trajectories/react_agent.jsonl',
        `{"t":"start","user_message":"Question ${task}"}\n{"t":"llm","content":"Answer ${task}"}`),
      await file('agent/bcg/state.json',{problem_id:task,snapshot:{nodes:[{id:1,belief:`Belief ${task}`}],relations:[]}}),
      await file('agent/bcg/requests.jsonl',`{"call":1,"system_memory":"B1 of ${task}","selected_node_ids":[1]}\n`),
      await file('agent/bcg/graph_snapshots.jsonl',`{"call":1,"snapshot":{"nodes":[{"id":1,"belief":"Old ${task}"}],"relations":[]}}\n`),
    ],
  };
}
const bundle = cases => ({format:BUNDLE_FORMAT,version:1,cases});

test('imports multiple tasks/modes/attempts into separate unannotated cases',async()=>{
  const a=await entry(), b=await entry('43'), otherMode=await entry('42','default'), retry=await entry('42','bcg','second');
  const {cases}=await importBundles([bundle([a,b]),bundle([otherMode,retry])]);
  assert.equal(cases.length,4);assert.equal(new Set(cases.map(c=>c.id)).size,4);
  assert.deepEqual(cases.map(c=>c.taskId),['42','43','42','42']);
  for(const c of cases){
    assert.equal(c.report,`Answer ${c.taskId}`);
    assert.equal(agentTrajectories(c).length,1);
    assert.equal(c.contexts.length,1);
    assert.equal(graphViews(c).full.length,1);assert.equal(graphViews(c).turns.length,1);
    assert.equal(c.annotation.events.length,0);assert.equal(c.annotation.result,'');
    assert.equal(c.annotation.reviewer,'');
    assert.equal(graphViews(c).turns[0].nodes[0].belief,`Old ${c.taskId}`);
  }
});

test('repeat imports preserve existing annotations and never duplicate source evidence',async()=>{
  const source=bundle([await entry()]);
  const first=await importBundles([source]);
  first.cases[0].annotation.reviewer='Reviewer';first.cases[0].annotation.status='reviewed';
  const before=structuredClone(first.cases);
  assert.deepEqual(await importBundles([source,source],first.cases),{cases:[],skipped:2,revisions:0});
  assert.deepEqual(first.cases,before);
  assert.deepEqual(restoreWorkspace(workspaceExport(first.cases)),first.cases);
});

test('updated evidence becomes a separate revision without breaking old citations',async()=>{
  const original=await entry();
  const first=await importBundles([bundle([original])]);
  const updated=structuredClone(original);
  updated.files.push(await file('extra.md','New observation'));
  const next=await importBundles([bundle([updated])],first.cases);
  assert.equal(next.revisions,1);assert.equal(next.cases.length,1);
  assert.notEqual(next.cases[0].id,first.cases[0].id);
  assert.equal(first.cases[0].files.length,5);
});

test('rejects corruption, unsafe paths, duplicate paths and mixed task metadata before import',async()=>{
  const source=await entry();
  for(const mutate of [
    e=>{e.files[0].text+=' ';},
    e=>{e.files[0].size++;},
    e=>{e.files[0].path='../case.json';},
    e=>{e.files[0].path='/case.json';},
    e=>{e.files.push(e.files[0]);},
    e=>{e.mode='default';},
    e=>{e.files=e.files.filter(f=>f.path!=='case.json');},
    e=>{e.id='__proto__';},
  ]){
    const bad=structuredClone(source);mutate(bad);
    await assert.rejects(importBundles([bundle([source,bad])]));
  }
  await assert.rejects(importBundles([{...bundle([source]),version:99}]));
});

test('uncompressed and gzip imports share hashes and have bounded decompression',async()=>{
  const text=JSON.stringify(bundle([await entry()]));
  const plain=await readImportFile(new File([text],'case-study.json'));
  const compressed=await readImportFile(new File([gzipSync(text)],'case-study.json.gz'));
  assert.deepEqual(plain,compressed);
  const noWrapperHash=await readImportFile(new File([gzipSync(text)],'case-study.json.gz'),undefined,{hash:false});
  assert.equal(noWrapperHash.digest,undefined);
  assert.equal(noWrapperHash.text,plain.text);
  const corrupted=JSON.parse(noWrapperHash.text);corrupted.cases[0].files[0].text+='corrupt';
  await assert.rejects(importBundles([corrupted]),/校验|损坏|不匹配/);
  await assert.rejects(readImportFile(new File([gzipSync('x'.repeat(10000))],'huge.json.gz'),100),/超过/);
  await assert.rejects(readImportFile(new File(['not gzip'],'broken.json.gz')));
});

test('empty benchmark metadata takes precedence over an internal trajectory task id',async()=>{
  const source=await entry();
  source.benchmark='';
  source.files[0]=await file('case.json',{task_id:'42',benchmark:'',mode:'bcg'});
  source.files[1]=await file('agent/trajectories/react_agent.json',{task_id:'internal-uuid',messages:[
    {role:'user',content:'Question'},{role:'assistant',content:'Answer'},
  ]});
  const {cases}=await importBundles([bundle([source])]);
  assert.equal(cases[0].taskId,'42');assert.equal(agentTrajectories(cases[0]).length,1);
});

test('source annotation payloads cannot pre-fill ratings',async()=>{
  const source=await entry();
  source.annotation={result:'正确 / 达标',reviewer:'Generated'};
  const {cases}=await importBundles([bundle([source])]);
  assert.equal(cases[0].annotation.result,'');assert.equal(cases[0].annotation.reviewer,'');
});

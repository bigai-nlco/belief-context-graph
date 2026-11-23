// Creates only synthetic data in /tmp. Its private server has NO scheduler.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, writeFile, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import assert from 'node:assert/strict';

const root=await mkdtemp(join(tmpdir(),'clm-launch-qa-'));
const server=spawn(resolve('.venv/bin/python'),['-u','tools/case_study_studio/tests/shared_smoke_server.py','--root',join(root,'data'),'--port','0'],{stdio:['ignore','pipe','pipe']});
let logs='';server.stdout.on('data',b=>logs+=b);server.stderr.on('data',b=>logs+=b);
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<200;i++){if(await fn())return;await delay(100);}throw new Error(label+' '+logs.slice(-500));}
let browser,ws;const errors=[],pending=new Map();let id=0;
try{
  await until(()=>/Offline QA server ready: (\d+)/.test(logs),'server start');
  const origin='http://127.0.0.1:'+logs.match(/Offline QA server ready: (\d+)/)[1];
  browser=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+join(root,'chrome'),'about:blank'],{stdio:'ignore'});
  await until(async()=>{try{await readFile(join(root,'chrome/DevToolsActivePort'));return true;}catch{return false;}},'browser start');
  const port=(await readFile(join(root,'chrome/DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const cmd=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
  ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(new Error(JSON.stringify(v.error))):p.resolve(v.result);}if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails);});
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=s=>evaluate(`document.querySelector(${JSON.stringify(s)}).click()`);
  const choose=(s,v)=>evaluate(`{const e=document.querySelector(${JSON.stringify(s)});e.value=${JSON.stringify(v)};e.dispatchEvent(new Event('change',{bubbles:true}));}`);
  const shot=async name=>{const r=await cmd('Page.captureScreenshot',{format:'png'});await writeFile(join(root,name),Buffer.from(r.data,'base64'));};
  await cmd('Page.enable');await cmd('Runtime.enable');
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cmd('Page.navigate',{url:origin});
  await until(()=>evaluate('!!document.querySelector("#identity-dialog")?.open'),'identity');
  await evaluate('document.querySelector("#identity-form [name=name]").value="CLM 离线验收"');
  await click('#identity-form [type=submit]');
  await until(()=>evaluate('!!document.querySelector(".run-table")'),'runs');
  await click('#launch-button');await choose('#launch-mode','clm');
  assert(await evaluate('!document.querySelector("#launch-mode option[value=clm]").disabled'));
  assert(await evaluate('document.querySelector("#launch-graph-profile").disabled'));
  assert(await evaluate('!document.querySelector("#launch-task-state").checked'));
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("26,624")'));
  await choose('#launch-profile','gpt6-luna');
  assert.equal(await evaluate('document.querySelector("#launch-form [name=workers]").max'),'6');
  for(const width of [1440,390]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'));
    assert(await evaluate('document.querySelector("#launch-dialog").scrollWidth<=document.querySelector("#launch-dialog").clientWidth+1'));
    await shot('clm-launch-'+width+'.png');
  }
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0002"'),'queued CLM');
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("模式：CLM")'));
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("Graph：不启用")'));
  await shot('clm-queued.png');
  // The same private fixture also verifies BCG Compaction. Scheduling remains
  // disabled; this exercises launch validation and rendering without an LLM.
  await click('#launch-button');await choose('#launch-mode','bcg');
  await choose('#launch-classification','decision_model');
  await choose('#launch-mode','bcg_compaction');
  assert(await evaluate('!document.querySelector("#launch-graph-profile").disabled'));
  assert(await evaluate('document.querySelector("#launch-task-state").checked'));
  assert(await evaluate('!document.querySelector("#launch-compaction-field").hidden'));
  assert(await evaluate('document.querySelector("#launch-classification").disabled && getComputedStyle(document.querySelector("#launch-classification-field")).display==="none"'));
  assert.equal(await evaluate('document.querySelector("#launch-compaction-policy").value'),'recent');
  assert.equal(await evaluate('document.querySelector("#launch-compaction-turns").value'),'5');
  await choose('#launch-mode','default');
  assert(await evaluate('document.querySelector("#launch-compaction-field").hidden&&document.querySelector("#launch-compaction-turns").disabled'));
  await choose('#launch-mode','bcg_compaction');
  await evaluate('document.querySelector("#launch-compaction-turns").value="3";document.querySelector("#launch-compaction-turns").dispatchEvent(new Event("input",{bubbles:true}))');
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("最近 3 轮")'));
  await evaluate('document.querySelector("#launch-compaction-turns").value="5";document.querySelector("#launch-compaction-turns").dispatchEvent(new Event("input",{bubbles:true}))');
  for(const width of [1440,390]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'));
    assert(await evaluate('document.querySelector("#launch-dialog").scrollWidth<=document.querySelector("#launch-dialog").clientWidth+1'));
    assert(await evaluate('getComputedStyle(document.querySelector("#launch-compaction-field")).display!=="none"'));
    await shot('compaction-launch-'+width+'.png');
  }
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0003"'),'queued Compaction');
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("模式：BCG Compaction · 最近 5 轮")'));
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("分类：原始 BCG")'));
  assert(await evaluate('document.querySelector(\'[data-run="3"] .run-mode-label\').textContent.includes("最近 5 轮")'));
  await shot('compaction-queued.png');
  await click('#launch-button');await choose('#launch-compaction-policy','all');
  assert(await evaluate('getComputedStyle(document.querySelector("#launch-compaction-count-field")).display==="none" && document.querySelector("#launch-compaction-turns").disabled'));
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("全部轮次（不淘汰）")'));
  await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0004"'),'queued all-history Compaction');
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("模式：BCG Compaction · 全部轮次（不淘汰）")'));
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("分类：原始 BCG")'));
  assert(await evaluate('document.querySelector(\'[data-run="4"] .run-mode-label\').textContent.includes("全部轮次（不淘汰）")'));
  await click('[data-run="2"]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0002"'),'return to CLM fixture');
  const trial=join(root,'data/results/studio/RUN-0002/drb2/trials/2');
  await mkdir(join(trial,'clm/turns/0001'),{recursive:true});
  const put=(p,v)=>writeFile(join(trial,p),JSON.stringify(v));
  await put('clm/summary.json',{status:'running',task_id:'drb2:2',model_calls:1});
  await put('clm/manifest.json',{settings:{model_name:'openai/gpt-6-luna',context_budget_tokens:28672,context_budget_reserve_tokens:2048}});
  await put('clm/turns/0001/input.json',{messages:[{role:'system',content:'Protected system'},{role:'user',content:'Synthetic task 2'}]});
  await put('clm/turns/0001/output.json',{choices:[{message:{role:'assistant',content:'Synthetic CLM output'}}]});
  await click('#refresh-button');
  await until(()=>evaluate('!!document.querySelector(\'a[href="./clm.html?run=RUN-0002-task-2"]\')'),'CLM record link');
  await click('a[href="./clm.html?run=RUN-0002-task-2"]');
  await until(()=>evaluate('document.querySelector("#clm-model")?.textContent.includes("Synthetic CLM output")'),'managed recording');
  await shot('clm-managed-record.png');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,paidCalls:0,screenshots:root}));
}finally{ws?.close();browser?.kill('SIGTERM');server.kill('SIGTERM');}

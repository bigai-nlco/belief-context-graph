// Read-only browser check. Use an isolated QA state with scheduling disabled.
// Never launches a task, invokes a model, or saves an annotation.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8880';
const run=Number(process.env.STUDIO_RUN||1),task=process.env.STUDIO_TASK||'2';
const dir=await mkdtemp(join(tmpdir(),'bcg-records-browser-'));
const browser=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<450;i++){if(await fn())return;await delay(100);}throw new Error(label);}
let ws,n=0;const pending=new Map(),errors=[];
try{
  await until(async()=>{try{await readFile(join(dir,'DevToolsActivePort'));return true;}catch{return false;}},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const cmd=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(new Error(JSON.stringify(v.error))):p.resolve(v.result);}if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails);});
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const navigate=path=>cmd('Page.navigate',{url:origin+path});
  const shot=async name=>{const image=await cmd('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(image.data,'base64'));};
  await cmd('Page.enable');await cmd('Runtime.enable');
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'identity');
  await evaluate(`{document.querySelector('#identity-form [name=name]').value='只读 QA-'+Date.now();document.querySelector('#identity-form [type=submit]').click();}`);
  await until(()=>evaluate('!!document.querySelector(".run-table")'),'runs');
  await navigate('/?run='+run);
  await until(()=>evaluate(`!!document.querySelector('a[href*="task=${task}&view=records"]')`),'view records entry');
  assert.equal(await evaluate(`document.querySelectorAll('tr[data-live-key="drb2:${task}"] .task-review-actions button:disabled').length`),2);
  await shot('run-records-entry.png');
  await click(`a[href*="task=${task}&view=records"]`);
  await until(()=>evaluate('!!document.querySelector(".records-banner")||!!document.querySelector(".notice h2")'),'records load');
  assert(await evaluate('!!document.querySelector(".records-banner")'),await evaluate('document.querySelector(".content")?.textContent'));
  assert(await evaluate('document.querySelector(".records-banner").textContent.includes("无最终报告")'));
  assert.equal(await evaluate('getComputedStyle(document.querySelector(".annotation-panel")).display'),'none');
  const meta=await evaluate('document.querySelector(".case-meta").textContent');
  await click('[data-tab="trajectory"]');
  assert(await evaluate('document.querySelectorAll(".record").length>0'));
  assert(await evaluate('[...document.querySelectorAll("[data-action=cite]")].every(e=>e.hidden||e.disabled)'));
  await click('[data-tab="context"]');
  assert(await evaluate('document.querySelector("#evidence-content").textContent.includes("Belief")'));
  await shot('run-records-context.png');
  await click('[data-tab="graph"]');
  await until(()=>evaluate('!!document.querySelector(".graph-stats")'),'last saved graph loads');
  const graph=await evaluate('document.querySelector(".graph-stats").textContent');
  assert(await evaluate('document.querySelectorAll(".node-row").length>0'));
  await shot('run-records-graph.png');
  await click('[data-tab="report"]');
  assert(await evaluate('document.querySelector("#evidence-content").textContent.includes("未保存最终报告")'));
  for(const width of [1440,390]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page fits '+width);
    await shot('run-records-report-'+width+'.png');
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,meta,graph,screenshots:dir}));
}finally{ws?.close();browser.kill('SIGTERM');}

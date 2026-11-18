// Read-only browser QA: does not register a user, start a task, or call a model.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const origin=process.env.STUDIO_URL||'http://127.0.0.1:8882';
const dir=await mkdtemp(join(tmpdir(),'clm-viewer-qa-'));
const browser=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<300;i++){if(await fn())return;await delay(100);}throw new Error(label);}
let ws,n=0;const pending=new Map(),errors=[];
try{
  await until(async()=>{try{await readFile(join(dir,'DevToolsActivePort'));return true;}catch{return false;}},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const cmd=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(new Error(JSON.stringify(v.error))):p.resolve(v.result);}if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails);});
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const ready=number=>until(()=>evaluate(`document.querySelector('#clm-panels')?.getAttribute('aria-busy')==='false'&&document.querySelector('#clm-turn-title')?.textContent.includes('第 ${number} 次')&&!!document.querySelector('#clm-context')?.textContent&&!document.querySelector('#clm-context .clm-loading')`),'turn '+number);
  const shot=async name=>{const image=await cmd('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(image.data,'base64'));};
  await cmd('Page.enable');await cmd('Runtime.enable');
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await cmd('Page.navigate',{url:origin+'/clm.html?turn=60'});await ready(60);
  assert.equal(await evaluate('document.querySelectorAll(".clm-turn").length'),70);
  assert(await evaluate('document.querySelector("#clm-context").textContent.includes("VERIFIED")'));
  assert(await evaluate('document.querySelector("#clm-context").textContent.includes("NEXT:")'));
  assert(await evaluate('document.querySelector("#clm-model").textContent.includes("Thinking")'));
  assert(await evaluate('(()=>{const a=document.querySelector(".clm-turn[aria-current=step]").getBoundingClientRect(),b=document.querySelector("#clm-turn-list").getBoundingClientRect();return a.top>=b.top&&a.bottom<=b.bottom;})()'));
  await shot('context-output-desktop.png');
  await click('[data-context="diff"]');
  assert(await evaluate('document.querySelectorAll(".clm-diff-add").length>0'));
  await shot('context-diff-desktop.png');
  await click('#edits-only');
  assert.equal(await evaluate('document.querySelectorAll(".clm-turn").length'),4);
  await click('#prev-turn');await ready(48);
  await click('[data-context="after"]');
  await click('[data-turn="60"]');await ready(60);
  await click('#show-next-input');await ready(61);
  assert(await evaluate('document.querySelector("#clm-model").textContent.includes("VERIFIED")'));
  assert(await evaluate('document.querySelector("#clm-model").textContent.includes("system")'));
  await click('[data-model="raw-input"]');
  assert(await evaluate('document.querySelector("#clm-model pre").textContent.includes("chat_template_kwargs")'));
  assert.equal(await evaluate('document.querySelector("#edits-only").checked'),false);
  await click('[data-turn="1"]');await ready(1);
  // First editable context is empty (only the pinned prefix exists).
  assert(await evaluate('document.querySelector("#clm-context").textContent.includes("可编辑历史为空")'));
  await click('[data-section="report"]');
  await until(()=>evaluate('!!document.querySelector("#download-report")'),'report');
  assert(await evaluate('document.querySelectorAll("#clm-report-section .report-markdown h1").length>0'));
  await shot('report-desktop.png');
  await click('[data-section="settings"]');
  assert(await evaluate('document.querySelector("#clm-settings-section").textContent.includes("26,624")'));
  await click('[data-section="turns"]');await click('[data-turn="60"]');await ready(60);
  await click('[data-model="output"]');
  for(const width of [1440,1024,390]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:1080,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'page fits '+width);
    const box=await evaluate('(()=>{const [a,b]=[...document.querySelectorAll(".clm-panel")].map(x=>x.getBoundingClientRect());return {overlap:Math.min(a.right,b.right)>Math.max(a.left,b.left)&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)};})()');
    assert.equal(box.overlap,false);
    await shot('context-output-'+width+'.png');
  }
  await cmd('Page.navigate',{url:origin+'/'});
  await until(()=>evaluate('!!document.querySelector(\'a[href="./clm.html"]\')'),'run center link');
  assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'run center mobile width');
  await shot('run-center-mobile.png');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,turns:70,edits:4,screenshots:dir}));
}finally{ws?.close();browser.kill('SIGTERM');}

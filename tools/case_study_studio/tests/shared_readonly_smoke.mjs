// Read-only evidence smoke. Creates one reviewer in the supplied QA state only.
// NEVER launches jobs, saves annotations, or changes experiment files.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8880';
const dir=await mkdtemp(join(tmpdir(),'bcg-real-preview-'));
const browser=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<300;i++){try{if(await fn())return;}catch{}await delay(100);}throw new Error(label);}
let ws,n=0;const pending=new Map();
try{
  await until(async()=>{await readFile(join(dir,'DevToolsActivePort'));return true;},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const cmd=(method,params={})=>new Promise((resolve,reject)=>{const id=++n;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(new Error(JSON.stringify(v.error))):p.resolve(v.result);}});
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description);return r.result.value;};
  await cmd('Page.enable');await cmd('Runtime.enable');await cmd('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'identity');
  await evaluate(`{document.querySelector('#identity-form [name=name]').value='只读预览-'+Date.now();document.querySelector('#identity-form [type=submit]').click();}`);
  await until(()=>evaluate('!!document.querySelector(".run-table a")'),'real runs');
  let screenshot=await cmd('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,'real-runs.png'),Buffer.from(screenshot.data,'base64'));
  const runs=await evaluate(`[...document.querySelectorAll('.run-item')].map(e=>({number:e.querySelector('strong').textContent,text:e.textContent}))`);
  await evaluate(`document.querySelector('.run-table a').click()`);
  await until(()=>evaluate('!!document.querySelector(".workbench")'),'real bundle opens');
  const summary=await evaluate(`({task:document.querySelector('.case-heading h1').textContent,stats:document.querySelector('.case-meta').textContent,reviewer:document.querySelector('[data-field=reviewer]')?.readOnly,sidebarDisplay:getComputedStyle(document.querySelector('.shared-sidebar .case-item')).display})`);
  assert.equal(summary.reviewer,true);assert.equal(summary.sidebarDisplay,'block');
  for(const tab of ['context','graph','report']){
    await evaluate(`document.querySelector('[data-tab=${tab}]').click()`);
    assert(await evaluate('document.querySelector("#evidence-content").textContent.length>10'));
  }
  await evaluate(`document.querySelector('[data-action=help]').click()`);
  assert(await evaluate('document.querySelector("#guide-step-heading-0").textContent.includes("运行编号")'));
  await evaluate(`document.querySelector('#help-dialog').close()`);
  screenshot=await cmd('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,'real-case.png'),Buffer.from(screenshot.data,'base64'));
  console.log(JSON.stringify({ok:true,runs,summary,screenshots:dir},null,2));
}finally{ws?.close();browser.kill('SIGTERM');}

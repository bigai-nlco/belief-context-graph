// Only shared_smoke_server.py --mock-summary. Never calls a real provider.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8877';
assert.equal((await(await fetch(origin+'/api/settings')).json()).run_summary_offline_fixture,true,'refuse production');
const dir=await mkdtemp(join(tmpdir(),'bcg-run-summary-browser-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<300;i++){try{if(await fn())return;}catch{}await pause(100);}throw Error(label);}
let ws,next=0,summaryPosts=0;const pending=new Map(),errors=[];
try{
  await until(async()=>{await readFile(join(dir,'DevToolsActivePort'));return true;},'Chrome');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);message.error?p.reject(Error(JSON.stringify(message.error))):p.resolve(message.result);}if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);if(message.method==='Network.requestWillBeSent'&&message.params.request.method==='POST'&&new URL(message.params.request.url).pathname.endsWith('/annotation-summary'))summaryPosts++;});
  await command('Page.enable');await command('Runtime.enable');await command('Network.enable');
  const evaluate=async expression=>{const result=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw Error(result.exceptionDetails.exception?.description||result.exceptionDetails.text);return result.result.value;};
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const shot=async name=>{const r=await command('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(r.data,'base64'));};
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'login');
  await evaluate(`document.querySelector('#identity-form [name=name]').value='Summary browser QA';document.querySelector('#identity-form [type=submit]').click()`);
  await until(()=>evaluate('!!document.querySelector("a[href*=annotation-stats]")'),'statistics entry');
  await click('a[href*=annotation-stats]');
  await until(()=>evaluate('document.querySelector("#stats-report")?.contentDocument?.querySelectorAll(".cohort").length===1'),'deterministic statistics first');
  await until(()=>evaluate('document.querySelector("#summary-generate")?.hidden===false'),'manual generation button');
  assert.equal(summaryPosts,0,'opening statistics never generates a summary');
  await click('#stats-refresh');
  await until(()=>evaluate('!document.querySelector("#stats-refresh").disabled&&!document.querySelector("#summary-generate").hidden'),'statistics refresh without generation');
  assert.equal(summaryPosts,0,'refresh without saved summary remains read-only');
  await shot('before-generation.png');
  await click('#summary-generate');
  await until(()=>summaryPosts===1,'one explicit summary request');
  await click('#stats-refresh');
  await until(()=>evaluate('!document.querySelector("#stats-refresh").disabled'),'refresh while summary is queued/running');
  await until(()=>evaluate('document.querySelector("#summary-status-text")?.textContent.includes("已完成")'),'summary completes');
  await until(()=>evaluate('!!document.querySelector("#stats-report").contentDocument.querySelector(".run-ai-summary")'),'summary displayed');
  const headings=await evaluate(`[...document.querySelector('#stats-report').contentDocument.querySelectorAll('.run-ai-summary h4')].map(e=>e.textContent)`);
  assert.deepEqual(headings,['1. 当前 Run 的主要利用问题','2. 固定维度检查','3. 未充分利用的具体表现与可能原因','4. 证据边界','5. 可能的改进方案']);
  assert(await evaluate('document.querySelector("#stats-report").contentDocument.querySelector(".run-ai-summary").textContent.includes("待验证假设")'));
  assert(await evaluate('document.querySelector("#stats-report").contentDocument.querySelector(".run-ai-summary").textContent.includes("1/1（100.0%）")'));
  for(const width of [1440,1024,390,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('document.querySelector("#stats-report").contentDocument.querySelector(".run-ai-summary").scrollIntoView({block:"start"})');
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'outer layout '+width);
    assert(await evaluate('(()=>{const d=document.querySelector("#stats-report").contentDocument;return d.documentElement.scrollWidth<=d.documentElement.clientWidth+1;})()'),'report layout '+width);
    await shot('summary-'+width+'.png');
  }
  await click('#stats-refresh');
  await until(()=>evaluate('!document.querySelector("#stats-refresh").disabled'),'statistics refresh complete');
  await until(()=>evaluate('document.querySelector("#summary-status-text").textContent.includes("AI 问题总结已完成")'),'cached summary');
  assert.equal(summaryPosts,1,'refreshing pending/completed summaries never submits a new job');
  await command('Page.reload');
  await until(()=>evaluate('document.querySelector("#summary-status-text")?.textContent.includes("AI 问题总结已完成")'),'saved summary loads after full-page reload');
  assert.equal(summaryPosts,1,'full-page reload is read-only');
  await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:dir});
  await click('#stats-download');
  await until(async()=>{try{return (await readFile(join(dir,'RUN-0001-annotation-stats.zip'))).subarray(0,2).toString()==='PK';}catch{return false;}},'summary-inclusive ZIP');
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({ok:true,screenshots:dir},null,2));
}finally{ws?.close();chrome.kill('SIGTERM');}

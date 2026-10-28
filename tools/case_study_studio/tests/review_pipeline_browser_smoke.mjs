// Read-only UI checks. No review submissions, model requests or experiment writes.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8877';
const contract=await(await fetch(origin+'/review-pipeline-contract.json')).json();
const dir=await mkdtemp(join(tmpdir(),'bcg-review-method-browser-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<200;i++){try{if(await fn())return;}catch{}await delay(100);}throw new Error(label);}
let ws,next=0;const pending=new Map(),errors=[],requests=[];
try{
  await until(async()=>{await readFile(join(dir,'DevToolsActivePort'));return true;},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{
    const d=JSON.parse(e.data);
    if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}
    if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails);
    if(d.method==='Log.entryAdded'&&d.params.entry.level==='error')errors.push(d.params.entry.text);
    if(d.method==='Network.requestWillBeSent')requests.push(d.params.request);
  });
  await command('Page.enable');await command('Page.bringToFront');await command('Runtime.enable');await command('Log.enable');await command('Network.enable');
  const evaluate=async expression=>{const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=s=>evaluate(`document.querySelector(${JSON.stringify(s)}).click()`);
  const userClick=async selector=>{
    const point=await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.scrollIntoView({block:'center',behavior:'instant'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    await command('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});
    await command('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});
  };
  const shot=async name=>{const r=await command('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(r.data,'base64'));};
  const size=width=>command('Emulation.setDeviceMetricsOverride',{width,height:1050,deviceScaleFactor:1,mobile:false});
  const scrollTo=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start',behavior:'instant'})`);
  await size(1600);await command('Page.navigate',{url:origin+'/review-pipeline.html'});
  await until(()=>evaluate('document.querySelectorAll(".flow-node").length===9'),'method page loads');
  assert.equal(await evaluate('document.querySelectorAll(".stage-nav button").length'),9);
  assert.equal(await evaluate('document.querySelector(".stage-sheet").id'),'detail-intake');
  await shot('method-overview.png');

  for(const id of ['intake','align','batch','find','find-gate','pool','synthesize','summary-gate','publish']){
    await click(`.stage-nav [data-stage="${id}"]`);
    assert.equal(await evaluate('document.querySelector(".stage-sheet").id'),'detail-'+id);
    assert.equal(await evaluate('document.querySelectorAll(".stage-sheet .method-table").length'),2);
    assert(await evaluate('document.querySelector(".handoff").textContent.length>40'));
    await click('.stage-sheet .source-link');
    assert.equal(await evaluate('document.querySelector("#source-dialog").open'),true);
    assert(await evaluate('document.querySelectorAll(".source-line").length>2'));
    await command('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});
    await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});
    await until(()=>evaluate('!document.querySelector("#source-dialog").open'),'Escape closes source viewer');
  }
  await click('.stage-nav [data-stage="find-gate"]');
  await click('.stage-sheet .guard-box summary');
  await scrollTo('#pipeline');await shot('method-validation.png');
  await click('.stage-sheet .example-io summary');
  for(const text of await evaluate('[...document.querySelectorAll(".stage-sheet .code-panel pre")].map(e=>e.textContent)'))assert.doesNotThrow(()=>JSON.parse(text));
  await click('[data-expand]');assert.equal(await evaluate('document.querySelectorAll(".stage-sheet").length'),9);
  await click('[data-expand]');assert.equal(await evaluate('document.querySelectorAll(".stage-sheet").length'),1);

  for(let turn=0;turn<3;turn++){
    await click(`[data-turn="${turn}"]`);
    assert.equal(await evaluate('document.querySelector(".verdict-pill").textContent'),turn===0?'合理核验':'无效重复');
    assert(await evaluate(`document.querySelector('#turn-example').textContent.includes('F00${turn+1}')`));
  }
  for(const stage of ['findings','summary'])for(const view of ['template','schema','request']){
    await click(`[data-prompt-stage="${stage}"]`);await click(`[data-prompt-view="${view}"]`);
    const content=await evaluate('document.querySelector("#contract-display pre").textContent');
    if(view==='template'){assert(content.includes(contract.prompts.system));assert(content.includes(contract.prompts[stage]));}
    if(view==='schema')assert.deepEqual(JSON.parse(content),contract.schemas[stage]);
    if(view==='request'){
      const body=JSON.parse(content);assert.deepEqual(body.messages,contract.example.requests[stage]);
      assert.equal(body.model,contract.model.model);assert.equal(body.reasoning_effort,contract.model.reasoning_effort);
      assert(!('tools' in body));assert(!('api_key' in body));
    }
  }
  await userClick('[data-copy="contract"]');
  await until(()=>evaluate('document.querySelector("#notice").textContent.includes("已复制")'),'copy works');
  for(const width of [1600,1440,1024,760,390]){
    await size(width);
    await evaluate('window.scrollTo({top:0,behavior:"instant"})');
    if(!await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')){
      console.log(await evaluate(`[...document.querySelectorAll('main *,header *')].filter(e=>{const r=e.getBoundingClientRect();return r.right>innerWidth+1&&!e.closest('pre')&&!e.closest('.method-table');}).slice(0,25).map(e=>({tag:e.tagName,class:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width}))`));
      await shot('overflow-'+width+'.png');
    }
    assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'no page overflow at '+width);
    assert(await evaluate(`(()=>{const h=document.querySelector('.site-header'),r=h.getBoundingClientRect();return [...h.children].every(e=>{const b=e.getBoundingClientRect();return b.right<=r.right&&b.left>=r.left;});})()`),'header stays inside viewport');
    await shot('method-top-'+width+'.png');
    await scrollTo('#pipeline');await shot('method-stage-'+width+'.png');
    await scrollTo('#example');await shot('method-example-'+width+'.png');
    assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'no page overflow after scrolling at '+width);
  }
  await size(1440);
  await command('Page.navigate',{url:origin+'/review-pipeline.html#stage-synthesize'});
  await until(()=>evaluate('document.querySelector(".stage-sheet")?.id==="detail-synthesize"'),'deep link');
  await evaluate('window.dispatchEvent(new Event("beforeprint"))');
  assert.equal(await evaluate('document.querySelectorAll(".stage-sheet").length'),9,'print expands all steps');
  await evaluate('window.dispatchEvent(new Event("afterprint"))');
  assert.equal(await evaluate('document.querySelectorAll(".stage-sheet").length'),1);
  const pdf=await command('Page.printToPDF',{printBackground:true,preferCSSPageSize:true});
  await writeFile(join(dir,'agent-review-pipeline.pdf'),Buffer.from(pdf.data,'base64'));
  assert.equal(requests.filter(r=>r.method!=='GET').length,0,'page must never submit a job');
  assert.equal(requests.filter(r=>r.url.includes('/api/')).length,0,'method page needs no live user data');
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({ok:true,screenshots:dir,checkedWidths:[1600,1440,1024,760,390],modelRequests:0},null,2));
}finally{ws?.close();chrome.kill('SIGTERM');}

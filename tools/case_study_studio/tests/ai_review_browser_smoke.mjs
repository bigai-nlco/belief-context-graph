// Only shared_smoke_server.py --mock-ai. Never starts real model requests.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8877';
assert.equal((await(await fetch(origin+'/api/settings')).json()).ai_review_offline_fixture,true,'refuse to run against production');
const dir=await mkdtemp(join(tmpdir(),'bcg-ai-review-browser-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:'ignore'});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<300;i++){try{if(await fn())return;}catch{}await delay(100);}throw new Error(label);}
let ws,next=0;const pending=new Map(),errors=[],requests=[];
try{
  await until(async()=>{await readFile(join(dir,'DevToolsActivePort'));return true;},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails);if(d.method==='Network.requestWillBeSent')requests.push(d.params.request.url);});
  await command('Page.enable');await command('Runtime.enable');await command('Network.enable');
  await command('Page.addScriptToEvaluateOnNewDocument',{source:`
    if(new URLSearchParams(location.search).get('review')==='ai'){
      for(const name of ['get','getAll']){const read=IDBObjectStore.prototype[name];
        IDBObjectStore.prototype[name]=function(...args){if(this.name==='cases')throw Error('AI page read personal drafts');return read.apply(this,args);};}
    }`});
  await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
  const evaluate=async expression=>{const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=s=>evaluate(`document.querySelector(${JSON.stringify(s)}).click()`);
  const shot=async name=>{const r=await command('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(r.data,'base64'));};
  const navigate=path=>command('Page.navigate',{url:origin+path});
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'identity');
  await evaluate(`document.querySelector('#identity-form [name=name]').value='AI Review QA '+Date.now();document.querySelector('#identity-form [type=submit]').click()`);
  await until(()=>evaluate('!!document.querySelector("[data-ai-review=\\"2\\"]")'),'review button');
  assert.equal(await evaluate('document.querySelector("[data-ai-review=\\"5\\"]").disabled'),true);
  assert.equal(await evaluate('document.querySelector("[data-ai-review=\\"2\\"]").textContent'),'AI 标注');
  assert.equal(await evaluate('document.querySelector(".run-table a[href*=\\"task=2\\"]").textContent'),'人工标注');
  // Preview server states in this offline browser only, without writing reviews.
  await evaluate(`window.annotationTestFetch=window.fetch;window.annotationPreview={};
    window.fetch=async(...args)=>{
      const response=await window.annotationTestFetch(...args);
      if(args[0]!=='/api/runs/1')return response;
      const data=await response.json();Object.assign(data.tasks.find(t=>t.id==='2'),window.annotationPreview);
      return new Response(JSON.stringify(data),{status:response.status,headers:{'Content-Type':'application/json'}});
    };`);
  const preview=async(data,label)=>{
    await evaluate(`window.annotationPreview=${JSON.stringify(data)};document.dispatchEvent(new Event('visibilitychange'));`);
    await until(()=>evaluate(`document.querySelector('.run-table tbody tr:first-child [data-annotation-status]')?.textContent===${JSON.stringify(label)}`),'annotation state '+label);
  };
  const progressValues=()=>evaluate(`[...document.querySelectorAll('.run-progress progress')].map(e=>({id:e.id,value:e.value,max:e.max}))`);
  for(const status of ['timeout','agent_error','graph_fallback','interrupted']){
    for(const aiStatus of ['not_started','completed']){
      await preview({status,annotatable:true,report:true,reviews:[],ai_review:{status:aiStatus}},aiStatus==='completed'?'AI 已标注':'未标注');
      await until(()=>evaluate(`document.querySelector('.run-table tbody tr:first-child td:nth-child(2) .run-status').classList.contains(${JSON.stringify(status)})`),'failed task state '+status);
      const actions=await evaluate(`(()=>{const e=document.querySelector('.run-table tbody tr:first-child .task-review-actions');return {links:e.querySelectorAll('a').length,buttons:[...e.querySelectorAll('button')].map(b=>({label:b.textContent,disabled:b.disabled}))};})()`);
      assert.deepEqual(actions,{links:0,buttons:[{label:'人工标注',disabled:true},{label:'AI 标注',disabled:true}]},'failed tasks cannot open either annotation, even with a saved AI review');
      assert.deepEqual((await progressValues())[1],{id:'run-annotation-progress',value:0,max:1},'no eligible tasks leaves an empty determinate annotation bar');
      assert(await evaluate('document.querySelector(".progress-annotation").textContent.includes("暂无可标注任务")'));
    }
  }
  await shot('failed-task-disabled.png');
  await preview({status:'completed',annotatable:true,reviews:[],ai_review:{status:'completed'}},'AI 已标注');
  await until(()=>evaluate('!!document.querySelector(".ai-review-link")'),'successful task annotation restored');
  assert.equal(await evaluate('document.querySelector(".ai-review-link").textContent'),'查看 AI 标注');
  assert.equal(await evaluate('document.querySelector(".annotation-sources")'),null);
  assert(await evaluate('document.querySelector(".run-progress").textContent.includes("已标注 1 / 1 题")'));
  assert.deepEqual(await progressValues(),[{id:'run-test-progress',value:1,max:2},{id:'run-annotation-progress',value:1,max:1}],'execution and annotation have independent totals');
  assert.equal(await evaluate('document.querySelector(".review-names").textContent.includes("暂无人工标注")'),false);
  await shot('annotation-ai-only.png');
  await preview({reviews:[{name:'Reviewer',status:'reviewed'}],ai_review:{status:'not_started'}},'人工已标注');
  assert.equal((await progressValues())[1].value,1,'human annotation advances the annotation bar');
  await preview({reviews:[{name:'Reviewer',status:'reviewed'}],ai_review:{status:'completed'}},'AI 和人工已标注');
  assert.equal((await progressValues())[1].value,1,'human plus AI counts only once');
  for(const width of [1920,1440,1024,390]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1050,deviceScaleFactor:1,mobile:false});
    assert(await evaluate(`[...document.querySelectorAll('.run-table th,.run-table td')].every(e=>{const s=getComputedStyle(e);return s.textAlign==='center'&&s.verticalAlign==='middle';})`),'all table headers and cells are centered');
    assert(await evaluate(`[...document.querySelectorAll('.run-table .run-status,.run-table .button')].every(e=>{const r=e.getBoundingClientRect(),c=e.closest('th,td').getBoundingClientRect();return Math.abs(r.left+r.width/2-c.left-c.width/2)<1;})`),'status badges and action buttons stay centered in their columns');
    assert(await evaluate(`(()=>{const s=document.querySelector('[data-annotation-status]'),c=s.closest('td'),r=s.getBoundingClientRect(),b=c.getBoundingClientRect();return s.scrollWidth<=s.clientWidth+1&&r.left>=b.left&&r.right<=b.right&&r.height<30;})()`),'combined annotation label fits on one line');
    assert(await evaluate(`(()=>{
      const card=document.querySelector('.run-progress'),bars=[...card.querySelectorAll('progress')].map(e=>e.getBoundingClientRect());
      return card.scrollWidth<=card.clientWidth+1&&bars[0].bottom<bars[1].top&&Math.abs(bars[0].width-bars[1].width)<1&&Math.abs(bars[0].left-bars[1].left)<1&&
        [...card.querySelectorAll('.progress-track-heading')].every(e=>{const label=e.querySelector('label').getBoundingClientRect(),count=e.querySelector('span').getBoundingClientRect();return label.right<=count.left||label.bottom<=count.top;});
    })()`),'progress bars align and labels do not overlap');
    await shot('dual-progress-'+width+'.png');
  }
  await preview({reviews:[],ai_review:{status:'running',completed_batches:5,batches:5}},'标注中');
  assert.equal((await progressValues())[1].value,0,'AI work in progress is not counted as completed');
  for(const width of [1920,1440,1024,390]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1050,deviceScaleFactor:1,mobile:false});
    await evaluate('document.querySelector(".run-table-wrap").scrollLeft=99999');
    const layout=await evaluate(`(()=>{
      const button=document.querySelector('.ai-review-busy'),label=button.querySelector('.ai-review-label'),spinner=button.querySelector('.ai-review-spinner');
      const l=label.getBoundingClientRect(),s=spinner.getBoundingClientRect(),cell=button.closest('td').getBoundingClientRect();
      return {spinnerBelow:s.top>=l.bottom+2,centered:Math.abs(s.left+s.width/2-l.left-l.width/2)<1,
        lines:[...document.querySelectorAll('.ai-review-progress>span')].map(e=>{const r=e.getBoundingClientRect(),style=getComputedStyle(e);return {text:e.textContent,height:r.height,lineHeight:parseFloat(style.lineHeight),fits:e.scrollWidth<=e.clientWidth+1&&r.left>=cell.left&&r.right<=cell.right};}),
        tableOverflow:[...document.querySelectorAll('.run-table td')].some(e=>e.scrollWidth>e.clientWidth+1)};
    })()`);
    assert(layout.spinnerBelow && layout.centered,'spinner is centered on its own line below the label: '+JSON.stringify(layout));
    assert.equal(layout.tableOverflow,false,'table cells do not overflow');
    assert.equal(layout.lines.length,2,'batch count and synthesis are separate lines');
    for(const line of layout.lines){assert(line.fits,'progress fits its column');assert(line.height<=line.lineHeight+1,'progress text never wraps');}
    await shot('annotation-progress-'+width+'.png');
  }
  await evaluate('window.fetch=window.annotationTestFetch');
  await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1050,deviceScaleFactor:1,mobile:false});
  await evaluate('document.dispatchEvent(new Event("visibilitychange"))');
  await until(()=>evaluate('!!document.querySelector("[data-ai-review=\\"2\\"]")'),'restore real fixture state');
  await click('.run-table a[href*="task=2"]');
  await until(()=>evaluate('!!document.querySelector("[data-field=goal]")'),'human workbench');
  await evaluate(`{const e=document.querySelector('[data-field=goal]');e.value='人工标注保持独立';e.dispatchEvent(new Event('input',{bubbles:true}));}`);
  await until(()=>evaluate('document.querySelector("#save-label").textContent==="已保存到服务器"'),'human saved');
  await navigate('/?run=1');
  await until(()=>evaluate('!!document.querySelector("[data-ai-review=\\"2\\"]")'),'return to runs');
  await click('[data-ai-review="2"]');
  await until(()=>evaluate('!!document.querySelector(".ai-review-busy")'),'review in background');
  await shot('review-running.png');
  await until(()=>evaluate('!!document.querySelector(".ai-review-link")'),'review finished');
  assert.equal(await evaluate('document.querySelector(".ai-review-link").textContent'),'查看 AI 标注');
  assert.equal(await evaluate('document.querySelector(".run-table tbody tr:first-child [data-annotation-status]").textContent'),'AI 已标注');
  await shot('review-completed.png');
  const initialEvidence=requests.filter(url=>url.endsWith('/evidence')).length;
  await click('.ai-review-link');
  await until(()=>evaluate('!!document.querySelector(".ai-review-banner")'),'AI workbench loaded');
  assert.equal(requests.filter(url=>url.endsWith('/evidence')).length,initialEvidence,'AI initial load never downloads full graph');
  assert(requests.some(url=>url.endsWith('/evidence-preview')),'initial AI result loads the small preview');
  assert(await evaluate('document.querySelector(".case-meta").textContent.includes("按需加载")'));
  assert.equal(await evaluate('document.querySelector(".case-item.active .case-item-bottom").textContent'),'查看 AI 标注');
  assert(await evaluate('document.querySelector(".ai-review-banner").textContent.includes("gpt-6.1-sol")'));
  assert(await evaluate('document.querySelector(".event-belief-content").textContent.includes("B1")'));
  assert(await evaluate('document.querySelector(".event-report-quote").textContent.includes("方案一适合较小样本")'));
  assert.equal(await evaluate('document.querySelector("[data-action=delete-event]").hidden'),true);
  assert.equal(await evaluate('document.querySelector("[data-action=complete]").hidden'),true);
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-action=complete]")).display'),'none');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("[data-action=cite]")).display'),'none');
  assert.equal(await evaluate('document.querySelectorAll("[data-field]").length'),0);
  await shot('review-events.png');
  const beliefBefore=await evaluate('document.querySelector(".event-belief-content").textContent');
  await evaluate(`window.originalEvidenceFetch=fetch;window.failGraphOnce=true;window.fetch=async(...args)=>{
    if(String(args[0]).endsWith('/evidence')&&window.failGraphOnce){window.failGraphOnce=false;return Response.json({error:'Offline retry test'},{status:500});}
    return window.originalEvidenceFetch(...args);
  }`);
  await click('[data-tab=graph]');
  await until(()=>evaluate('document.querySelector(".deferred-graph")?.textContent.includes("Offline retry test")'),'lazy Graph failure is retryable');
  assert.equal(await evaluate('document.querySelector(".event-belief-content").textContent'),beliefBefore,'Graph failure does not remove annotation evidence');
  await click('[data-action=load-full-evidence]');
  await until(()=>evaluate('!!document.querySelector(".graph-stats")'),'full Graph loaded only on demand');
  assert.equal(await evaluate('document.querySelector(".event-belief-content").textContent'),beliefBefore,'rebound annotation still shows the same Belief');
  assert.equal(requests.filter(url=>url.endsWith('/evidence')).length,initialEvidence+1);
  await shot('lazy-full-graph.png');
  await click('[data-tab=context]');
  assert(await evaluate('document.querySelector("#evidence-content").textContent.includes("B1")'));
  for(const stage of ['dimensions','conclusion','basic']){
    await click(`[data-stage=${stage}]`);
    assert.equal(await evaluate('document.querySelectorAll("#form-content input:not(:disabled),#form-content select:not(:disabled),#form-content textarea:not(:disabled)").length'),0);
    if(stage==='dimensions'){
      assert.equal(await evaluate('document.querySelectorAll(".ai-dimension-card").length'),1);
      assert.equal(await evaluate('document.querySelector("#form-content .process-options")'),null,'AI dimensions link evidence without repeating the full event checklist');
      assert.equal(await evaluate('document.querySelector(".ai-dimension-note").open'),false);
    }
    if(stage==='conclusion'){
      assert(await evaluate('document.querySelector("#form-content").textContent.includes("重点与改进方向")'));
      assert.equal(await evaluate('document.querySelector("#form-content").textContent.includes("回顾已记录的评价")'),false);
    }
    await shot('review-'+stage+'.png');
  }
  assert(await evaluate('document.querySelector("#form-content").textContent.includes("任务结果（AI 评估）")'));
  assert.equal(await evaluate('[...document.querySelectorAll("#form-content .ai-review-field")].find(e=>e.textContent.includes("标注人"))?.querySelector(".ai-review-value")?.textContent'),'gpt-6.1-sol');
  assert.equal(await evaluate('document.querySelector(".ai-task-assessment>strong").textContent'),'正确 / 达标');
  assert(await evaluate('document.querySelector(".ai-task-rationale").textContent.includes("小样本和长期观察")'));
  assert(await evaluate('document.querySelector(".ai-task-limitations").textContent.includes("未独立核查")'));
  await click('.ai-task-assessment summary');
  assert.deepEqual(await evaluate('[...document.querySelectorAll(".ai-task-assessment blockquote")].map(e=>e.textContent)'),['比较两个研究方案的适用场景','方案一适合较小样本。方案二适合长期观察。']);
  for(const width of [1440,1024,390]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'AI task assessment no horizontal overflow');
    assert(await evaluate('(()=>{const e=document.querySelector(".ai-task-assessment");return e.scrollWidth<=e.clientWidth+1;})()'),'assessment text and quotes stay inside their card');
    await evaluate('document.querySelector(".ai-task-assessment").scrollIntoView({block:"center"})');
    await shot('task-assessment-'+width+'.png');
  }
  await evaluate('window.beforeAIRefresh=true');
  const beforeReload=requests.filter(url=>/\/evidence(?:-preview)?$/.test(url)).length;
  await command('Page.reload');
  await until(()=>evaluate('!window.beforeAIRefresh && !!document.querySelector(".ai-review-banner")'),'AI survives refresh and fresh source IDs');
  assert.equal(requests.filter(url=>/\/evidence(?:-preview)?$/.test(url)).length,beforeReload,'cached immutable preview skips evidence download and parse on repeat open');
  assert(await evaluate(`(async()=>{
    const cache=await import('./evidence-cache.mjs');const review=await(await fetch('/api/runs/1/tasks/drb2/2/ai-review',{headers:{Authorization:'Bearer '+localStorage.getItem('bcg-studio-personal-token')}})).json();
    const value=await cache.readPreviewCache(cache.previewCacheKey({run:1,benchmark:'drb2',task:'2'},review.evidence_sha));
    return value.c.annotation.events.length===0&&!value.c.annotation.reviewer&&!value.c.remote;
  })()`),'preview cache contains no annotation or identity');
  await click('[data-stage=events]');
  assert(await evaluate('document.querySelector(".event-belief-content").textContent.includes("B1")'));
  for(const width of [1440,1024,390]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),'AI workbench no horizontal overflow');
    await shot('review-width-'+width+'.png');
  }
  await evaluate(`window.exportedWorkspace=null;window.originalObjectURL=URL.createObjectURL;URL.createObjectURL=function(blob){blob.text().then(text=>{window.exportedWorkspace=JSON.parse(text);});return window.originalObjectURL(blob);}`);
  await click('[data-action=export-json]');
  await until(()=>evaluate('!!window.exportedWorkspace'),'complete export from preview');
  assert(await evaluate('window.exportedWorkspace.cases[0].graphs.length>0'),'JSON workspace export hydrates the full graph');
  assert.equal(requests.filter(url=>url.endsWith('/evidence')).length,initialEvidence+2,'export downloads missing graph once');
  await click('.ai-review-banner a');
  await until(()=>evaluate('!!document.querySelector("[data-field=goal]")'),'human annotation restored');
  assert.equal(await evaluate('document.querySelector("[data-field=goal]").value'),'人工标注保持独立');
  assert.equal(await evaluate('document.querySelector(".shared-review-links a").textContent'),'查看 AI 标注');
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({ok:true,screenshots:dir},null,2));
}finally{ws?.close();chrome.kill('SIGTERM');}

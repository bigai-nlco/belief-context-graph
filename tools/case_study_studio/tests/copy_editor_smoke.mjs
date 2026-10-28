// Browser test for the real same-origin editor. No model calls or case uploads.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import assert from 'node:assert/strict';
const origin=process.env.STUDIO_URL||'http://127.0.0.1:8876';
const root=await mkdtemp(join(tmpdir(),'bcg-copy-editor-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--disable-component-update','--no-first-run','--remote-debugging-port=0','--user-data-dir='+root,'about:blank'],{stdio:['ignore','ignore','pipe']});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function eventually(fn,message,tries=150){for(let i=0;i<tries;i++){try{if(await fn())return;}catch{}await pause(100);}throw new Error(message);}
let ws,id=0;const pending=new Map(),errors=[],requests=[];
try{
  await eventually(async()=>!!await readFile(join(root,'DevToolsActivePort'),'utf8'),'Chrome startup');
  const port=(await readFile(join(root,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin+'/copy-editor.html')}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
  ws.addEventListener('message',event=>{
    const d=JSON.parse(event.data);
    if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}
    if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails.exception?.description||d.params.exceptionDetails.text);
    if(d.method==='Network.requestWillBeSent')requests.push(d.params.request.url);
    if(d.method==='Page.javascriptDialogOpening')command('Page.handleJavaScriptDialog',{accept:true});
  });
  await command('Runtime.enable');await command('Page.enable');await command('Network.enable');
  await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:root});
  await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false});
  const evaluate=async expression=>{
    const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;
  };
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const child=expression=>evaluate(`(()=>{const w=document.querySelector('#preview').contentWindow;const d=w.document;return ${expression};})()`);
  const childClick=selector=>child(`d.querySelector(${JSON.stringify(selector)}).click()`);
  const input=(selector,value)=>evaluate(`{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('input',{bubbles:true}));}`);
  const screenshot=async name=>{const r=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(root,name),Buffer.from(r.data,'base64'));};
  const choose=async(original,kind='text')=>{
    await input('#copy-search',original);
    await evaluate(`{const b=[...document.querySelectorAll('#copy-list button')].find(b=>{const k=JSON.parse(b.dataset.copyKey);return k[0]===${JSON.stringify(kind)}&&k[1]===${JSON.stringify(original)};});if(!b)throw new Error('copy missing');b.click();}`);
  };
  await eventually(()=>evaluate('document.querySelectorAll("#copy-list button").length>200'),'full copy catalog did not load');
  assert.equal(await child("!!d.querySelector('.welcome')"),true);
  await screenshot('editor-start.png');
  // Click-to-edit with actual frame coordinates.
  const point=await evaluate(`(()=>{const f=document.querySelector('#preview'),r=f.getBoundingClientRect(),p=f.contentDocument.querySelector('.welcome h1').getBoundingClientRect();return{x:r.x+p.x+30,y:r.y+p.y+20};})()`);
  await command('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});
  await command('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});
  await eventually(()=>evaluate('!document.querySelector("#selected-copy").hidden'),'picker did not select');
  await choose('标注教程');
  await input('#replacement-copy','新手标注指南');await click('#apply-copy');
  await eventually(()=>child("d.querySelector('.sidebar-foot button').textContent.includes('新手标注指南')"),'live copy not applied');
  assert.equal(await evaluate('JSON.parse(localStorage.getItem("bcg-studio-copy-draft-v1")).entries.length'),1);
  // Tutorial is modal inside the iframe, editor remains reachable outside it.
  await click('[data-jump="guide"]');
  await eventually(()=>child("d.querySelector('#help-dialog').open"),'guide did not open');
  await click('#pick-mode');
  await choose('完成第一份 Case 标注');await input('#replacement-copy','从这里开始标注');await click('#apply-copy');
  await eventually(()=>child("d.querySelector('#help-title').textContent==='从这里开始标注'"),'tutorial copy not applied');
  await screenshot('editor-tutorial.png');
  // Dynamic counters retain values; form placeholders and option labels can change.
  await click('[data-jump="demo"]');
  await eventually(()=>child("!!d.querySelector('.case-heading')"),'real demo did not load');
  await childClick('[data-form="basic"]');
  await choose('姓名 / 编号','placeholder');await input('#replacement-copy','填写标注者姓名');await click('#apply-copy');
  await eventually(()=>child("d.querySelector('[data-field=reviewer]').placeholder==='填写标注者姓名'"),'placeholder not applied');
  await choose('正确 / 达标');await input('#replacement-copy','符合任务要求');await click('#apply-copy');
  await eventually(()=>child("[...d.querySelector('[data-field=result]').options].some(o=>o.textContent==='符合任务要求'&&o.value==='正确 / 达标')"),'option label changed its saved value');
  await choose('轨迹');await input('#replacement-copy','<img src=x onerror="window.__copyXss=true">');await click('#apply-copy');
  await eventually(()=>child("d.querySelector('.evidence-panel .panel-heading h2').textContent.startsWith('<img')"),'text preview not applied');
  assert.equal(await child('w.__copyXss===true'),false);
  assert.equal(await child('!!d.querySelector(".evidence-panel .panel-heading h2 img")'),false);
  await click('#reset-entry');
  await eventually(()=>child("d.querySelector('.evidence-panel .panel-heading h2').textContent==='轨迹'"),'reset entry failed');
  // Catalog excludes task identity and original evidence even after traversing views.
  await childClick('[data-tab="context"]');await pause(150);
  await childClick('[data-tab="graph"]');await pause(150);
  await childClick('[data-tab="trajectory"]');await pause(150);
  await input('#copy-search','site:nasa.gov Europa Clipper launched');
  assert.equal(await evaluate('document.querySelectorAll("#copy-list button").length'),0,'tool arguments leaked into editable copy');
  await childClick('[data-tab="report"]');await pause(150);
  await input('#copy-search','Europa Clipper');
  assert.equal(await evaluate('document.querySelectorAll("#copy-list button").length>0'),true); // static tutorial discusses this topic
  const exportedKeys=await evaluate('[...document.querySelectorAll("#copy-list button")].map(b=>JSON.parse(b.dataset.copyKey)[1])');
  assert(!exportedKeys.some(k=>k.includes('frontier_ddd76')||k.includes('Europa Clipper 与 JUICE：')));
  assert.equal(await child('d.querySelector(".report-markdown").hasAttribute("data-copy-lock")'),true);
  await input('#copy-search','两项任务的互补首先');
  assert.equal(await evaluate('document.querySelectorAll("#copy-list button").length'),0,'rendered report leaked into editable copy');
  await click('#export-copy');
  await eventually(async()=> (await readdir(root)).includes('bcg-ui-copy.json'),'export missing');
  const exported=JSON.parse(await readFile(join(root,'bcg-ui-copy.json'),'utf8'));
  assert.equal(exported.format,'bcg-ui-copy-draft');assert.equal(exported.entries.length,4);
  assert.equal(JSON.stringify(exported).includes('final_response'),false);
  await command('Page.reload');
  await eventually(()=>child("d.querySelector('.sidebar-foot button')?.textContent.includes('新手标注指南')"),'draft was not restored');
  // Standard page never uses browser-local drafts.
  await command('Page.navigate',{url:origin+'/'});
  await eventually(()=>evaluate('document.querySelector(".sidebar-foot button")?.textContent.includes("标注教程")'),'standard page missing');
  assert.equal(await evaluate('document.querySelector(".sidebar-foot button").textContent.includes("新手标注指南")'),false);
  await command('Page.navigate',{url:origin+'/copy-editor.html'});
  await eventually(()=>evaluate('document.querySelectorAll("#copy-list button").length>200'),'editor reload');
  await click('#reset-all');await eventually(()=>evaluate('document.querySelector("#change-count").textContent==="0 项修改"'),'reset draft failed');
  const doc=await command('DOM.getDocument'),node=await command('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#copy-file'});
  await command('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[join(root,'bcg-ui-copy.json')]});
  await eventually(()=>evaluate('document.querySelector("#change-count").textContent==="4 项修改"'),'import failed');
  await command('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:true});
  assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'mobile overflow');
  await screenshot('editor-mobile.png');
  assert.deepEqual(errors,[]);
  assert.equal(requests.filter(url=>!url.startsWith(origin)&&!url.startsWith('blob:')&&!url.startsWith('data:')).length,0);
  console.log(JSON.stringify({ok:true,catalog:'200+ entries',checks:['real click picker','tutorial modal editing','draft autosave/reload','placeholder editing','option value preservation','literal HTML/XSS safety','source evidence excluded','normal page isolated','JSON export/import','reset','mobile layout'],screenshots:root},null,2));
}finally{ws?.close();chrome.kill('SIGTERM');}

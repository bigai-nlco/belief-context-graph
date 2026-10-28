// Use only with shared_smoke_server.py: its scheduler is disabled and data is synthetic.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const origin=process.env.STUDIO_URL || 'http://127.0.0.1:8877';
const dir=await mkdtemp(join(tmpdir(),'bcg-shared-browser-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--remote-debugging-port=0','--user-data-dir='+dir,'about:blank'],{stdio:['ignore','ignore','ignore']});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label){for(let i=0;i<150;i++){try{if(await fn())return;}catch{}await pause(100);}throw new Error(label);}
let ws,next=0;const pending=new Map(),errors=[];
try{
  await until(async()=>{try{await readFile(join(dir,'DevToolsActivePort'));return true;}catch{return false;}},'Chrome start');
  const port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  ws.addEventListener('message',e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result);}if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails);if(d.method==='Page.javascriptDialogOpening')command('Page.handleJavaScriptDialog',{accept:true});});
  await command('Page.enable');await command('Runtime.enable');
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const evaluate=async expression=>{const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
  const click=s=>evaluate(`document.querySelector(${JSON.stringify(s)}).click()`);
  const fill=(s,value)=>evaluate(`{const e=document.querySelector(${JSON.stringify(s)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('input',{bubbles:true}));}`);
  const navigate=async path=>{await command('Page.navigate',{url:origin+path});};
  const shot=async name=>{const r=await command('Page.captureScreenshot',{format:'png'});await writeFile(join(dir,name),Buffer.from(r.data,'base64'));};
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'identity prompt');
  const identityLayout=()=>evaluate(`(()=>{
    const dialog=document.querySelector('#identity-dialog'),title=document.querySelector('#identity-title');
    const button=dialog.querySelector('[data-close]'),icon=button.querySelector('svg');
    const box=e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,cx:r.left+r.width/2,cy:r.top+r.height/2};};
    return {dialog:box(dialog),title:box(title),button:box(button),icon:box(icon),overflow:dialog.scrollWidth>dialog.clientWidth+1,pageWidth:document.documentElement.scrollWidth};
  })()`);
  const checkIdentityLayout=async width=>{
    const layout=await identityLayout();
    assert.equal(layout.overflow,false,'identity dialog does not overflow horizontally');
    assert(layout.dialog.left>=15 && layout.dialog.right<=width-15,'identity dialog has viewport margins');
    assert(layout.title.right+10<=layout.button.left,'title does not overlap close button');
    assert(Math.abs(layout.title.cy-layout.button.cy)<1,'title and close button align');
    assert(Math.abs(layout.icon.cx-layout.button.cx)<.5 && Math.abs(layout.icon.cy-layout.button.cy)<.5,'identity close icon is centered');
    assert(layout.pageWidth<=width+1,'long reviewer names do not widen the page');
  };
  for(const width of [1440,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
    await checkIdentityLayout(width);
    assert.equal(await evaluate('getComputedStyle(document.querySelector("#identity-current")).display'),'none');
    await shot('identity-signin-'+width+'.png');
  }
  const reviewerName=('离线标注甲-'+Date.now()+'-ABCDEFGHIJKLMNOPQRSTUVWXYZ').slice(0,40);
  await fill('#identity-form [name=name]',reviewerName);await click('#identity-form [type=submit]');
  await until(()=>evaluate('!!document.querySelector(".run-table")'),'run list');
  const aliceToken=await evaluate('localStorage.getItem("bcg-studio-personal-token")');
  const api=async(path,token=aliceToken,method='GET',body)=>{const r=await fetch(origin+'/api/'+path,{method,headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const data=await r.json();assert(r.ok,JSON.stringify(data));return data;};
  await click('#identity-button');
  assert.equal(await evaluate('document.querySelector("#identity-title").textContent'),'标注者信息');
  assert.equal(await evaluate('document.querySelector("#identity-name").textContent'),reviewerName);
  assert.equal(await evaluate('document.querySelectorAll("#identity-dialog [data-close]").length'),1,'one persistent close button');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("#identity-form")).display'),'none');
  for(const width of [1440,430,360,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
    await checkIdentityLayout(width);
    assert.equal(await evaluate('getComputedStyle(document.querySelector("#identity-token-panel")).display'),'none');
    await shot('identity-profile-'+width+'.png');
    await click('#show-token');
    assert(await evaluate('document.querySelector("#identity-token").value===localStorage.getItem("bcg-studio-personal-token")'),'explicit reveal shows the current identity code');
    assert.equal(await evaluate('document.querySelector("#show-token").getAttribute("aria-expanded")'),'true');
    await checkIdentityLayout(width);
    assert(await evaluate('document.querySelector("#identity-token").getBoundingClientRect().bottom<document.querySelector(".identity-footer").getBoundingClientRect().top'),'revealed code stays above footer');
    await click('#show-token');
    assert.equal(await evaluate('document.querySelector("#identity-token").value'),'','hiding clears the identity code');
  }
  await click('#show-token');await click('#identity-dialog [data-close]');
  await until(()=>evaluate('document.querySelector("#identity-token").value===""'),'closing clears the identity code');
  await click('#identity-button');
  assert.equal(await evaluate('document.querySelector("#show-token").getAttribute("aria-expanded")'),'false');
  await click('#show-token');
  await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await until(()=>evaluate('!document.querySelector("#identity-dialog").open && document.querySelector("#identity-token").value===""'),'Escape closes and hides the identity code');
  await click('#identity-button');await click('#logout-button');
  assert.equal(await evaluate('document.querySelector("#identity-title").textContent'),'标注身份');
  assert.equal(await evaluate('getComputedStyle(document.querySelector("#identity-current")).display'),'none');
  await fill('#identity-form [name=token]',aliceToken);await click('#identity-form [type=submit]');
  await until(()=>evaluate('!document.querySelector("#identity-dialog").open'),'restore same reviewer by identity code');
  assert(await evaluate('localStorage.getItem("bcg-studio-personal-token")')===aliceToken,'switching and restoring preserves the original identity');
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  assert.equal(await evaluate('document.querySelector(".run-heading .eyebrow").textContent'),'RUN-0001');
  await shot('runs-desktop.png');
  assert.equal(await evaluate('document.querySelector(".run-table tbody strong").textContent'),'DeepResearch Bench II');
  // Exercise digit growth on fixture DOM only; never change saved experiment data.
  const tableLayout=async value=>evaluate(`(()=>{
    const table=document.querySelector('.run-table'),wrap=table.parentElement;
    table.querySelectorAll('.token-total').forEach(e=>e.textContent=${JSON.stringify(value)});
    table.querySelectorAll('.token-breakdown>span').forEach((e,i)=>e.textContent=(i%2?'输出 ':'输入 ')+${JSON.stringify(value)});
    const box=table.getBoundingClientRect();
    return {width:box.width,scrollWidth:wrap.scrollWidth,scrollLeft:wrap.scrollLeft,columns:[...table.querySelectorAll('th')].map(e=>e.getBoundingClientRect().width),overflow:[...table.querySelectorAll('td')].some(e=>e.scrollWidth>e.clientWidth+1),text:table.querySelector('.token-total').textContent};
  })()`);
  for(const width of [1920,1440,1024,360]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('document.querySelector(".run-table-wrap").scrollLeft=100');
    const before=await tableLayout('0');
    for(const value of ['—','1,000','9,999,999','9,876,543,210','9,007,199,254,740,991']){
      const after=await tableLayout(value);
      assert.equal(after.width,before.width,'token growth does not change table width');
      assert.deepEqual(after.columns,before.columns,'token growth does not move columns');
      assert.equal(after.scrollWidth,before.scrollWidth,'token growth does not widen scrolling area');
      assert.equal(after.scrollLeft,before.scrollLeft,'token growth preserves horizontal position');
      assert.equal(after.overflow,false,'complete token values stay inside cells');
      assert.equal(after.text,value,'token values are not abbreviated');
    }
    await tableLayout('9,876,543,210');
    await evaluate('document.querySelector(".run-table-wrap").scrollLeft=0');
    await shot('stable-token-columns-'+width+'.png');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const refreshState=()=>evaluate('document.querySelector("#refresh-button").dataset.state');
  const refreshWidth=await evaluate('document.querySelector("#refresh-button").getBoundingClientRect().width');
  await click('#refresh-button');
  assert.equal(await refreshState(),'loading');
  assert(await evaluate('document.querySelector("#refresh-button").disabled'));
  assert.equal(await evaluate('document.querySelector("#refresh-button").getAttribute("aria-busy")'),'true');
  const rotation=await evaluate('getComputedStyle(document.querySelector(".refresh-spinner")).transform');
  await pause(150);
  assert.notEqual(await evaluate('getComputedStyle(document.querySelector(".refresh-spinner")).transform'),rotation);
  await shot('refresh-loading.png');
  await until(async()=>await refreshState()==='success','refresh success feedback');
  assert.equal(await evaluate('document.querySelector(".refresh-label").textContent'),'已刷新');
  assert.equal(await evaluate('document.querySelector("#refresh-button").getBoundingClientRect().width'),refreshWidth);
  assert.equal(await evaluate('document.querySelector("#refresh-button").disabled'),false);
  await shot('refresh-success.png');
  await until(async()=>await refreshState()==='idle','refresh feedback clears');
  // Join an in-flight background request; repeated clicks must not duplicate it.
  await evaluate(`window.refreshTestFetch=window.fetch;window.refreshTestRequests=0;
    window.fetch=async(...args)=>{if(args[0]==='/api/runs'){window.refreshTestRequests++;await new Promise(resolve=>window.releaseRefreshTest=resolve);}return window.refreshTestFetch(...args);};
    document.dispatchEvent(new Event('visibilitychange'));`);
  await until(()=>evaluate('typeof window.releaseRefreshTest==="function"'),'background request started');
  await click('#refresh-button');await click('#refresh-button');await pause(750);
  assert.equal(await refreshState(),'loading');
  assert.equal(await evaluate('window.refreshTestRequests'),1);
  await evaluate('window.fetch=window.refreshTestFetch;window.releaseRefreshTest()');
  await until(async()=>await refreshState()==='success','manual refresh waits for real completion');
  // A failure must restore the button and never claim success.
  await evaluate(`window.fetch=async(...args)=>args[0]==='/api/runs'?new Response(JSON.stringify({error:'QA simulated refresh failure'}),{status:503,headers:{'Content-Type':'application/json'}}):window.refreshTestFetch(...args)`);
  await click('#refresh-button');
  await until(async()=>await refreshState()==='error','refresh failure feedback');
  assert.equal(await evaluate('document.querySelector("#refresh-button").disabled'),false);
  assert.equal(await evaluate('document.querySelector("#refresh-button").getAttribute("aria-busy")'),'false');
  await shot('refresh-error.png');
  await evaluate('window.fetch=window.refreshTestFetch');
  await command('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  await click('#refresh-button');
  assert.equal(await refreshState(),'loading');
  assert.equal(await evaluate('getComputedStyle(document.querySelector(".refresh-spinner")).animationName'),'none');
  await until(async()=>await refreshState()==='success','refresh retry succeeds');
  await until(async()=>await refreshState()==='idle','refresh retry feedback clears');
  await command('Emulation.setEmulatedMedia',{features:[]});
  await click('#launch-button');
  for(const width of [1440,360]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    const close=await evaluate(`(()=>{
      const button=document.querySelector('[data-close=launch-dialog]');
      const b=button.getBoundingClientRect(),icon=button.querySelector('svg').getBoundingClientRect();
      return {dx:icon.left+icon.width/2-b.left-b.width/2,dy:icon.top+icon.height/2-b.top-b.height/2,label:button.getAttribute('aria-label')};
    })()`);
    assert(Math.abs(close.dx)<.5 && Math.abs(close.dy)<.5,'launch close icon is centered at '+width+'px');
    assert.equal(close.label,'关闭');
    await shot('launch-close-'+width+'.png');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const suggestedTitle=await evaluate('document.querySelector("#launch-form [name=title]").value');
  assert.match(suggestedTitle,/^DeepResearch Bench II 测试 · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
  assert(await evaluate('document.querySelector("#launch-form [name=title]").clientWidth>document.querySelector("#launch-dialog").clientWidth*.8'),'timestamped name gets a full-width field');
  await fill('#launch-form [name=title]','自定义测试名称');
  await click('[data-close=launch-dialog]');await click('#launch-button');
  assert.equal(await evaluate('document.querySelector("#launch-form [name=title]").value'),'自定义测试名称','reopening does not discard a custom name');
  await fill('#launch-form [name=title]',suggestedTitle);
  assert.equal(await evaluate('document.querySelectorAll("#launch-form [name=drbench]").length'),0);
  assert.equal(await evaluate('document.querySelectorAll("#launch-form [name=admin]").length'),0);
  assert.equal((await api('settings')).launch_requires_admin,false);
  assert(await evaluate('document.querySelector("#launch-form [name=drb2]").required'));
  const launchSettings=await api('settings');
  assert.equal(await evaluate('document.querySelector("#launch-form [name=drb2]").value'),launchSettings.default_tasks.drb2.join(','),'launch form uses the server default task list');
  assert.equal(launchSettings.default_tasks.drb2.length,32);
  assert.equal(await evaluate('document.querySelector("#launch-max-tasks").textContent'),'32');
  await fill('#launch-form [name=drb2]','2');
  assert.equal((await api('runs')).runs.length,1);
  await shot('launch-dialog.png');await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0002"'),'queued batch');
  assert.equal((await api('runs/2')).status,'queued');
  assert.equal((await api('runs/2')).title,suggestedTitle,'timestamped name is persisted');
  await click('#launch-button');
  assert.match(await evaluate('document.querySelector("#launch-form [name=title]").value'),/^DeepResearch Bench II 测试 · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
  await click('[data-close=launch-dialog]');
  // Exercise active/cancelling labels in the fixture browser, without starting a worker.
  const runTitle=await evaluate('document.querySelector(".run-heading h1").textContent');
  const headingLayout=async(stopping=false,longTitle=false)=>evaluate(`(()=>{
    const title=document.querySelector('.run-heading h1'),actions=document.querySelector('.run-heading-actions');
    const status=actions.querySelector('.run-status'),stop=document.querySelector('#stop-run');
    title.textContent=${longTitle?JSON.stringify('A'.repeat(100)):JSON.stringify(runTitle)};
    status.textContent='运行中';status.className='run-status running';
    stop.textContent=${JSON.stringify(stopping?'正在停止…':'停止测试')};stop.disabled=${stopping};
    const box=e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,cy:r.top+r.height/2};};
    return {title:box(title),actions:box(actions),status:box(status),stop:box(stop),width:innerWidth,pageWidth:document.documentElement.scrollWidth};
  })()`);
  for(const width of [1440,1024,768,430,360]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    const layout=await headingLayout();
    assert(Math.abs(layout.status.cy-layout.stop.cy)<1,'status and stop button share a centerline');
    assert(layout.status.right+8<=layout.stop.left,'status and stop button have a visible gap');
    assert(layout.actions.right<=layout.width && layout.actions.left>=0,'actions stay inside the viewport');
    assert(layout.pageWidth<=layout.width+1,'heading does not widen the page');
    if(width<=680)assert(layout.actions.top>=layout.title.bottom+10,'mobile actions sit below the title');
    else assert(layout.title.right<=layout.actions.left || layout.title.bottom<=layout.actions.top,'title and actions do not overlap');
    await shot('run-heading-'+width+'.png');
    const extended=await headingLayout(true,true);
    assert(extended.pageWidth<=width+1,'long titles and stopping text do not overflow');
    assert(Math.abs(extended.status.cy-extended.stop.cy)<1,'stopping label stays aligned');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await headingLayout();
  await click('#stop-run');
  assert.equal(await evaluate('document.querySelectorAll("#cancel-form [name=admin]").length'),0);
  await shot('cancel-dialog.png');await click('#cancel-form [type=submit]');
  await until(async()=> (await api('runs/2')).status==='cancelled','cancel queue');
  await navigate('/?run=1');await until(()=>evaluate('!!document.querySelector(".run-table a")'),'historical results');
  // Hold only the synthetic evidence response to inspect the real loading screen.
  let heldEvidence;
  const holdEvidence=e=>{const d=JSON.parse(e.data);if(d.method==='Fetch.requestPaused')heldEvidence=d.params.requestId;};
  ws.addEventListener('message',holdEvidence);
  await command('Fetch.enable',{patterns:[{urlPattern:'*/api/runs/1/tasks/drb2/2/evidence',requestStage:'Request'}]});
  await click('.run-table a');
  await until(()=>!!heldEvidence,'case evidence request paused');
  for(const [width,height] of [[1440,1000],[360,800]]){
    await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    const loading=await evaluate(`(()=>{
      const card=document.querySelector('.workbench-loading-card'),box=card.getBoundingClientRect();
      return {cx:box.left+box.width/2,cy:box.top+box.height/2,left:box.left,right:box.right,role:card.getAttribute('role'),text:card.textContent.trim(),pageWidth:document.documentElement.scrollWidth};
    })()`);
    assert(Math.abs(loading.cx-width/2)<1 && Math.abs(loading.cy-height/2)<1,'workbench loader centers in the viewport');
    assert(loading.left>=16 && loading.right<=width-16 && loading.pageWidth<=width,'loading card fits the viewport');
    assert.equal(loading.role,'status');
    assert.equal(loading.text,'正在打开标注工作台…');
    await shot('workbench-loading-'+width+'.png');
  }
  const loaderRotation=await evaluate('getComputedStyle(document.querySelector(".workbench-loading-ring")).transform');
  await pause(150);
  assert.notEqual(await evaluate('getComputedStyle(document.querySelector(".workbench-loading-ring")).transform'),loaderRotation,'loading ring animates');
  await command('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  assert.equal(await evaluate('getComputedStyle(document.querySelector(".workbench-loading-ring")).animationName'),'none');
  await command('Emulation.setEmulatedMedia',{features:[]});
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await command('Fetch.continueRequest',{requestId:heldEvidence});
  await command('Fetch.disable');ws.removeEventListener('message',holdEvidence);
  await until(()=>evaluate('!!document.querySelector("[data-field=goal]")'),'case loads automatically');
  assert.equal(await evaluate('document.querySelector(".workbench-loading")'),null,'loading screen is removed after opening');
  assert(await evaluate('document.querySelector("[data-field=reviewer]").readOnly'));
  assert.equal(await evaluate('document.querySelectorAll(".topbar [data-action=import]").length'),0);
  await fill('[data-field=goal]','甲：检查 BCG 已有信息是否被复用');
  await until(async()=> (await api('runs/1/tasks/drb2/2/annotation')).annotation?.goal?.startsWith('甲：'),'server autosave');
  await until(()=>evaluate('document.querySelector("#save-label").textContent.includes("已保存到服务器")'),'save label');
  await shot('shared-case.png');
  await navigate('/index.html?run=1&benchmark=drb2&task=2');
  await until(()=>evaluate('document.querySelector("[data-field=goal]")?.value.startsWith("甲：")'),'persist reload');
  const bob=await api('session','', 'POST',{name:'离线标注乙-'+Date.now()});
  await evaluate(`localStorage.setItem('bcg-studio-personal-token',${JSON.stringify(bob.token)})`);
  await fill('[data-field=goal]','不可越权写入');
  await until(()=>evaluate('document.querySelector("#save-label").classList.contains("save-error")'),'identity switch cannot overwrite');
  assert.equal((await api('runs/1/tasks/drb2/2/annotation')).annotation.goal,'甲：检查 BCG 已有信息是否被复用');
  await navigate('/index.html?run=1&benchmark=drb2&task=2');
  await until(()=>evaluate('document.querySelector("[data-field=reviewer]")?.value.startsWith("离线标注乙")'),'second reviewer');
  assert.equal(await evaluate('document.querySelector("[data-field=goal]").value'),'');
  await fill('[data-field=goal]','乙：独立分析');
  await until(async()=> (await api('runs/1/tasks/drb2/2/annotation',bob.token)).annotation?.goal==='乙：独立分析','second review saved');
  assert.equal((await api('runs/1')).annotation_count,2);
  // A second tab updates this reviewer. Stale browser editing must not win.
  const prior=await api('runs/1/tasks/drb2/2/annotation',bob.token);
  await api('runs/1/tasks/drb2/2/annotation',bob.token,'PUT',{...prior,annotation:{...prior.annotation,goal:'另一页的新标注'}});
  await fill('[data-field=goal]','当前页的过期标注');
  await until(()=>evaluate('document.querySelector("#save-label").classList.contains("save-error")'),'revision conflict visible');
  assert.equal((await api('runs/1/tasks/drb2/2/annotation',bob.token)).annotation.goal,'另一页的新标注');
  await click('[data-action=reload-shared]');
  await until(()=>evaluate('document.querySelector("[data-field=goal]")?.value==="另一页的新标注"'),'explicit conflict recovery');
  await navigate('/');await until(()=>evaluate('!!document.querySelector(".run-table")'),'back to dashboard');
  await command('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:true});
  assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'mobile no page overflow');
  await shot('runs-mobile.png');
  await navigate('/index.html');await until(()=>evaluate('!!document.querySelector(".welcome")'),'local workspace remains separate');
  assert.equal(await evaluate('document.querySelectorAll(".case-item").length'),0);
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log('Shared browser smoke passed: refresh animation/success/error/retry/reduced motion, background-request coalescing, launch/cancel without admin code, automatic case load, separate reviewers, persistence, conflict recovery, local mode.');
  console.log('Screenshots: '+dir);
}finally{ws?.close();chrome.kill('SIGTERM');}

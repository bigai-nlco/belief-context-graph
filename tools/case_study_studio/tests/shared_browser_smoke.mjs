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
  for(const selector of ['.runs-actions a[href="./review-pipeline.html"]','.run-heading-actions a[href*="annotation-stats.html"]']){
    assert.equal(await evaluate(`getComputedStyle(document.querySelector(${JSON.stringify(selector)})).textDecorationLine`),'none','navigation buttons have no text underline');
  }
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
  assert.equal(await evaluate('document.querySelector(".run-version-missing").textContent'),'代码版本：未记录');
  // Synthetic provenance overlays the offline response, not any saved run.
  await evaluate(`window.versionOriginalFetch=window.fetch;
    window.versionFixture={schema_version:1,commit:'a'.repeat(40),branch:'feat/'+('long-branch-'.repeat(12)),dirty:true,
      repository_url:'https://github.com/example/project',source_sha256:'b'.repeat(64),agent_sha256:'c'.repeat(64),
      graph_sha256:'d'.repeat(64),lock_sha256:'e'.repeat(64),fingerprint_complete:true,captured_at:'2026-11-03T00:00:00Z'};
    window.fetch=async(...args)=>{const response=await window.versionOriginalFetch(...args);
      const url=String(args[0]);if(!['/api/runs','/api/runs/1'].includes(url))return response;
      const data=await response.json();
      if(url==='/api/runs')data.runs.forEach(run=>run.code_version=window.versionFixture);
      else data.code_version=window.versionFixture;
      return Response.json(data);};document.dispatchEvent(new Event('visibilitychange'));`);
  await until(()=>evaluate('!!document.querySelector(".run-version")'),'run version appears');
  assert(await evaluate('document.querySelector(".run-item .run-version-label").textContent.includes("aaaaaaaa")'));
  await click('.run-version>summary');
  assert(await evaluate('document.querySelector(".run-version").open'));
  assert.equal(await evaluate('document.querySelector(".version-body a").getAttribute("href")'),'https://github.com/example/project/commit/'+'a'.repeat(40));
  await evaluate(`document.dispatchEvent(new Event('visibilitychange'))`);
  await pause(3500);
  assert(await evaluate('document.querySelector(".run-version").open'),'polling preserves expanded version details');
  for(const width of [1440,430,360,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'long branch and hashes never widen the page');
    assert(await evaluate(`(()=>{const p=document.querySelector('.run-version'),r=p.getBoundingClientRect();return p.scrollWidth<=p.clientWidth+1&&r.left>=0&&r.right<=innerWidth;})()`),'version card stays inside viewport');
    await shot('run-version-'+width+'.png');
  }
  await click('.run-version>summary');
  await evaluate('window.fetch=window.versionOriginalFetch;document.dispatchEvent(new Event("visibilitychange"))');
  await until(()=>evaluate('!!document.querySelector(".run-version-missing")'),'legacy unknown version restored');
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await shot('runs-desktop.png');
  assert.equal(await evaluate('document.querySelector(".run-table tbody strong").textContent'),'DeepResearch Bench II');
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('.run-table tbody tr:first-child .task-agent-tokens .token-breakdown>span')].map(e=>e.textContent)`),['输入 100','输出 20']);
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('.run-table tbody tr:first-child .task-graph-tokens .token-breakdown>span')].map(e=>e.textContent)`),['输入 30','输出 10']);
  assert.deepEqual(await evaluate(`[...document.querySelectorAll('.run-table tbody tr:nth-child(2) .task-graph-tokens .token-breakdown>span')].map(e=>e.textContent)`),['输入 —','输出 —']);
  assert.equal(await evaluate('document.querySelectorAll(".run-table th")[2].textContent'),'分数');
  assert(await evaluate('[...document.querySelectorAll(".run-table tbody tr")].every(row=>row.cells.length===9)'),'each task has a dedicated score cell');
  assert.equal(await evaluate('document.querySelector(".run-table tbody tr:nth-child(2) .task-score").textContent'),'—');
  // Simulate scoring transitions in the fixture response, without changing any saved result.
  assert.equal(await evaluate('document.querySelector(".run-score-total").textContent'),'—');
  assert.equal(await evaluate('document.querySelector(".run-score-coverage").textContent'),'已评分 0 / 2 题');
  await evaluate(`window.scoreTestFetch=window.fetch;window.scoreTestTask={};window.scoreTestTask2={};
    window.fetch=async(...args)=>{const response=await window.scoreTestFetch(...args);
      if(String(args[0])!=='/api/runs/1')return response;
      const data=await response.json();Object.assign(data.tasks[0],window.scoreTestTask);Object.assign(data.tasks[1],window.scoreTestTask2);return Response.json(data);};`);
  for(const [scoring_status,official_score,label] of [
    ['official_protocol_scored',0,'0.0 / 100'],
    ['official_protocol_scored',0.10091743119266056,'10.1 / 100'],
    ['official_protocol_scored',1,'100.0 / 100'],
    ['judge_error',null,'评分失败'],
    ['not_judged',null,'未评分'],
    ['official_pending',null,'待评分'],
  ]){
    await evaluate(`window.scoreTestTask=${JSON.stringify({scoring_status,official_score})};document.dispatchEvent(new Event('visibilitychange'));`);
    await until(()=>evaluate(`document.querySelector('.run-table tbody tr:first-child .task-score').textContent===${JSON.stringify(label)}`),'score cell '+label);
    assert(await evaluate(`!document.querySelector('.run-table tbody tr:first-child .task-status').textContent.includes('评分')`),'score is not duplicated under execution status');
    const scored=scoring_status==='official_protocol_scored';
    assert.equal(await evaluate('document.querySelector(".run-score-total").textContent'),scored?(official_score*100).toFixed(1):'—');
    assert.equal(await evaluate('document.querySelector(".run-score-coverage").textContent'),`已评分 ${scored?1:0} / 2 题`);
    if(official_score===0.10091743119266056)await shot('score-column.png');
  }
  await evaluate(`window.scoreTestTask={scoring_status:'official_protocol_scored',official_score:1};
    window.scoreTestTask2={scoring_status:'official_protocol_scored',official_score:.5,status:'completed'};
    document.dispatchEvent(new Event('visibilitychange'));`);
  await until(()=>evaluate('document.querySelector(".run-score-total").textContent==="150.0"'),'sum updates after second score');
  assert.equal(await evaluate('document.querySelector(".run-score-average").textContent'),'已评分题均分 75.0 / 100');
  assert.equal(await evaluate('document.querySelector(".run-score-coverage").textContent'),'已评分 2 / 2 题');
  for(const width of [1440,1100,430,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert.equal(await evaluate('document.querySelectorAll(".run-stats>.run-stat").length'),5);
    assert(await evaluate(`(()=>{const p=document.querySelector('.run-stats').getBoundingClientRect();
      return [...document.querySelectorAll('.run-stat')].every(c=>{const r=c.getBoundingClientRect();
        return r.left>=p.left-1&&r.right<=p.right+1&&c.scrollWidth<=c.clientWidth+1;});})()`),'all five stat cards fit at '+width);
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'score card does not widen page');
    await shot('score-total-'+width+'.png');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate('window.fetch=window.scoreTestFetch;document.dispatchEvent(new Event("visibilitychange"))');
  // Saved Judge explanations load only on demand. All data below is synthetic.
  await evaluate(`window.scoreDetailOriginalFetch=window.fetch;window.scoreDetailRequests=0;
    window.scoreDetailMode='normal';window.scoreDetailFixture={run:'RUN-0001',benchmark:'drb2',task_id:'2',status:'scored',score:1/3,model:'offline-judge',reasoning_effort:'high',judged_at:'2026-10-28T12:00:00Z',duration_seconds:20,dimensions:[
      {id:'info_recall',label:'信息召回',score:.5,items:[{criterion:'Criterion A: preserve precise evidence',score:1,reason:'Reason A',evidence:'Original report evidence A'},{criterion:'Criterion B: a missing constraint',score:0,reason:'Reason B',evidence:''}]},
      {id:'analysis',label:'分析',score:0,items:[{criterion:'Criterion C: use permitted references',score:-1,reason:'Blocked citation',evidence:'Original report evidence C'}]},
      {id:'presentation',label:'表达呈现',score:null,items:[]}]};
    window.fetch=async(...args)=>{
      if(String(args[0])==='/api/runs/1/tasks/drb2/2/score'){
        window.scoreDetailRequests++;
        if(window.scoreDetailMode==='fail')return Response.json({error:'QA read failure'},{status:503});
        if(window.scoreDetailMode==='delayed')await new Promise(resolve=>window.releaseScoreDetail=resolve);
        return Response.json(window.scoreDetailFixture);
      }
      const response=await window.scoreDetailOriginalFetch(...args);
      if(String(args[0])!=='/api/runs/1')return response;
      const data=await response.json();Object.assign(data.tasks[0],{score_details_available:true,scoring_status:'official_protocol_scored',official_score:1/3});return Response.json(data);
    };document.dispatchEvent(new Event('visibilitychange'));`);
  await until(()=>evaluate('!!document.querySelector("[data-score-task=\\"2\\"]")'),'score view button');
  assert.equal(await evaluate('window.scoreDetailRequests'),0,'polling never fetches explanations');
  assert.equal(await evaluate('document.querySelector("[data-score-task=\\"5\\"]")'),null,'no view button without a saved Judge record');
  await click('[data-score-task="2"]');
  await until(()=>evaluate('document.querySelectorAll("#score-dialog .score-criterion").length===3'),'all saved rubric items');
  assert.equal(await evaluate('window.scoreDetailRequests'),1);
  assert(await evaluate('document.querySelector("#score-dialog").textContent.includes("Reason A")'));
  assert(await evaluate('document.querySelector("#score-dialog").textContent.includes("Original report evidence A")'));
  assert.equal(await evaluate('document.querySelectorAll("#score-dialog [data-evidence-status=not-required]").length'),1);
  assert.equal(await evaluate('document.querySelectorAll("#score-dialog [data-evidence-status=missing]").length'),0);
  assert(await evaluate('document.querySelector("#score-dialog .score-legend").textContent.includes("从 Agent 最终报告中摘录")'));
  for(const width of [1440,390,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:false});
    const layout=await evaluate(`(()=>{const d=document.querySelector('#score-dialog'),c=document.querySelector('#score-dialog-content'),b=d.querySelector('[data-close]'),svg=b.querySelector('svg');const r=d.getBoundingClientRect(),h=b.getBoundingClientRect(),s=svg.getBoundingClientRect();return {left:r.left,right:r.right,bottom:r.bottom,overflow:d.scrollWidth>d.clientWidth+1||c.scrollWidth>c.clientWidth+1,centered:Math.abs(h.left+h.width/2-s.left-s.width/2)<1&&Math.abs(h.top+h.height/2-s.top-s.height/2)<1};})()`);
    await shot('score-details-'+width+'.png');
    assert(!layout.overflow&&layout.left>=15&&layout.right<=width-15&&layout.bottom<=800,'score dialog fits viewport: '+JSON.stringify({width,...layout}));
    assert(layout.centered,'score close icon is centered');
  }
  await evaluate(`const e=document.querySelector('[data-score-dimension]');e.value='info_recall';e.dispatchEvent(new Event('change'));const s=document.querySelector('[data-score-filter]');s.value='0';s.dispatchEvent(new Event('change'));`);
  assert.equal(await evaluate('document.querySelectorAll("#score-dialog .score-criterion").length'),1);
  assert(await evaluate('document.querySelector("#score-dialog .score-items").textContent.includes("Reason B")'));
  assert(await evaluate('document.querySelector("#score-dialog .score-items").textContent.includes("0 分项按评分规则不提供原文证据")'));
  assert.equal(await evaluate('window.scoreDetailRequests'),1,'filtering never refetches or grades');
  await click('[data-close="score-dialog"]');
  await until(()=>evaluate('!document.querySelector("#score-dialog").open'),'score dialog closed');
  await evaluate('window.scoreDetailMode="fail"');await click('[data-score-task="2"]');
  await until(()=>evaluate('!!document.querySelector("[data-reload-score]")'),'read error with retry');
  await evaluate('window.scoreDetailMode="normal"');await click('[data-reload-score]');
  await until(()=>evaluate('document.querySelectorAll("#score-dialog .score-criterion").length===3'),'retry loads saved details');
  await click('[data-close="score-dialog"]');
  await until(()=>evaluate('!document.querySelector("#score-dialog").open'),'retry dialog closed');
  await evaluate('window.scoreDetailMode="delayed"');await click('[data-score-task="2"]');
  await until(()=>evaluate('typeof window.releaseScoreDetail==="function"'),'delayed load started');
  await click('[data-close="score-dialog"]');
  await until(()=>evaluate('!document.querySelector("#score-dialog").open'),'close while loading');
  await evaluate('window.releaseScoreDetail()');await pause(100);
  assert.equal(await evaluate('document.querySelector("#score-dialog-content").textContent'),'','late responses never repopulate a closed dialog');
  await evaluate('window.fetch=window.scoreDetailOriginalFetch;document.dispatchEvent(new Event("visibilitychange"))');
  await until(()=>evaluate('!document.querySelector("[data-score-task=\\"2\\"]")'),'restore fixture scoring');
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
    await evaluate(`document.querySelector('.task-score .report-score').textContent='100.0 / 100';document.querySelector('.task-score').dataset.scoringStatus='official_protocol_scored'`);
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
    assert(await evaluate(`(()=>{
      const picker=document.querySelector('#launch-task-picker'),chips=document.querySelector('#selected-tasks'),dialog=document.querySelector('#launch-dialog');
      return picker.scrollWidth<=picker.clientWidth+1&&chips.scrollWidth<=chips.clientWidth+1&&dialog.scrollWidth<=dialog.clientWidth+1;
    })()`),'task picker fits the launch dialog at '+width+'px');
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
  assert.equal(await evaluate('document.querySelector("#launch-form [name=drb2]").type'),'hidden');
  const launchSettings=await api('settings');
  assert.deepEqual(launchSettings.context_modes,['bcg','bcg_no_retrieval','default']);
  assert.equal(await evaluate('document.querySelector("#launch-mode").value'),'bcg','existing launch defaults to BCG');
  assert.equal(await evaluate('document.querySelector("#launch-mode option[value=default]").disabled'),false);
  await evaluate('document.querySelector("#launch-mode").value="default";document.querySelector("#launch-mode").dispatchEvent(new Event("change",{bubbles:true}))');
  assert(await evaluate('document.querySelector("#launch-task-state").disabled'));
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("不建图、不注入 Graph")'));
  await click('[data-close=launch-dialog]');await click('#launch-button');
  assert.equal(await evaluate('document.querySelector("#launch-mode").value'),'default','reopening preserves the selected context mode');
  await shot('launch-default-mode.png');
  assert.equal(await evaluate('document.querySelector("#launch-mode option[value=bcg_no_retrieval]").disabled'),false);
  await evaluate('document.querySelector("#launch-mode").value="bcg_no_retrieval";document.querySelector("#launch-mode").dispatchEvent(new Event("change",{bubbles:true}))');
  assert(await evaluate('!document.querySelector("#launch-task-state").disabled'));
  await click('#launch-task-state');
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("全部节点和边")'));
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("不按 6,600 字符裁剪")'));
  for(const width of [1440,430,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'full-graph mode dialog does not overflow');
    assert(await evaluate('document.querySelector("#launch-dialog").scrollWidth<=document.querySelector("#launch-dialog").clientWidth+1'),'full-graph mode option fits dialog');
    await shot('launch-no-retrieval-'+width+'.png');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate('document.querySelector("#launch-mode").value="bcg";document.querySelector("#launch-mode").dispatchEvent(new Event("change",{bubbles:true}))');
  assert(await evaluate('document.querySelector("#launch-context-info").textContent.includes("Graph 6,600 字符")'));
  assert.equal(await evaluate('document.querySelector("#launch-form [name=drb2]").value'),launchSettings.default_tasks.drb2.join(','),'launch form uses the server default task list');
  assert.equal(launchSettings.default_tasks.drb2.length,32);
  assert.equal(await evaluate('document.querySelector("#launch-max-tasks").textContent'),'32');
  const taskIds=()=>evaluate('document.querySelector("#launch-form [name=drb2]").value');
  assert.equal(await evaluate('document.querySelectorAll(".task-chip").length'),32);
  const markedIds=()=>evaluate('[...document.querySelectorAll(".task-chip.is-marked")].map(e=>e.dataset.removeTask)');
  const selectionGeometry=()=>evaluate(`(()=>{
    const b=document.querySelector('#selected-tasks').getBoundingClientRect();
    return {box:{left:b.left,top:b.top,right:b.right,bottom:b.bottom},chips:[...document.querySelectorAll('.task-chip')].map(e=>{
      const r=e.getBoundingClientRect();return {id:e.dataset.removeTask,left:r.left,top:r.top,right:r.right,bottom:r.bottom};})};
  })()`);
  const dragChips=async(start,end,release=true)=>{
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:start.x,y:start.y});
    await command('Input.dispatchMouseEvent',{type:'mousePressed',x:start.x,y:start.y,button:'left',buttons:1,clickCount:1});
    for(let i=1;i<=6;i++)await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:start.x+(end.x-start.x)*i/6,y:start.y+(end.y-start.y)*i/6,button:'left',buttons:1});
    if(release)await command('Input.dispatchMouseEvent',{type:'mouseReleased',x:end.x,y:end.y,button:'left',clickCount:1});
  };
  let geometry=await selectionGeometry();
  const firstThree=geometry.chips.slice(0,3).map(c=>c.id),first=geometry.chips[0],third=geometry.chips[2];
  await dragChips({x:first.left+4,y:first.top+4},{x:third.right-2,y:third.bottom-2});
  assert.deepEqual(await markedIds(),firstThree,'drag may start on a chip and selects several IDs');
  assert.equal(await taskIds(),launchSettings.default_tasks.drb2.join(','),'releasing a drag does not single-delete or change launch selection');
  await shot('task-picker-marquee-desktop.png');
  await click('#task-remove-marked');
  assert.deepEqual((await taskIds()).split(','),launchSettings.default_tasks.drb2.filter(id=>!firstThree.includes(id)),'batch removal preserves the order of all remaining IDs');
  assert.deepEqual(await markedIds(),[]);
  await click('#task-restore');
  await dragChips({x:third.right-2,y:third.bottom-2},{x:first.left+4,y:first.top+4});
  assert.deepEqual(await markedIds(),firstThree,'reverse drag selects the same IDs');
  await click('#task-unmark');assert.deepEqual(await markedIds(),[]);
  assert.equal(await taskIds(),launchSettings.default_tasks.drb2.join(','),'cancel selection does not delete IDs');
  await dragChips({x:first.left+4,y:first.top+4},{x:third.right-2,y:third.bottom-2});
  await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  assert.deepEqual(await markedIds(),[]);
  assert(await evaluate('document.querySelector("#launch-dialog").open'),'Escape cancels marking before closing dialog');
  await dragChips({x:first.left+4,y:first.top+4},{x:third.right-2,y:third.bottom-2});
  const fourth=geometry.chips[3],cx=(fourth.left+fourth.right)/2,cy=(fourth.top+fourth.bottom)/2;
  await command('Input.dispatchMouseEvent',{type:'mousePressed',x:cx,y:cy,button:'left',buttons:1,clickCount:1,modifiers:2});
  await command('Input.dispatchMouseEvent',{type:'mouseReleased',x:cx,y:cy,button:'left',clickCount:1,modifiers:2});
  assert.deepEqual(await markedIds(),geometry.chips.slice(0,4).map(c=>c.id),'Ctrl-click adds another ID to the removal selection');
  await evaluate('document.querySelector("#selected-tasks").focus()');
  await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Delete',code:'Delete',windowsVirtualKeyCode:46});
  await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Delete',code:'Delete',windowsVirtualKeyCode:46});
  assert.equal((await taskIds()).split(',').length,28,'Delete removes only the marked IDs');
  await click('#task-restore');
  await command('Emulation.setDeviceMetricsOverride',{width:360,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate('document.querySelector("#selected-tasks").scrollTop=0;document.querySelector("#selected-tasks").scrollIntoView({block:"center"})');
  geometry=await selectionGeometry();
  const bottom={x:geometry.box.right-4,y:geometry.box.bottom-4};
  await dragChips({x:geometry.box.left+3,y:geometry.box.top+3},bottom,false);
  await until(()=>evaluate('(()=>{const e=document.querySelector("#selected-tasks");return e.scrollTop+e.clientHeight>=e.scrollHeight-1;})()'),'drag near the edge scrolls the chip list');
  await command('Input.dispatchMouseEvent',{type:'mouseReleased',...bottom,button:'left',clickCount:1});
  assert.equal((await markedIds()).length,32,'cross-row scrolling allows selecting all IDs');
  assert.equal(await taskIds(),launchSettings.default_tasks.drb2.join(','));
  assert(await evaluate('document.querySelector("#launch-task-picker").scrollWidth<=document.querySelector("#launch-task-picker").clientWidth+1'),'batch controls fit mobile width');
  await shot('task-picker-marquee-mobile.png');
  await click('#task-remove-marked');assert.equal(await taskIds(),'');
  assert.equal((await api('runs')).runs.length,1,'selection never starts a test');
  await click('#task-restore');
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await click('[data-remove-task="2"]');
  assert.equal(await evaluate('document.querySelectorAll(".task-chip").length'),31);
  assert(!(await taskIds()).split(',').includes('2'));
  await click('#task-restore');
  assert.equal(await taskIds(),launchSettings.default_tasks.drb2.join(','));
  await click('#task-clear');
  assert.equal(await taskIds(),'');
  await click('#launch-form [type=submit]');
  assert(await evaluate('document.querySelector("#task-picker-error").textContent.includes("至少选择")'));
  assert.equal((await api('runs')).runs.length,1,'empty selection does not queue a run');
  await fill('#task-id-input','drb2:2,5,5');await click('#task-add');
  assert.equal(await taskIds(),'2,5','batch add deduplicates');
  await fill('#task-id-input','1-33');await click('#task-add');
  assert.equal(await taskIds(),'2,5','over-limit paste leaves selection unchanged');
  assert(await evaluate('document.querySelector("#task-picker-error").textContent.includes("最多选择")'));
  await fill('#task-id-input','');
  await click('.task-library summary');
  await click('[data-task-id="5"]');assert.equal(await taskIds(),'2');
  await fill('#task-catalog-search','30');await click('#task-select-filtered');
  assert.equal(await taskIds(),'2,30');
  await click('[data-close=launch-dialog]');await click('#launch-button');
  assert.equal(await taskIds(),'2,30','reopening keeps the selection');
  await shot('task-picker.png');
  await click('#task-clear');await fill('#task-id-input','2');
  await evaluate('document.querySelector("#task-id-input").dispatchEvent(new KeyboardEvent("keydown",{key:"Enter",bubbles:true,cancelable:true}))');
  assert.equal(await taskIds(),'2','Enter adds IDs rather than submitting the launch form');
  assert.equal((await api('runs')).runs.length,1);
  await evaluate('document.querySelector("#launch-mode").value="default";document.querySelector("#launch-mode").dispatchEvent(new Event("change",{bubbles:true}))');
  await shot('launch-dialog.png');await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0002"'),'queued batch');
  assert.equal((await api('runs/2')).status,'queued');
  assert.equal((await api('runs/2')).title,suggestedTitle,'timestamped name is persisted');
  const defaultRun=await api('runs/2');
  assert.equal(defaultRun.mode,'default','the real launch form sends and persists Default');
  assert.equal(defaultRun.graph,null);
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("模式：Default")'));
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("Graph：不启用")'));
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
  await click('#launch-button');
  await evaluate('document.querySelector("#launch-mode").value="bcg_no_retrieval";document.querySelector("#launch-mode").dispatchEvent(new Event("change",{bubbles:true}))');
  await click('#launch-form [type=submit]');
  await until(()=>evaluate('document.querySelector(".run-heading .eyebrow")?.textContent==="RUN-0003"'),'queued full-graph batch');
  const fullGraphRun=await api('runs/3');
  assert.equal(fullGraphRun.mode,'bcg_no_retrieval');
  assert.equal(fullGraphRun.task_state,true);
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("任务状态")'));
  assert.equal(fullGraphRun.graph,'gpt-6-luna');
  assert.equal(fullGraphRun.judge.enabled,true);
  assert(await evaluate('document.querySelector(".run-heading").textContent.includes("模式：BCG w/o Retrieval")'));
  assert(await evaluate('document.querySelector(".run-item.selected").textContent.includes("BCG w/o Retrieval")'));
  for(const width of [1440,430,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'long mode name does not widen run list');
    await shot('run-no-retrieval-'+width+'.png');
  }
  await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await api('runs/3/cancel',aliceToken,'POST',{});
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
  // Annotation statistics are a read-only export, never an AI review trigger.
  await navigate('/?run=1');await until(()=>evaluate('!!document.querySelector("a[href*=annotation-stats]")'),'statistics entry');
  await click('a[href*=annotation-stats]');
  await until(()=>evaluate('document.querySelector("#stats-report")?.contentDocument?.body?.textContent.includes("暂无已完成标注")'),'draft-only statistics empty state');
  const annotationBeforeStats=await api('runs/1/tasks/drb2/2/annotation',bob.token);
  await api('runs/1/tasks/drb2/2/annotation',bob.token,'PUT',{
    ...annotationBeforeStats,annotation:{...annotationBeforeStats.annotation,status:'reviewed',result:'部分正确 / 部分达标',
      dimensions:{...annotationBeforeStats.annotation.dimensions,A1:{opportunity:'有',utilization:'未利用',effect:'负向',eventIds:['stats-e1']}},
      events:[{id:'stats-e1',label:'离线统计示例',strength:'明确可见'}],conclusion:{focus:['Agent 使用'],evidence:'明确可见'}},
  });
  await click('#stats-refresh');
  await until(()=>evaluate('document.querySelector("#stats-report")?.contentDocument?.querySelectorAll(".cohort").length===1'),'completed review chart');
  assert(await evaluate('document.querySelector("#stats-report").contentDocument.querySelectorAll("svg").length===7'));
  for(const width of [1440,1024,390,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    assert(await evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'statistics outer page fits '+width);
    assert(await evaluate('(()=>{const d=document.querySelector("#stats-report").contentDocument;return d.documentElement.scrollWidth<=d.documentElement.clientWidth+1;})()'),'statistics report fits '+width);
    assert(await evaluate('(()=>{const b=document.querySelector("#stats-download").getBoundingClientRect();return b.left>=0&&b.right<=innerWidth;})()'),'statistics export visible '+width);
    await shot('annotation-stats-'+width+'.png');
  }
  const beforeExport=await api('runs/1/tasks/drb2/2/annotation',bob.token);
  await api('runs/1/tasks/drb2/2/annotation',bob.token,'PUT',{...beforeExport,annotation:beforeExport.annotation});
  await click('#stats-download');
  await until(()=>evaluate('document.querySelector("#stats-feedback").textContent.includes("标注已有更新")'),'stale export cannot mismatch chart');
  await click('#stats-refresh');
  await until(()=>evaluate('!document.querySelector("#stats-refresh").disabled'),'statistics refresh complete');
  await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:dir});
  await click('#stats-download');
  await until(async()=>{try{return (await readFile(join(dir,'RUN-0001-annotation-stats.zip'))).subarray(0,2).toString()==='PK';}catch{return false;}},'statistics ZIP download');
  await evaluate('localStorage.removeItem("bcg-studio-personal-token")');
  await navigate('/annotation-stats.html?run=1');
  await until(()=>evaluate('document.querySelector("#stats-feedback").textContent.includes("填写昵称")'),'statistics requires identity');
  await evaluate(`localStorage.setItem('bcg-studio-personal-token',${JSON.stringify(bob.token)})`);
  await navigate('/index.html');await until(()=>evaluate('!!document.querySelector(".welcome")'),'local workspace remains separate');
  assert.equal(await evaluate('document.querySelectorAll(".case-item").length'),0);
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log('Shared browser smoke passed: refresh animation/success/error/retry/reduced motion, background-request coalescing, launch/cancel without admin code, automatic case load, separate reviewers, persistence, conflict recovery, local mode.');
  console.log('Screenshots: '+dir);
}finally{ws?.close();chrome.kill('SIGTERM');}

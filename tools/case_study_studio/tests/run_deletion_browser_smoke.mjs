// Run only against shared_smoke_server.py with a fresh synthetic directory.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const origin=process.env.STUDIO_URL||'http://127.0.0.1:8877';
const output=await mkdtemp(join(tmpdir(),'bcg-delete-browser-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu',
  '--disable-background-networking','--no-first-run','--remote-debugging-port=0',
  '--user-data-dir='+output,'about:blank'],{stdio:'ignore'});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(check,label){for(let i=0;i<100;i++){try{if(await check())return;}catch{}await pause(100);}throw new Error(label);}
let ws,sequence=0;
const pending=new Map(),errors=[];
try{
  await until(async()=>!!await readFile(join(output,'DevToolsActivePort')),'Chrome starts');
  const port=(await readFile(join(output,'DevToolsActivePort'),'utf8')).split('\n')[0];
  const target=await(await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(origin)}`,{method:'PUT'})).json();
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
  const command=(method,params={})=>new Promise((resolve,reject)=>{
    const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));
  });
  ws.addEventListener('message',event=>{
    const data=JSON.parse(event.data);
    if(data.id){const item=pending.get(data.id);pending.delete(data.id);data.error?item.reject(data.error):item.resolve(data.result);}
    if(data.method==='Runtime.exceptionThrown')errors.push(data.params.exceptionDetails);
  });
  await command('Runtime.enable');await command('Page.enable');
  const evaluate=async expression=>{
    const result=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const fill=(selector,value)=>evaluate(`{const field=document.querySelector(${JSON.stringify(selector)});field.value=${JSON.stringify(value)};field.dispatchEvent(new Event('input',{bubbles:true}));}`);
  await until(()=>evaluate('document.querySelector("#identity-dialog")?.open'),'identity dialog');
  await fill('#identity-form [name=name]','delete-fixture-'+Date.now());
  await click('#identity-form [type=submit]');
  await until(()=>evaluate('!!document.querySelector("#delete-run")'),'run loaded');
  const token=await evaluate('localStorage.getItem("bcg-studio-personal-token")');
  const api=async(path,method='GET',body)=>{
    const response=await fetch(origin+'/api/'+path,{method,headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    const data=await response.json();assert(response.ok,JSON.stringify(data));return data;
  };
  assert.equal((await api('runs/1')).agent,'offline-fixture','refuse to delete real results');
  assert.equal((await api('runs')).runs.length,1,'requires fresh QA fixture');
  await click('#delete-run');
  assert(await evaluate('document.querySelector("#delete-dialog").open'));
  assert(await evaluate('document.querySelector("#delete-submit").disabled'));
  await fill('#delete-confirmation','RUN-0002');
  assert(await evaluate('document.querySelector("#delete-submit").disabled'));
  await fill('#delete-confirmation','RUN-0001');
  assert.equal(await evaluate('document.querySelector("#delete-submit").disabled'),false);
  for(const width of [1440,430,320]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false});
    await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    const layout=await evaluate(`(()=>{const dialog=document.querySelector('#delete-dialog'),r=dialog.getBoundingClientRect();
      const close=dialog.querySelector('.icon-button').getBoundingClientRect(),icon=dialog.querySelector('svg').getBoundingClientRect();
      return {left:r.left,right:r.right,overflow:dialog.scrollWidth>dialog.clientWidth+1,pageOverflow:document.documentElement.scrollWidth>innerWidth+1,
        dx:Math.abs(close.left+close.width/2-icon.left-icon.width/2),dy:Math.abs(close.top+close.height/2-icon.top-icon.height/2)};})()`);
    assert(layout.left>=15&&layout.right<=width-15,JSON.stringify({width,...layout}));
    assert(!layout.overflow&&!layout.pageOverflow,'dialog and page fit '+width);
    assert(layout.dx<1&&layout.dy<1,'close icon centered');
    const shot=await command('Page.captureScreenshot',{format:'png'});
    await writeFile(join(output,`delete-${width}.png`),Buffer.from(shot.data,'base64'));
  }
  await click('#delete-dialog .delete-actions [data-close]');
  assert.equal((await api('runs')).runs.length,1,'Cancel is non-destructive');
  const queued=await api('runs','POST',{title:'Delete-only offline fixture',profile:'qwen35-9b',workers:1,tasks:{drb2:['2']}});
  await evaluate(`history.replaceState(null,'','?run=${queued.id}')`);
  await command('Page.navigate',{url:origin+'/?run='+queued.id});
  await until(()=>evaluate(`document.querySelector('#delete-run')?.dataset.runId==='${queued.id}'`),'queued run selected');
  assert(await evaluate('document.querySelector("#delete-run").disabled'));
  assert(await evaluate('document.querySelector("#delete-run").title.includes("先停止")'));
  await api(`runs/${queued.id}/cancel`,'POST',{});
  await evaluate('document.dispatchEvent(new Event("visibilitychange"))');
  await until(()=>evaluate('!document.querySelector("#delete-run").disabled'),'stopped run deletable');
  await click('#delete-run');await fill('#delete-confirmation',queued.number);await click('#delete-submit');
  await until(()=>evaluate('!document.querySelector("#delete-dialog").open && document.querySelector("#delete-run")?.dataset.runId==="1"'),'deleted selection falls back');
  assert.equal(await evaluate('location.search'),'?run=1');
  assert.deepEqual((await api('runs')).runs.map(run=>run.id),[1]);
  await click('#delete-run');await fill('#delete-confirmation','RUN-0001');await click('#delete-submit');
  await until(()=>evaluate('!document.querySelector("#delete-dialog").open && !document.querySelector("#delete-run")'),'empty state after last deletion');
  await pause(3500);
  assert.deepEqual((await api('runs')).runs,[],'polling does not resurrect deleted run');
  assert.equal(await evaluate('location.search'),'');
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({ok:true,screenshots:output}));
}finally{ws?.close();chrome.kill();}

// Run against Studio's static /index.html. Uses a private Chrome profile; no
// shared identity, annotations, model calls or experiment launches.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {gunzipSync} from 'node:zlib';

const root=await mkdtemp(join(tmpdir(),'bcg-atlas-qa-'));
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--disable-background-networking',
  '--disable-sync','--disable-component-update','--no-first-run','--remote-debugging-pipe','--user-data-dir='+join(root,'profile'),'about:blank'],{stdio:['ignore','ignore','ignore','pipe','pipe']});
const wait=ms=>new Promise(r=>setTimeout(r,ms)), pending=new Map(), errors=[];
const stopped=new Promise(r=>chrome.once('exit',r));let bytes='',seq=0,session;
chrome.stdio[4].setEncoding('utf8').on('data',chunk=>{
  bytes+=chunk;
  for(let split;(split=bytes.indexOf('\0'))!==-1;){
    const v=JSON.parse(bytes.slice(0,split));bytes=bytes.slice(split+1);
    if(v.id&&pending.has(v.id)){const p=pending.get(v.id);pending.delete(v.id);clearTimeout(p.timer);v.error?p.reject(Error(v.error.message)):p.resolve(v.result);}
    if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails.exception?.description||v.params.exceptionDetails.text);
  }
});
function cdp(method,params={},sid=session){return new Promise((resolve,reject)=>{
  const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error('Chrome timeout: '+method));},15000);
  pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,sessionId:sid})+'\0');
});}
const ev=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;};
const click=s=>ev(`document.querySelector(${JSON.stringify(s)}).click()`);
async function until(s){for(let i=0;i<500;i++){if(await ev(s))return;await wait(80);}throw Error('Did not reach '+s+'; '+await ev("document.querySelector('#toast')?.textContent"));}
async function files(paths){const {root}=await cdp('DOM.getDocument');const {nodeId}=await cdp('DOM.querySelector',{nodeId:root.nodeId,selector:'#file-input'});await cdp('DOM.setFileInputFiles',{nodeId,files:paths});}
const paint=()=>ev('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))).then(()=>true)');
const bitmap=()=>ev("document.querySelector('.atlas-stage canvas').toDataURL()");
const isFull='Boolean(document.fullscreenElement || document.querySelector(".atlas-modal")?.open)';
try{
  const {targetId}=await cdp('Target.createTarget',{url:'about:blank'});
  ({sessionId:session}=await cdp('Target.attachToTarget',{targetId,flatten:true}));
  await cdp('Page.enable');await cdp('Runtime.enable');
  await cdp('Page.navigate',{url:(process.env.STUDIO_URL||'http://127.0.0.1:8876')+'/index.html'});
  await until('!!document.querySelector(".welcome")');
  const graph={nodes:[0,2,4,114,116,999].map((id,i)=>({id,belief:'Belief '+id,source:{turn_id:[1,1,2,4,4,6][i]},
    role:['user','user','assistant','tool','tool','assistant'][i],confidence:i/10,node_type:i===5?'decision':'belief',task_role:id===116?'evidence':id===114?'progress':'other'})),relations:[{from_id:114,to_id:116,type:'supplements'}],
    task_state:{nodes:[{id:'T2',text:'两个区域的研究',status:'in_progress',status_source_id:114},{id:'T4',text:'独立任务',status:'open',status_provenance:'not_reported'}],
      relations:[{from_id:'T2',to_id:'B2',type:'described_by'},{from_id:'B114',to_id:'T2',type:'addresses'},{from_id:'T2',to_id:'B114',type:'updated_by'},
        {from_id:'B114',to_id:'T4',type:'addresses'},{from_id:'T2',to_id:'B116',type:'related_via_graph',relation_ids:[9]},{from_id:'T4',to_id:'T2',type:'part_of'}]}};
  const f=join(root,'state.json'),h=join(root,'graph_snapshots.jsonl');
  const question='比较两个区域的研究 <原始问题>\n给出证据与结论。';
  await writeFile(f,JSON.stringify({task_id:'atlas-qa',acknowledged:[{role:'user',content:question}],snapshot:graph}));
  await writeFile(h,JSON.stringify({call:1,snapshot:{nodes:[{id:0,belief:'first'}],relations:[]}})+'\n');
  await files([f,h]);await until("document.querySelector('.breadcrumb strong')?.textContent==='atlas-qa'");
  await click('[data-tab=graph]');await until("document.querySelector('.graph-atlas')?.dataset.nodes==='6'");
  assert.equal(await ev("document.querySelector('.graph-atlas').dataset.columns"),'4');
  assert.equal(await ev("document.querySelector('.graph-atlas').dataset.layout"),'trajectory');
  await click('[data-map=screen]');await until(isFull);await click('[data-map=fit]');await paint();
  const fills=await ev(`(()=>{const c=document.querySelector('.atlas-stage canvas'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
    const seen=new Set();for(let i=0;i<p.length;i+=4)if(p[i+3]===255)seen.add(p[i]*65536+p[i+1]*256+p[i+2]);
    const navy=parseInt(getComputedStyle(c).getPropertyValue('--navy').trim().slice(1),16);
    return seen.has(navy) && [0xe0ecfa,0xfbe3ee,0xe1f0e4].every(n=>!seen.has(n))})()`);
  assert(fills,'BCG uses the Studio navy dots, not pastel source-role cards');
  assert(await ev("document.querySelector('.graph-atlas').getBoundingClientRect().width>=innerWidth-2"));
  const {x,y}=await ev("(()=>{const b=document.querySelector('.atlas-stage canvas').getBoundingClientRect();return{x:b.x+b.width*.3,y:b.y+b.height*.3}})()");
  const before=await bitmap();
  for(const [type,dx,dy] of [['mousePressed',0,0],['mouseMoved',83,55],['mouseReleased',83,55]])
    await cdp('Input.dispatchMouseEvent',{type,x:x+dx,y:y+dy,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1});
  await paint();assert.notEqual(await bitmap(),before,'drag must move the graph');
  await click('[data-map=fit]');await paint();assert.equal(await bitmap(),before,'fit restores the whole graph');
  await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',x,y,deltaX:0,deltaY:-150});
  await until("parseInt(document.querySelector('.atlas-scale').value)>100");
  await ev("(()=>{const q=document.querySelector('.atlas-query');q.value='D#999';q.dispatchEvent(new Event('input'));q.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}));})()");
  await until("document.querySelector('.atlas-detail')?.textContent.includes('traj[6]') && document.querySelector('.atlas-detail')?.textContent.includes('D#999 决策')");
  await paint();
  const blank=await ev("(()=>{const b=document.querySelector('.atlas-stage canvas').getBoundingClientRect();return{x:b.left+7,y:b.top+7}})()");
  for(const type of ['mousePressed','mouseReleased'])await cdp('Input.dispatchMouseEvent',{type,...blank,button:'left',clickCount:1});
  await until("!document.querySelector('.atlas-detail')");
  await click('button[data-layer=tsg]');await until("document.querySelector('.graph-atlas').dataset.nodes==='5'");
  assert.equal(await ev("document.querySelector('.graph-atlas').dataset.layout"),'task-tree');
  assert.equal(await ev("document.querySelectorAll('[data-set-id]').length"),2);
  assert(await ev("document.querySelector('.atlas-count').textContent.includes('1 个跨集合共享')"));
  assert(await ev("document.querySelector('.atlas-timeline-nav').hidden"));
  assert(await ev(isFull));assert.equal(await ev("document.querySelector('.graph-atlas').dataset.relations"),'6');
  await paint();
  assert(await ev("!!document.querySelector('[data-wire-kind=root][data-wire-to=T2][marker-end]')"));
  assert(await ev("!!document.querySelector('[data-task-relation=part_of][data-wire-from=T2][data-wire-to=T4][data-record-from=T4][data-record-to=T2][marker-end]')"));
  assert(await ev("document.querySelector('[data-task-root]').offsetTop < document.querySelector('[data-set-id=T2]').offsetTop"));
  await click('[data-task-root]');assert.equal(await ev("document.querySelector('.atlas-detail p').textContent"),question);
  await click('[data-root-close]');await paint();assert.equal(await ev("!!document.querySelector('.atlas-detail')"),false);
  assert(await ev("document.querySelector('[data-set-id=T4]').offsetTop>document.querySelector('[data-set-id=T2]').offsetTop"));
  assert.equal(await ev("document.querySelectorAll('[data-shared-reference]').length"),0);
  await click('[data-shared-wires]');await paint();
  assert.equal(await ev("document.querySelectorAll('[data-shared-reference]:not([marker-end])').length"),2);
  assert.equal(await ev("document.querySelectorAll('[data-task-relation]').length"),1);
  await click('[data-hub-open="0"]');await until("document.querySelector('.atlas-member-head')?.textContent.includes('仅由以上集合共同引用（1）')");
  await click('[data-shared-wires]');
  await click('button[data-layer=bcg]');await click('button[data-layer=tsg]');await paint();
  assert.equal(await ev("document.querySelectorAll('[data-shared-reference]').length"),0);
  const transform=()=>ev("document.querySelector('.atlas-set-world').style.transform");
  await click('[data-map=fit]');await paint();const beforeSets=await transform();
  const emptySpot=await ev("(()=>{const b=document.querySelector('.atlas-set-board').getBoundingClientRect();return{x:b.x+4,y:b.y+4}})()");
  for(const [type,dx] of [['mousePressed',0],['mouseMoved',55],['mouseReleased',55]])
    await cdp('Input.dispatchMouseEvent',{type,x:emptySpot.x+dx,y:emptySpot.y+4,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1});
  await paint();assert.notEqual(await transform(),beforeSets,'task collection board pans');
  await click('[data-map=fit]');await paint();assert.equal(await transform(),beforeSets);
  const beforeScale=await ev("parseInt(document.querySelector('.atlas-scale').value)");
  await cdp('Input.dispatchMouseEvent',{type:'mouseWheel',...emptySpot,deltaX:0,deltaY:-160});
  await until("parseInt(document.querySelector('.atlas-scale').value)>"+beforeScale);await click('[data-map=fit]');
  await ev("let q=document.querySelector('.atlas-query');q.value='T2';q.dispatchEvent(new Event('input',{bubbles:true}));q.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))");
  await until("document.querySelector('.atlas-detail')?.textContent.includes('进行中')");await paint();
  assert.equal(await ev("document.querySelectorAll('.atlas-member-open').length"),3);
  assert(await ev("document.querySelector('.atlas-owner-list').textContent.includes('子任务：T4')"));
  const sharedButton=await ev("(()=>{const b=document.querySelector('[data-set-id=T2] [data-bucket=shared]').getBoundingClientRect();return{x:b.x+b.width/2,y:b.y+b.height/2}})()");
  for(const type of ['mousePressed','mouseReleased'])await cdp('Input.dispatchMouseEvent',{type,...sharedButton,button:'left',clickCount:1});
  await until("document.querySelector('.atlas-member-head')?.textContent.includes('跨集合共享（1）')");
  await click('[data-set-member=B114]');await until("document.querySelector('.atlas-owner-list')?.textContent.includes('所属集合（2）')");
  assert.equal(await ev("document.querySelectorAll('.atlas-set-card.contains-selection').length"),2);
  await click('[data-member-back]');assert(await ev("document.querySelector('.atlas-member-head').textContent.includes('跨集合共享（1）')"));
  await click('[data-overlap="0"]');await until("document.querySelector('.atlas-member-head')?.textContent.includes('仅由以上集合共同引用（1）')");
  assert.equal(await ev("document.querySelectorAll('.atlas-set-card.contains-selection').length"),2);
  await click('[data-set-member=B114]');await click('[data-member-back]');
  assert(await ev("document.querySelector('.atlas-member-head').textContent.includes('仅由以上集合共同引用（1）')"));
  await click('.atlas-owner-list [data-set-open=T2]');
  await click('.atlas-links summary');await until("!!document.querySelector('[data-map-target=B114]')");
  await click('[data-map-target=B114]');assert(await ev("document.querySelector('.atlas-detail').textContent.includes('Belief 114')"));
  await click('[data-map=screen]');await until('!'+isFull);
  await click('[data-graph-view=turns]');await until("document.querySelector('.atlas-empty')?.textContent.includes('未记录 TSG')");
  assert.equal(await ev("document.querySelector('[data-task-root]')"),null);
  await click('[data-graph-view=full]');await click('button[data-layer=bcg]');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
  await cdp('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:2});
  await ev("Object.defineProperty(document,'fullscreenEnabled',{value:false,configurable:true})");
  await click('[data-map=screen]');await until('document.querySelector(".atlas-modal").open');
  assert(await ev("document.documentElement.scrollWidth<=innerWidth+1 && document.querySelector('.atlas-stage').getBoundingClientRect().height>300"));
  await click('[data-map=fit]');await paint();
  const mid=await ev("(()=>{const b=document.querySelector('.atlas-stage canvas').getBoundingClientRect();return{x:b.x+b.width/2,y:b.y+b.height/2}})()");
  for(const [type,distance] of [['touchStart',15],['touchMove',40],['touchEnd',0]]){
    const touchPoints=distance?[-1,1].map((sign,id)=>({x:mid.x+sign*distance,y:mid.y,id})):[];
    await cdp('Input.dispatchTouchEvent',{type,touchPoints});
  }
  await until("parseInt(document.querySelector('.atlas-scale').value)>100");
  for(const type of ['keyDown','keyUp'])await cdp('Input.dispatchKeyEvent',{type,key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await until('!'+isFull);
  await cdp('Emulation.setTouchEmulationEnabled',{enabled:false});
  await cdp('Emulation.clearDeviceMetricsOverride');
  await click('[data-tab=report]');await ev("location.hash='tsg'");
  await until("document.querySelector('.graph-atlas')?.dataset.layer==='tsg'");
  await ev("location.hash='bcg'");await until("document.querySelector('.graph-atlas')?.dataset.layer==='bcg'");
  if(process.argv[2]){
    const file=resolve(process.argv[2]), bundle=JSON.parse(gunzipSync(await readFile(file))),row=bundle.cases[0];
    const source=row.files.find(f=>f.path.endsWith('bcg/state.json')),s=JSON.parse(source.text).snapshot;
    await files([file]);await until("document.querySelector('.breadcrumb strong')?.textContent!== 'atlas-qa' && !!document.querySelector('.case-heading')");
    await click('[data-tab=graph]');await until(`document.querySelector('.graph-atlas')?.dataset.nodes==='${s.nodes.length}'`);
    assert.equal(await ev("document.querySelector('.graph-atlas').dataset.relations"),String(s.relations.length));
    const turns=[...new Set(s.nodes.map(n=>n.source?.turn_id))].sort((a,b)=>a-b);
    assert.equal(await ev("document.querySelector('.graph-atlas').dataset.columns"),String(turns.length));
    await paint();
    assert((await ev("document.querySelector('.atlas-timeline-nav output').textContent")).includes('traj['+turns[0]+']'));
    if(turns.length>40){
      assert.equal(await ev("document.querySelector('.atlas-scale').value"),'100%','large graphs open at reading size');
      await click('[data-map=fit]');await paint();assert(parseInt(await ev("document.querySelector('.atlas-scale').value"))<20);
      await ev("(()=>{const s=document.querySelector('.atlas-timeline-nav input');s.value=s.max;s.dispatchEvent(new Event('input'));})()");
      await paint();assert.equal(await ev("document.querySelector('.atlas-scale').value"),'100%');
      assert((await ev("document.querySelector('.atlas-timeline-nav output').textContent")).includes('traj['+turns.at(-1)+']'));
    }
    await click('button[data-layer=tsg]');const t=s.task_state,ids=new Set([...t.nodes.map(n=>n.id),...t.relations.flatMap(e=>[e.from_id,e.to_id])]);
    await until(`document.querySelector('.graph-atlas')?.dataset.nodes==='${ids.size}'`);
    assert.equal(await ev("document.querySelector('.graph-atlas').dataset.relations"),String(t.relations.length));
    await paint();
    const taskIds=new Set(t.nodes.map(n=>n.id)),edges=t.relations.filter(e=>taskIds.has(e.from_id)&&taskIds.has(e.to_id));
    const rootId='@user-input';
    const arrows=await ev("Array.from(document.querySelectorAll('.atlas-task-wire[marker-end]'),p=>[p.dataset.wireFrom,p.dataset.wireTo,p.dataset.wireKind,p.dataset.recordFrom,p.dataset.recordTo])");
    assert.equal(arrows.length,t.nodes.length);
    assert.deepEqual(arrows.map(a=>a[1]).sort(),[...taskIds].sort());
    const parentOf=new Map(arrows.map(([from,to])=>[to,from]));
    for(const [from,to,kind,savedFrom,savedTo] of arrows){
      if(from===rootId)assert.equal(kind,'root');
      else {assert.equal(savedFrom,to);assert.equal(savedTo,from);assert(edges.some(e=>e.type==='part_of' && e.from_id===to && e.to_id===from));}
      const seen=new Set();let id=to;
      while(id!==rootId){assert(!seen.has(id) && parentOf.has(id));seen.add(id);id=parentOf.get(id);}
    }
    const original=JSON.parse(row.files.find(f=>f.path==='case.json').text).question;
    assert(original);await click('[data-map=root]');assert.equal(await ev("document.querySelector('.atlas-detail p').textContent"),original);
    await click('[data-root-close]');await paint();
    console.log('Rooted tree:',arrows.filter(a=>a[0]===rootId).length,'top-level tasks;',arrows.filter(a=>a[0]!==rootId).length,'subtask branches from',edges.length,'recorded task relations.');
    const memberships=new Map();
    for(const e of t.relations){
      const task=[e.from_id,e.to_id].find(id=>id.startsWith('T')),belief=[e.from_id,e.to_id].find(id=>id.startsWith('B'));
      if(task && belief){if(!memberships.has(belief))memberships.set(belief,new Set());memberships.get(belief).add(task);}
    }
    const shared=[...memberships].filter(([,owners])=>owners.size>1);
    assert.equal(await ev("document.querySelectorAll('[data-set-id]').length"),t.nodes.length);
    assert((await ev("document.querySelector('.atlas-count').textContent")).includes(shared.length+' 个跨集合共享'));
    for(const task of t.nodes){
      const count=[...memberships.values()].filter(owners=>owners.has(task.id)).length;
      assert.equal(await ev(`Number(document.querySelector('[data-set-id=${task.id}] .atlas-set-totals [data-bucket=all] b').textContent)`),count);
    }
    const partitions=new Map();
    for(const [,owners] of shared){const key=[...owners].sort((a,b)=>a.localeCompare(b,'en',{numeric:true})).join('、');partitions.set(key,(partitions.get(key)||0)+1);}
    const signatures=await ev("Array.from(document.querySelectorAll('.atlas-overlaps button,.atlas-overlaps option:not([value=\"\"])'),n=>n.textContent)");
    assert.deepEqual(signatures.sort(),[...partitions.entries()].map(([owners,n])=>'仅 '+owners+'（'+n+'）').sort());
    const spokeCount=Array.from(partitions.keys()).reduce((n,k)=>n+k.split('、').length,0);
    if(spokeCount){
      assert.equal(await ev("document.querySelector('[data-shared-wires]').checked"),false);
      assert.equal(await ev("document.querySelectorAll('.atlas-shared-wire:not([marker-end])').length"),0);
      await click('[data-shared-wires]');await paint();
      assert.equal(await ev("document.querySelectorAll('.atlas-shared-wire:not([marker-end])').length"),spokeCount);
      assert.equal(await ev("document.querySelectorAll('.atlas-task-wire[marker-end]').length"),t.nodes.length);
      await click('[data-shared-wires]');await paint();
    }
    if(process.env.ATLAS_SCREENSHOT){await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
      await click('[data-map=screen]');await until(isFull);await click('[data-map=fit]');await paint();
      const img=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(process.env.ATLAS_SCREENSHOT,Buffer.from(img.data,'base64'));}
    if(await ev("document.querySelector('.atlas-set-world').dataset.density==='compact' && !!document.querySelector('.atlas-set-mini')")){
      await click('.atlas-set-mini');await paint();
      assert.equal(await ev("document.querySelector('.atlas-set-world').dataset.density"),'detail');
      assert(await ev("!!document.querySelector('.atlas-member-head')"));
    }
    if(shared.length){const [id,owners]=shared.reduce((a,b)=>a[1].size>=b[1].size?a:b);
      await ev(`(()=>{const q=document.querySelector('.atlas-query');q.value='${id}';q.dispatchEvent(new Event('input'));q.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}))})()`);
      assert.equal(await ev("document.querySelectorAll('.atlas-owner-list [data-set-open]').length"),owners.size);
    }
    if(shared.length){
      await click('[data-overlap="0"]');
      if(await ev("!!document.querySelector('[data-member-page=\"1\"]')")){
        const first=await ev("document.querySelector('.atlas-member-open').dataset.setMember");
        await click('[data-member-page="1"]');assert.notEqual(await ev("document.querySelector('.atlas-member-open').dataset.setMember"),first);
      }
    }
    console.log('Real BCG:',s.nodes.length,'nodes /',s.relations.length,'edges; TSG:',t.nodes.length,'tasks /',ids.size,'nodes /',t.relations.length,'links');
  }
  assert.deepEqual(errors,[]);console.log('Browser: whole maps, native and modal fullscreen, drag, wheel, fit, inspector, search, snapshot isolation and mobile passed.');
}finally{
  chrome.kill('SIGTERM');await stopped;await rm(root,{recursive:true,force:true,maxRetries:6,retryDelay:140});
}

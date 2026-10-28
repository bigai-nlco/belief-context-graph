import {copyKey, emptyDraft, mergedCopy, slots, validateCopy, validateEntry} from './copy-model.mjs';

const $=selector=>document.querySelector(selector);
const STORAGE='bcg-studio-copy-draft-v1';
const kinds={text:'页面文字',placeholder:'输入框提示',title:'悬停提示','aria-label':'无障碍标签'};
const frame=$('#preview'), catalog=new Map(), history=[];
let published, draft, selected='', mode='pick', saveTimer, ready=false, compatible=true;
const send=(type,extra={})=>frame.contentWindow?.postMessage({channel:'bcg-copy-editor',type,...extra},location.origin);
function notice(message,error=false){$('#editor-notice').textContent=message;$('#editor-notice').classList.toggle('error',error);}
function baseline(entry){return published.entries.find(e=>copyKey(e)===copyKey(entry))?.value??entry.original;}
function currentValue(entry){return draft.entries.find(e=>copyKey(e)===copyKey(entry))?.value??baseline(entry);}
function persist(){
  try{localStorage.setItem(STORAGE,JSON.stringify(draft));$('#draft-status').textContent='草稿已保存到此浏览器';}
  catch{$('#draft-status').textContent='本地保存失败，请导出备份';notice('浏览器未能保存草稿。请使用「导出修改稿」保存文件。',true);}
}
function refreshCount(){
  $('#change-count').textContent=`${draft.entries.length} 项修改`;
  $('#undo-copy').disabled=!history.length||!compatible;
}
function sync(){
  if(ready&&compatible)send('apply',{published,draft});
  refreshCount();renderList();
}
function select(entry){
  if(!commit())return;
  selected=copyKey(entry);catalog.set(selected,entry);
  $('#selected-copy').hidden=false;
  $('#selection-kind').textContent=`${entry.label||'界面'} · ${kinds[entry.kind]}`;
  $('#original-copy').value=entry.original;
  $('#replacement-copy').value=currentValue(entry);
  $('#replacement-copy').disabled=!compatible;
  $('#slot-hint').hidden=!slots(entry.original).length;
  notice('修改后自动预览并保存草稿。同一文案的其他位置会同步更新。');
  renderList();
}
function commit(){
  clearTimeout(saveTimer);
  if(!selected||!compatible)return true;
  const entry=catalog.get(selected), value=$('#replacement-copy').value;
  try{validateEntry({...entry,value});}catch(error){notice(error.message,true);return false;}
  if(value===currentValue(entry))return true;
  history.push(structuredClone(draft));if(history.length>60)history.shift();
  draft.entries=draft.entries.filter(e=>copyKey(e)!==selected);
  if(value!==baseline(entry))draft.entries.push({kind:entry.kind,original:entry.original,value});
  persist();sync();notice('已应用到预览。正式页面保持不变。');return true;
}
function renderList(){
  const query=$('#copy-search').value.toLowerCase(), changed=$('#changed-only').checked;
  const modified=new Set(draft.entries.map(copyKey));
  const list=[...catalog.entries()].filter(([key,e])=>(!changed||modified.has(key))&&
    `${e.original} ${currentValue(e)}`.toLowerCase().includes(query));
  $('#catalog-count').textContent=`${list.length} / ${catalog.size} 条`;
  const root=$('#copy-list');root.replaceChildren();
  for(const [key,entry] of list){
    const button=document.createElement('button');
    button.type='button';button.dataset.copyKey=key;
    button.classList.toggle('changed',modified.has(key));button.setAttribute('aria-current',String(key===selected));
    const label=document.createElement('small');label.textContent=`${entry.label||'界面'} · ${kinds[entry.kind]}${modified.has(key)?' · 已修改':''}`;
    const value=document.createElement('span');value.textContent=currentValue(entry)||'（留空）';
    button.append(label,value);button.addEventListener('click',()=>select(entry));root.append(button);
  }
  if(!list.length){const p=document.createElement('p');p.className='catalog-empty';p.textContent='没有匹配的文案。';root.append(p);}
}
function setMode(next){
  if(!commit())return;
  mode=next;send('mode',{mode});
  $('#pick-mode').setAttribute('aria-pressed',String(mode==='pick'));
  $('#browse-mode').setAttribute('aria-pressed',String(mode==='browse'));
  $('#preview-status').textContent=mode==='pick'?'点击页面文字进行编辑':'可以操作页面、打开教程与切换标签';
}
function exportDraft(){
  if(!commit())return;
  const value={...draft,exportedAt:new Date().toISOString()};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='bcg-ui-copy.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  notice('已导出 bcg-ui-copy.json。把文件发给维护者，即可按修改稿部署。');
}
$('#replacement-copy').addEventListener('input',()=>{clearTimeout(saveTimer);saveTimer=setTimeout(commit,450);});
$('#replacement-copy').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();commit();}});
$('#apply-copy').addEventListener('click',commit);
$('#pick-mode').addEventListener('click',()=>setMode('pick'));
$('#browse-mode').addEventListener('click',()=>setMode('browse'));
$('#copy-search').addEventListener('input',renderList);
$('#changed-only').addEventListener('change',renderList);
$('#reset-entry').addEventListener('click',()=>{
  if(!selected||!compatible)return;
  $('#replacement-copy').value=baseline(catalog.get(selected));commit();
});
$('#undo-copy').addEventListener('click',()=>{
  clearTimeout(saveTimer);if(!history.length||!compatible)return;
  draft=history.pop();persist();
  if(selected)$('#replacement-copy').value=currentValue(catalog.get(selected));
  sync();notice('已撤销上次修改。');
});
$('#reset-all').addEventListener('click',()=>{
  if(!confirm('清空当前浏览器的全部文案修改？Case 和标注数据不会改变。'))return;
  clearTimeout(saveTimer);
  if(compatible)history.push(structuredClone(draft));else history.length=0;
  draft=emptyDraft(published.revision);compatible=true;
  if(selected){$('#replacement-copy').value=currentValue(catalog.get(selected));$('#replacement-copy').disabled=false;}
  persist();sync();notice('草稿已清空，预览恢复为正式文案。');
});
$('#export-copy').addEventListener('click',exportDraft);
$('#import-copy').addEventListener('click',()=>$('#copy-file').click());
$('#copy-file').addEventListener('change',async event=>{
  const file=event.target.files[0];event.target.value='';if(!file)return;
  try{
    if(file.size>4*1024*1024)throw new Error('修改稿不能超过 4 MB。');
    const incoming=validateCopy(JSON.parse(await file.text()),true);mergedCopy(published,incoming);
    if(draft.entries.length&&!confirm('导入后替换当前文案草稿？现有草稿可先导出备份。'))return;
    clearTimeout(saveTimer);history.push(structuredClone(draft));draft=incoming;compatible=true;
    for(const entry of draft.entries)if(!catalog.has(copyKey(entry)))catalog.set(copyKey(entry),entry);
    if(selected){$('#replacement-copy').value=currentValue(catalog.get(selected));$('#replacement-copy').disabled=false;}
    persist();sync();notice('已导入修改稿，预览已更新。');
  }catch(error){notice(`导入失败：${error.message}`,true);}
});
document.querySelectorAll('[data-jump]').forEach(button=>button.addEventListener('click',()=>{
  if(!commit())return;setMode('browse');send('navigate',{target:button.dataset.jump});
}));
addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==frame.contentWindow||event.data?.channel!=='bcg-copy-editor')return;
  const message=event.data;
  if(message.type==='ready'){
    ready=true;sync();send('mode',{mode});send('catalog-request');
    $('#preview-status').textContent=mode==='pick'?'点击页面文字进行编辑':'浏览预览页面';
  }
  if(message.type==='catalog'){
    for(const entry of message.entries||[]) {
      try{validateEntry({...entry,value:entry.original});catalog.set(copyKey(entry),entry);}catch{}
    }
    renderList();
  }
  if(message.type==='selected')select(message.entry);
  if(message.type==='notice')notice(message.text);
});
addEventListener('beforeunload',event=>{
  if(!commit()){event.preventDefault();event.returnValue='';}
});
try{
  const response=await fetch('./copy-overrides.json',{cache:'no-store'});
  if(!response.ok)throw new Error('无法读取正式文案');
  published=validateCopy(await response.json());draft=emptyDraft(published.revision);
  try{
    const saved=localStorage.getItem(STORAGE);
    if(saved){draft=validateCopy(JSON.parse(saved),true);mergedCopy(published,draft);}
    $('#draft-status').textContent='草稿保存在当前浏览器';
  }catch(error){compatible=false;notice(error.message+' 请导出备份后清空草稿。',true);$('#draft-status').textContent='草稿需要核对';}
  for(const entry of draft.entries)catalog.set(copyKey(entry),entry);
  refreshCount();renderList();frame.src='./index.html?copy-editor=1';
}catch(error){notice(error.message+'。请刷新重试。',true);$('#preview-status').textContent='页面未载入';}

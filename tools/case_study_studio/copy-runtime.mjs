import {copyKey, describeText, emptyPublished, mergedCopy, renderCopy, validateCopy} from './copy-model.mjs';

// Exclude imported content, editable annotation values, IDs and source metadata.
// No case data is included in the copy catalog or sent to the editor.
const LOCKED = [
  '[data-copy-lock]', 'pre', 'script', 'style', 'svg', '.case-item', '.case-heading h1',
  '.breadcrumb strong', '.record-header .role', '.record-header .mono',
  '.belief-detail > p', '.belief-detail > .small', '.relation-list > div', '.node-row',
  '.context-meta', '.chip-row', '.file-list', '.event-reference p', '.event-switcher',
  '#agent-select option',
  '#role-select option:not([value=""])', '.demo-overview p', '.dimension-id',
  '.case-kicker > .badge:not(#review-status)',
].join(',');
const letter = /[\p{L}]/u;

export async function mountCopyRuntime({fragments = [], navigate} = {}) {
  const editor = new URLSearchParams(location.search).has('copy-editor') && parent !== window;
  let published = emptyPublished(), map = new Map(), mode = 'browse';
  try {
    const response = await fetch('./copy-overrides.json', {cache:'no-store'});
    if (!response.ok) throw new Error('无法读取正式文案');
    published = validateCopy(await response.json()); map = mergedCopy(published);
  } catch (error) {
    console.warn('界面文案使用源代码默认值：', error.message);
  }
  if (!editor && !map.size) return;
  const catalog = new Map(), records = new WeakMap(), attributes = new WeakMap();
  let highlighted, queued = false, catalogChanged = false;
  const send = (type, extra = {}) => {
    if (editor) parent.postMessage({channel:'bcg-copy-editor',type,...extra}, location.origin);
  };
  const register = (kind, text, dynamic, label) => {
    const description = describeText(text, dynamic);
    if (!letter.test(description.original.replace(/\{\{n\d+\}\}/g,''))) return null;
    const entry = {kind, original:description.original}, key = copyKey(entry);
    if (!catalog.has(key)) {
      catalog.set(key, {...entry, label}); catalogChanged = true;
    }
    return {key, description};
  };
  function textNode(node, dynamic, label, apply) {
    let record = records.get(node);
    if (!record || node.data !== record.displayed) {
      record = register('text', node.data, dynamic, label);
      if (!record) return;
      records.set(node, record);
    }
    const value = renderCopy(map.get(record.key), record.description);
    if (apply && node.data !== value) node.data = value;
    record.displayed = apply ? value : node.data;
  }
  function element(el, dynamic, label, apply) {
    if (el.tagName === 'OPTION' && !el.hasAttribute('value')) {
      // HTML defaults option.value to its text. Pin the data value first.
      el.setAttribute('value', el.textContent);
    }
    for (const kind of ['placeholder','title','aria-label']) {
      if (!el.hasAttribute(kind) || (kind === 'placeholder' && el.matches('[data-field="goal"]'))) continue;
      let store = attributes.get(el);
      if (!store) { store = new Map(); attributes.set(el, store); }
      let record = store.get(kind), value = el.getAttribute(kind);
      if (!record || record.displayed !== value) {
        record = register(kind, value, false, label);
        if (!record) continue;
        store.set(kind, record);
      }
      value = renderCopy(map.get(record.key), record.description);
      if (apply && el.getAttribute(kind) !== value) el.setAttribute(kind, value);
      record.displayed = apply ? value : el.getAttribute(kind);
    }
  }
  function scan(root, dynamic, label, apply = true) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (node.nodeType === 1 && node.matches(LOCKED)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    let node;
    while ((node = walker.nextNode())) {
      if (node.nodeType === 1) element(node, dynamic, label, apply);
      else if (!node.parentElement?.matches('textarea,input')) textNode(node, dynamic, label, apply);
    }
  }
  const observer = new MutationObserver(()=>schedule());
  function observe() {
    for (const root of [document.querySelector('#app'), document.querySelector('#help-dialog'), document.querySelector('#dimension-dialog'), document.querySelector('title')]) {
      if(root)observer.observe(root, {subtree:true,childList:true,characterData:true,attributes:true,
        attributeFilter:['placeholder','title','aria-label']});
    }
  }
  function sync() {
    queued = false; observer.disconnect();
    scan(document.querySelector('#app'), true, '工作台');
    scan(document.querySelector('#help-dialog'), false, '标注教程');
    scan(document.querySelector('#dimension-dialog'), false, '自定义维度');
    scan(document.querySelector('title'), false, '浏览器标题');
    observe();
    if(catalogChanged) {
      catalogChanged = false;
      send('catalog', {entries:[...catalog.values()]});
    }
  }
  function schedule() { if(!queued){queued=true;queueMicrotask(sync);} }
  // Discover panels that aren't currently open using empty, non-persisted data.
  if(editor)for(const html of fragments) {
    const root=document.createElement('div'); root.innerHTML=html;
    scan(root,true,'工作台',false);
  }
  sync();
  if(!editor)return;

  function clearHighlight() { highlighted?.classList.remove('copy-hover');highlighted=null; }
  function pick(event) {
    const el=event.target instanceof Element ? event.target : event.target.parentElement;
    if(!el?.closest('#app,#help-dialog,#dimension-dialog') || el.closest(LOCKED))return null;
    const attr=attributes.get(el);
    if(attr) {
      const r=attr.get('placeholder') || attr.get('aria-label') || attr.get('title');
      if(r)return {record:r,el};
    }
    const caret=document.caretPositionFromPoint?.(event.clientX,event.clientY);
    const text=caret?.offsetNode || document.caretRangeFromPoint?.(event.clientX,event.clientY)?.startContainer;
    if(text?.nodeType===3 && records.has(text) && el.contains(text) && !text.parentElement.closest(LOCKED)) {
      return {record:records.get(text),el:text.parentElement};
    }
    const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);
    let node;while((node=walker.nextNode()))if(records.has(node)&&!node.parentElement.closest(LOCKED)) {
      return {record:records.get(node),el:node.parentElement};
    }
    return null;
  }
  document.addEventListener('pointermove', event=>{
    if(mode!=='pick')return;
    const next=pick(event)?.el;
    if(highlighted!==next){clearHighlight();highlighted=next;highlighted?.classList.add('copy-hover');}
  },true);
  document.addEventListener('click', event=>{
    if(mode!=='pick')return;
    const target=pick(event);
    event.preventDefault();event.stopImmediatePropagation();
    if(target)send('selected',{entry:catalog.get(target.record.key)});
    else send('notice',{text:'这里是运行数据或原始记录。请点选标题、按钮、说明或表单提示。'});
  },true);
  addEventListener('message', event=>{
    if(event.origin!==location.origin || event.source!==parent || event.data?.channel!=='bcg-copy-editor')return;
    const message=event.data;
    try {
      if(message.type==='apply') {
        published=validateCopy(message.published);
        map=mergedCopy(published,message.draft);
        sync();
      }
      if(message.type==='mode') {
        mode=message.mode==='pick'?'pick':'browse'; clearHighlight();
        document.documentElement.classList.toggle('copy-picking',mode==='pick');
      }
      if(message.type==='navigate')navigate?.(message.target);
      if(message.type==='catalog-request')send('catalog',{entries:[...catalog.values()]});
    } catch(error) {send('notice',{text:error.message});}
  });
  send('ready',{revision:published.revision});
}

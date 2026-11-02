// Task selection is independent of run creation. All changes remain in the form
// until the user explicitly confirms a launch.
export function parseTaskIds(text, limit=32) {
  const input=String(text).trim().replace(/^--task-ids(?:\s+|=)/,'').trim()
    .replace(/^(["'])([\s\S]*)\1$/,'$2').replace(/^drb2\s*:\s*/i,'')
    .replace(/(\d)\s*[-–]\s*(?=\d)/g,'$1-');
  const ids=new Set();
  for(const token of input.split(/[,，;；\s]+/).filter(Boolean)) {
    const match=/^(\d+)(?:-(\d+))?$/.exec(token);
    if(!match) throw new Error(`无法识别题号「${token}」，请使用 2,5,30 或 20-25。`);
    const start=Number(match[1]),end=Number(match[2]??match[1]);
    if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<1||end<start)
      throw new Error(`题号范围「${token}」无效。`);
    if(end-start+1>limit) throw new Error(`单次最多选择 ${limit} 题。`);
    for(let id=start;id<=end;id++) {
      ids.add(String(id));
      if(ids.size>limit) throw new Error(`单次最多选择 ${limit} 题。`);
    }
  }
  return [...ids];
}

export function mergeTaskIds(current, added, limit, available=null) {
  const ids=[...new Set([...current,...added])];
  if(available) {
    const missing=ids.filter(id=>!available.has(id));
    if(missing.length) throw new Error(`题库中没有这些题号：${missing.join('、')}。`);
  }
  if(ids.length>limit) throw new Error(`单次最多选择 ${limit} 题，请先移除部分题号。`);
  return ids;
}

export function filterTasks(tasks, query) {
  if(/^\d+$/.test(query.trim())) return tasks.filter(task=>task.id===String(Number(query.trim())));
  const words=query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return tasks.filter(task=>words.every(word=>
    `${task.id} ${task.preview||''} ${task.theme||''} ${task.language||''}`.toLocaleLowerCase().includes(word)));
}

export function taskIdsInRect(items, start, end) {
  const left=Math.min(start.x,end.x),right=Math.max(start.x,end.x);
  const top=Math.min(start.y,end.y),bottom=Math.max(start.y,end.y);
  return items.filter(item=>item.right>left&&item.left<right&&item.bottom>top&&item.top<bottom).map(item=>item.id);
}

export function mountTaskPicker(root, {defaults, limit, catalog}) {
  const $=selector=>root.querySelector(selector),doc=root.ownerDocument;
  const selectedBox=$('#selected-tasks'),grid=$('#task-catalog-grid'),input=$('#task-id-input');
  const search=$('#task-catalog-search'),error=$('#task-picker-error'),hidden=$('[name=drb2]');
  const tasks=catalog?.available?catalog.tasks:defaults.map(id=>({id,preview:''}));
  const available=catalog?.available?new Set(tasks.map(task=>task.id)):null;
  const byId=new Map(tasks.map(task=>[task.id,task]));
  let selected=[];
  let marked=new Set(),drag=null,dragFrame=0,swallowClick=false;
  const win=doc.defaultView,marquee=doc.createElement('div');
  marquee.className='task-selection-marquee';marquee.hidden=true;marquee.setAttribute('aria-hidden','true');

  function renderMarked() {
    for(const chip of selectedBox.querySelectorAll('.task-chip')) {
      const active=marked.has(chip.dataset.removeTask);
      chip.classList.toggle('is-marked',active);
      chip.setAttribute('aria-label',`移除题号 ${chip.dataset.removeTask}${active?'，已框选':''}`);
    }
    const count=`框选 ${marked.size} 题`;
    if($('#task-marked-count').textContent!==count)$('#task-marked-count').textContent=count;
    $('#task-remove-marked').disabled=!marked.size;$('#task-unmark').disabled=!marked.size;
  }

  function point(clientX,clientY) {
    const box=selectedBox.getBoundingClientRect();
    return {x:Math.max(0,Math.min(selectedBox.clientWidth,clientX-box.left-selectedBox.clientLeft))+selectedBox.scrollLeft,
      y:Math.max(0,Math.min(selectedBox.clientHeight,clientY-box.top-selectedBox.clientTop))+selectedBox.scrollTop};
  }

  function updateDrag() {
    if(!drag?.active)return;
    const end=point(drag.x,drag.y),box=selectedBox.getBoundingClientRect();
    const items=[...selectedBox.querySelectorAll('.task-chip')].map(chip=>{
      const r=chip.getBoundingClientRect(),dx=selectedBox.scrollLeft-box.left-selectedBox.clientLeft,dy=selectedBox.scrollTop-box.top-selectedBox.clientTop;
      return {id:chip.dataset.removeTask,left:r.left+dx,right:r.right+dx,top:r.top+dy,bottom:r.bottom+dy};
    });
    marked=new Set([...drag.base,...taskIdsInRect(items,drag.start,end)]);
    Object.assign(marquee.style,{left:Math.min(drag.start.x,end.x)+'px',top:Math.min(drag.start.y,end.y)+'px',
      width:Math.abs(end.x-drag.start.x)+'px',height:Math.abs(end.y-drag.start.y)+'px'});
    renderMarked();
  }

  function dragScroll() {
    if(!drag?.active)return;
    const box=selectedBox.getBoundingClientRect(),top=box.top+selectedBox.clientTop,bottom=top+selectedBox.clientHeight;
    if(drag.x>=box.left&&drag.x<=box.right) {
      if(drag.y<top+20)selectedBox.scrollTop-=8;
      else if(drag.y>bottom-20)selectedBox.scrollTop+=8;
    }
    updateDrag();dragFrame=win.requestAnimationFrame(dragScroll);
  }

  function finishDrag(cancel=false) {
    if(!drag)return;
    const previous=drag;drag=null;win.cancelAnimationFrame(dragFrame);
    if(cancel)marked=previous.previous;
    if(previous.active)swallowClick=true;
    if(selectedBox.hasPointerCapture(previous.pointerId))selectedBox.releasePointerCapture(previous.pointerId);
    selectedBox.classList.remove('is-dragging');marquee.hidden=true;renderMarked();
  }

  function clearMarked() {finishDrag(true);marked.clear();renderMarked();}

  selectedBox.addEventListener('pointerdown',event=>{
    swallowClick=false;
    if(event.button!==0||event.pointerType==='touch'||!event.isPrimary||!selected.length)return;
    const box=selectedBox.getBoundingClientRect();
    // Keep native scrollbar dragging and touch scrolling intact.
    if(event.clientX>=box.left+selectedBox.clientLeft+selectedBox.clientWidth||event.clientY>=box.top+selectedBox.clientTop+selectedBox.clientHeight)return;
    finishDrag(true);
    drag={pointerId:event.pointerId,start:point(event.clientX,event.clientY),originX:event.clientX,originY:event.clientY,
      x:event.clientX,y:event.clientY,active:false,previous:new Set(marked),
      base:event.ctrlKey||event.metaKey||event.shiftKey?new Set(marked):new Set()};
  });
  win.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.pointerId)return;
    drag.x=event.clientX;drag.y=event.clientY;
    if(!drag.active&&Math.hypot(drag.x-drag.originX,drag.y-drag.originY)>=5) {
      drag.active=true;selectedBox.focus({preventScroll:true});selectedBox.setPointerCapture(drag.pointerId);
      selectedBox.classList.add('is-dragging');marquee.hidden=false;
      dragFrame=win.requestAnimationFrame(dragScroll);
    }
    if(drag.active){event.preventDefault();updateDrag();}
  });
  win.addEventListener('pointerup',event=>{if(drag?.pointerId===event.pointerId){updateDrag();finishDrag();}});
  win.addEventListener('pointercancel',event=>{if(drag?.pointerId===event.pointerId)finishDrag(true);});
  selectedBox.addEventListener('lostpointercapture',event=>{if(event.target===selectedBox&&event.pointerId===drag?.pointerId)finishDrag(true);});
  win.addEventListener('blur',()=>finishDrag(true));
  selectedBox.addEventListener('dragstart',event=>event.preventDefault());
  selectedBox.addEventListener('click',event=>{
    if(swallowClick&&event.detail>0){event.preventDefault();event.stopImmediatePropagation();swallowClick=false;}
  },true);
  root.addEventListener('keydown',event=>{
    if(event.key==='Escape'&&(drag||marked.size)) {event.preventDefault();event.stopPropagation();clearMarked();}
    else if((event.key==='Delete'||event.key==='Backspace')&&marked.size&&selectedBox.contains(event.target)) {
      event.preventDefault();apply(selected.filter(id=>!marked.has(id)));selectedBox.focus({preventScroll:true});
    }
  });
  root.closest('dialog')?.addEventListener('close',clearMarked);

  function showError(message='') {
    error.textContent=message;
    input.setAttribute('aria-invalid',String(Boolean(message)));
  }

  function button(label, className) {
    const element=doc.createElement('button');
    element.type='button';element.className=className;element.textContent=label;
    return element;
  }

  function render() {
    hidden.value=selected.join(',');
    $('#task-selection-count').textContent=`已选 ${selected.length} / ${limit} 题`;
    $('#task-selection-count').classList.toggle('at-limit',selected.length===limit);
    $('#task-clear').disabled=selected.length===0;
    selectedBox.replaceChildren(...selected.map(id=>{
      const chip=button(id,'task-chip');chip.dataset.removeTask=id;
      chip.setAttribute('aria-label',`移除题号 ${id}`);chip.title=byId.get(id)?.preview||`移除题号 ${id}`;
      const cross=doc.createElement('span');cross.textContent='×';cross.setAttribute('aria-hidden','true');
      chip.append(cross);return chip;
    }));
    if(!selected.length) {
      const empty=doc.createElement('span');empty.className='task-selection-empty';
      empty.textContent='尚未选择题目，可添加题号或从题库勾选。';selectedBox.append(empty);
    }
    selectedBox.append(marquee);renderMarked();
    const matches=filterTasks(tasks,search.value);
    $('#task-catalog-count').textContent=`${matches.length} / ${tasks.length} 题`;
    const choose=$('#task-select-filtered');
    choose.disabled=!matches.length || new Set([...selected,...matches.map(task=>task.id)]).size>limit;
    choose.title=choose.disabled?`选中结果后不能超过 ${limit} 题`:'将当前筛选结果加入已选题目';
    grid.replaceChildren(...matches.map(task=>{
      const checked=selected.includes(task.id),item=button(task.id,'task-catalog-item');
      item.dataset.taskId=task.id;item.setAttribute('aria-pressed',String(checked));
      item.setAttribute('aria-label',`题号 ${task.id}${task.preview?'：'+task.preview:''}`);
      item.title=`${task.id}${task.preview?' · '+task.preview:''}`;
      item.disabled=!checked&&selected.length>=limit;
      return item;
    }));
    if(!matches.length) {
      const empty=doc.createElement('span');empty.className='task-selection-empty';
      empty.textContent='没有匹配的题目。';grid.append(empty);
    }
  }

  function apply(ids) {
    const next=mergeTaskIds([],ids,limit,available);
    clearMarked();selected=next;showError();render();
  }

  function addInput() {
    if(!input.value.trim()) return;
    const ids=parseTaskIds(input.value,limit);
    apply(mergeTaskIds(selected,ids,limit,available));input.value='';
  }

  function restore() {
    apply(available?defaults.filter(id=>available.has(id)):defaults);
    input.value='';
    if(selected.length!==defaults.length) showError('部分默认题号不在当前题库，已选中其余默认题目。');
  }

  root.addEventListener('click',event=>{
    const target=event.target.closest('button');
    if(!target || !root.contains(target)) return;
    try {
      if(target.dataset.removeTask) {
        if(event.ctrlKey||event.metaKey||event.shiftKey) {
          const id=target.dataset.removeTask;
          if(marked.has(id))marked.delete(id);else marked.add(id);
          renderMarked();return;
        }
        const index=selected.indexOf(target.dataset.removeTask);
        apply(selected.filter(id=>id!==target.dataset.removeTask));
        (selectedBox.querySelectorAll('.task-chip')[Math.min(index,selected.length-1)]||input).focus();
      } else if(target.dataset.taskId) {
        const id=target.dataset.taskId,scroll=grid.scrollTop;
        apply(selected.includes(id)?selected.filter(value=>value!==id):[...selected,id]);
        grid.scrollTop=scroll;grid.querySelector(`[data-task-id="${id}"]`)?.focus({preventScroll:true});
      } else if(target.id==='task-add') {addInput();input.focus();}
      else if(target.id==='task-clear') {apply([]);input.value='';input.focus();}
      else if(target.id==='task-remove-marked') {apply(selected.filter(id=>!marked.has(id)));selectedBox.focus({preventScroll:true});}
      else if(target.id==='task-unmark') clearMarked();
      else if(target.id==='task-restore') restore();
      else if(target.id==='task-select-filtered')
        apply(mergeTaskIds(selected,filterTasks(tasks,search.value).map(task=>task.id),limit,available));
    } catch(exc) {showError(exc.message);}
  });
  input.addEventListener('keydown',event=>{
    if(event.key!=='Enter')return;
    event.preventDefault();
    try{addInput();}catch(exc){showError(exc.message);}
  });
  input.addEventListener('input',()=>showError());
  search.addEventListener('input',render);
  search.addEventListener('keydown',event=>{if(event.key==='Enter')event.preventDefault();});
  $('#task-catalog-note').textContent=catalog?.available?'搜索题号或关键词，点击题号勾选。':'题库尚未加载，下方显示默认题组；也可直接添加其他题号。';
  restore();
  return {value() {
    try {
      addInput();
      if(!selected.length) throw new Error('请至少选择一道题。');
      return [...selected];
    } catch(exc) {showError(exc.message);input.focus();throw exc;}
  }};
}

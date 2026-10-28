import {allDimensions, addCustomDimension, updateCustomDimension, removeCustomDimension, OPTIONS, IMPROVEMENT_STAGES, updateDemoTitle, uid, pretty, textOf, ingestFiles, graphViews, agentTrajectories, contextTurn, contextBeliefText, recordPresentation, eventEvidencePresentation, eventBeliefContext, eventFromTrajectory, newEvent, selectedDimensions, hasNoApplicableDimensions, setDimensionSelected, confirmNoApplicableDimensions, linkedEvents, setDimensionEvent, removeEvent, eventLabel, conclusionObservations, setConclusionFocus, completion, validateReview, tokenMetrics, markdownReport, csvSummary, workspaceExport, restoreWorkspace} from './model.mjs';
import {REAL_DEMO_ID, buildDemoCase, demoLabel} from './demo.mjs';
import {mountCopyRuntime} from './copy-runtime.mjs';
import {renderMarkdown} from './markdown.mjs';
import {VIEW_STORAGE_KEY, DEFAULT_VIEW, normalizeView, readViewState} from './view-state.mjs';
import {BUNDLE_FORMAT, MAX_IMPORT_BYTES, readImportFile, importBundles} from './bundles.mjs';
import {remoteLocation, loadSharedCase, loadFullSharedEvidence, saveSharedCase, caseLink, benchmarkLabel, canAnnotateTask} from './shared.mjs';

const $ = (q, root = document) => root.querySelector(q);
const escape = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const num = x => typeof x === 'number' ? new Intl.NumberFormat('en-US').format(x) : '—';
const state = {cases: [], active: '', ...DEFAULT_VIEW, sidebarSearch: '', attach: false, busy: false};
const annotationStages = [['basic','基本信息'], ['events','关键过程'], ['dimensions','维度评价'], ['conclusion','归因总结']];
let db, pending = new Map(), saveTimer, toastTimer, saving = 0, saveError = '', fullEvidenceTask;
let writeQueue = Promise.resolve();
const deletingCases = new Set();
let guideStep = 0;
let dimensionEditorCase = '', dimensionEditorId = '';
let navigation = readViewState(null, []), viewSaveTimer, renderedCase = '', viewSaveWarning = false;
let remote = null, sharedRun = null, startupError = '';
try {remote = remoteLocation();} catch (error) {startupError = error.message;}
const sharedMode = new URLSearchParams(location.search).has('run');
const viewStorageKey = VIEW_STORAGE_KEY + (sharedMode ? ':shared:' + location.search : new URLSearchParams(location.search).has('copy-editor') ? ':copy-editor' : '');
const current = () => state.cases.find(c => c.id === state.active);
const aiReadOnly = () => !!current()?.remote?.readOnly;
const reviewMutations = '[data-field], [data-toggle-dimension], [data-conclusion-focus], [data-no-dimensions], [data-link-dimension], [data-action="complete"], [data-action="cite"], [data-action="choose-event-source"], [data-action="delete-event"], [data-action="add-dimension"], [data-action="edit-dimension"], [data-action="delete-dimension"]';
const icons = {
  plus: '<path d="M12 5v14M5 12h14"/>', upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>', check: '<path d="m5 12 4 4 10-10"/>',
  download: '<path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4"/>', search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',
  file: '<path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8m-8 4h6"/>', folder: '<path d="M3 7h7l2 2h9v11H3zM3 7V4h7l2 3h8v2"/>',
  book: '<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1v15"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.file}</svg>`;
const button = (label, action, cls = 'button', extra = '') => `<button class="${cls}" data-action="${action}" ${extra}>${label}</button>`;

function openDB() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open('bcg-case-study-studio', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('cases', {keyPath: 'id'});
    r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
  });
}
function dbOperation(action, value) {
  if (!db) return Promise.reject(new Error('浏览器存储不可用，请及时导出工作包。'));
  return new Promise((resolve, reject) => {
    const tx = db.transaction('cases', ['getAll','get'].includes(action) ? 'readonly' : 'readwrite');
    const r = tx.objectStore('cases')[action](...(value === undefined ? [] : [value]));
    tx.oncomplete = () => resolve(r.result); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error || new Error('保存中断'));
  });
}
function dbPutCases(cases) {
  if (!db) return Promise.reject(new Error('浏览器存储不可用，无法保存导入的 Case。'));
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('cases','readwrite'), store=tx.objectStore('cases');
    for(const c of cases) store.put(c);
    tx.oncomplete=()=>resolve();
    tx.onerror=()=>reject(tx.error);
    tx.onabort=()=>reject(tx.error || new Error('批量保存中断'));
  });
}
function enqueueWrite(operation) {
  const result = writeQueue.then(operation);
  writeQueue = result.catch(() => {});
  return result;
}
function saveLabel(text, bad = false) {
  const el = $('#save-label'); if (el) { el.textContent = text; el.classList.toggle('save-error', bad); }
}
async function flush() {
  clearTimeout(saveTimer);
  const batch = [...pending.values()]; pending.clear();
  if (!batch.length) return;
  saving++;
  try {
    await enqueueWrite(async () => {
      for (const c of batch) {
        if(c.remote?.readOnly)continue;
        if (!sharedMode) {await dbOperation('put', c);continue;}
        // Persist a recovery draft first. Server success is authoritative even if
        // IndexedDB is full; failing local storage must not block shared saving.
        if (db) await dbOperation('put', c).catch(() => {});
        if (c.remote?.conflict) throw new Error('另一页面已更新标注，请先导出草稿，再加载服务器版本');
        await saveSharedCase(c);
        if (db) await dbOperation('put', c).catch(() => {});
      }
    }); saveError = '';
  }
  catch (e) {
    for (const c of batch) {
      const latest = state.cases.find(item => item.id === c.id);
      if (latest && !deletingCases.has(c.id)) pending.set(c.id, latest);
    }
    saveError = e.message; toast(`自动保存失败：${e.message}`, true);
  }
  finally {saving--; saveLabel(saveError ? '未保存 · 请导出备份' : pending.size || saving ? '正在保存…' : sharedMode ? '已保存到服务器' : '已保存到此浏览器', !!saveError);}
}
function dirty(c = current()) {
  if (!c || c.remote?.readOnly || deletingCases.has(c.id)) return;
  c.updatedAt = new Date().toISOString();
  if (c.annotation.status === 'reviewed') c.annotation.status = 'draft';
  if (sharedMode && c.remote) c.remote.dirty = true;
  pending.set(c.id, c); saveLabel('正在保存…');
  clearTimeout(saveTimer); saveTimer = setTimeout(flush, 450);
  updateProgress();
}
function toast(message, error = false) {
  const el = $('#toast'); el.textContent = message; el.className = error ? 'visible error' : 'visible';
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.className = ''; }, error ? 9000 : 4000);
}
function updateProgress() {
  const c = current(); if (!c) return;
  const p = completion(c);
  const el = $('#review-progress'); if (el) el.textContent = dimensionProgress(c);
  const meter = $('#progress-meter'); if (meter) meter.value = p.percent;
  const queue = $(`[data-case-progress="${c.id}"]`); if (queue) queue.textContent = dimensionProgress(c, true);
  const chip = $('#review-status'); if (chip) { chip.textContent = c.annotation.status === 'reviewed' ? '已完成' : '标注中'; chip.className = `badge ${c.annotation.status === 'reviewed' ? 'green' : 'neutral'}`; }
  const readiness = $('#conclusion-readiness');
  if (readiness) {
    const open = $('details', readiness)?.open;
    readiness.innerHTML = renderConclusionReadiness(c);
    if (open && $('details', readiness)) $('details', readiness).open = true;
  }
}
function dimensionProgress(c, compact = false) {
  const p = completion(c);
  return p.total ? compact ? `${p.completed}/${p.total}` : `${p.completed} / ${p.total} 个已选维度` :
    hasNoApplicableDimensions(c) ? '无相关维度' : '尚未选择维度';
}
function rememberView() {
  clearTimeout(viewSaveTimer);
  if (current() && renderedCase === state.active) {
    navigation.cases[state.active] = normalizeView({...state,
      evidenceScroll: $('#evidence-content')?.scrollTop ?? 0,
      formScroll: $('#form-content')?.scrollTop ?? 0,
    });
  }
  navigation.active = state.active;
  for (const id of Object.keys(navigation.cases)) if (!state.cases.some(c => c.id === id)) delete navigation.cases[id];
  try { localStorage.setItem(viewStorageKey, JSON.stringify(navigation)); }
  catch {
    if (!viewSaveWarning) {viewSaveWarning=true;toast('浏览器未允许保存页面位置，当前标注仍按原方式保存。',true);}
  }
}
function scheduleViewSave() {
  clearTimeout(viewSaveTimer);
  viewSaveTimer = setTimeout(rememberView, 80);
}
function activateCase(id) {
  rememberView();
  state.active = state.cases.some(c => c.id === id) ? id : '';
  Object.assign(state, normalizeView(navigation.cases[state.active]));
  // Never capture the outgoing DOM's scroll position as the incoming case's position.
  renderedCase = '';
}

function showGuide(step = guideStep) {
  if(sharedMode && !$('#help-dialog').dataset.sharedGuide) {
    $('#help-dialog').dataset.sharedGuide='true';
    $('[data-guide-page="0"]').innerHTML='<h3 id="guide-step-heading-0" tabindex="-1">从运行编号打开任务</h3><p class="guide-location">运行中心 → 选择 RUN 编号 → 人工标注</p><ol class="guide-actions-list"><li>填写昵称进入共享工作区。选择一次运行，查看各任务进度。</li><li>任务结束后点击「人工标注」，任务、最终报告、轨迹和 Belief Graph 自动载入。</li><li>左侧阅读轨迹与每轮 Belief 注入，右侧记录关键过程、维度评价与归因总结。</li><li>标注自动保存到服务器，每个人独立填写。换浏览器后用个人身份码恢复自己的标注。</li></ol><p>第一步阅读任务和最终报告，接着从运行轨迹引用关键过程。</p>';
    $('[data-guide-step="0"]').textContent='打开任务';
  }
  const pages = [...document.querySelectorAll('[data-guide-page]')];
  guideStep = Math.max(0, Math.min(pages.length - 1, step));
  pages.forEach((el, i) => { el.hidden = i !== guideStep; });
  document.querySelectorAll('[data-guide-step]').forEach(el => {
    if (Number(el.dataset.guideStep) === guideStep) el.setAttribute('aria-current', 'step');
    else el.removeAttribute('aria-current');
  });
  $('#guide-page-number').textContent = `${guideStep + 1} / ${pages.length}`;
  $('#guide-prev').disabled = guideStep === 0;
  $('#guide-next').textContent = guideStep === pages.length - 1 ? '返回工作台' : '下一步';
  document.querySelectorAll('[data-guide-open]').forEach(el => { el.disabled = !current(); });
  document.querySelectorAll('.guide-needs-case').forEach(el => { el.hidden = !!current(); });
  $('#guide-content').scrollTop = 0;
  const dialog = $('#help-dialog');
  if (!dialog.open) dialog.showModal();
  $('#guide-step-heading-' + guideStep).focus({preventScroll: true});
}

function openStage(stage) {
  if (!current()) return showGuide(0);
  if (!annotationStages.some(([id]) => id === stage)) return;
  $('#help-dialog').close();
  if (state.form === stage) return;
  state.form = stage;
  renderFormOnly();
  $('#form-content').scrollTop = 0;
}

function renderSidebar() {
  if (sharedMode) return `<aside class="sidebar shared-sidebar"><a class="brand" href="./?run=${remote?.run || ''}"><img src="./icon.svg" alt="" width="37" height="37"><span>BCG<span class="brand-sub">CASE STUDY STUDIO</span></span></a>
    <div class="workspace-label"><span class="eyebrow">${escape(sharedRun?.number || '共享运行')}</span></div>
    <a class="button full" href="./?run=${remote?.run || ''}">返回运行进度</a>
    <div class="queue-heading">本次运行的任务</div><nav class="case-list" aria-label="Case 列表">${(sharedRun?.tasks || []).map(t=>`<div class="case-row"><a class="case-item ${t.benchmark===remote.benchmark && t.id===remote.task?'active':''}" ${canAnnotateTask(t)?`href="${escape(caseLink(remote.run,t.benchmark,t.id,aiReadOnly()&&t.ai_review?.status==='completed'?'ai':undefined))}"`:'aria-disabled="true"'}><span class="case-item-top"><span class="mono">${escape(t.id)}</span></span><span class="case-name">${escape(benchmarkLabel(t.benchmark))}</span><span class="case-item-bottom">${!canAnnotateTask(t)?'暂不可标注':aiReadOnly()&&t.ai_review?.status==='completed'?'查看 AI 标注':'人工标注'}</span></a></div>`).join('')}</nav>
    <div class="sidebar-foot">${button(icon('book')+'标注教程','help','button ghost full')}<div class="privacy">${icon('lock')}<div>${escape(current()?.remote?.user?.name || '共享工作区')}<span>独立标注 · 服务器保存</span></div></div></div></aside>`;
  const cases = state.cases.filter(c => `${c.taskId} ${c.title}`.toLowerCase().includes(state.sidebarSearch.toLowerCase()));
  return `<aside class="sidebar"><a class="brand" href="./" aria-label="BCG Case Study Studio"><img src="./icon.svg" alt="" width="37" height="37"><span>BCG<span class="brand-sub">CASE STUDY STUDIO</span></span></a>
    <div class="workspace-label"><span class="eyebrow">研究工作区</span><span class="tiny-count">${state.cases.length}</span></div>
    ${button(icon('plus') + '导入 Case', 'import', 'button primary full')}
    <label class="sidebar-search">${icon('search')}<input id="case-search" placeholder="搜索 Case…" value="${escape(state.sidebarSearch)}" aria-label="搜索 Case"></label>
    <div class="queue-heading">CASE QUEUE <span>${state.cases.filter(c => c.annotation.status === 'reviewed').length} 已完成</span></div>
    <nav class="case-list" aria-label="Case 列表">${cases.length ? cases.map(c => `<div class="case-row"><button class="case-item ${c.id === state.active ? 'active' : ''}" data-case="${escape(c.id)}"><span class="case-item-top"><span class="mono">${escape(c.taskId)}</span><span class="status-dot ${c.annotation.status === 'reviewed' ? 'done' : ''}" title="${c.annotation.status === 'reviewed' ? '已完成' : '草稿'}"></span></span><span class="case-name">${escape(c.title)}</span><span class="case-item-bottom">${escape(c.benchmark || (c.demo ? '示例 · 非真实实验' : '本地 Case'))}<span data-case-progress="${escape(c.id)}">${dimensionProgress(c, true)}</span></span></button><button type="button" class="case-delete" data-delete-case="${escape(c.id)}" title="删除此 Case" aria-label="删除 Case ${escape(c.taskId)}" data-copy-lock>${icon('trash')}</button></div>`).join('') : `<div class="queue-empty">${state.cases.length ? '没有匹配的 Case' : '尚无 Case<br><span>导入文件，开始第一份标注。</span>'}</div>`}</nav>
    <div class="sidebar-foot">${button(icon('book') + '标注教程', 'help', 'button ghost full')}
    <div class="privacy">${icon('lock')}<div>LOCAL-FIRST<span>数据保存在当前浏览器</span></div></div></div></aside>`;
}
function renderShell() {
  if (renderedCase === state.active && current()) {
    state.evidenceScroll = $('#evidence-content')?.scrollTop ?? 0;
    state.formScroll = $('#form-content')?.scrollTop ?? 0;
  }
  const c = current();
  document.body.classList.toggle('ai-review-mode',aiReadOnly());
  $('#app').innerHTML = `${renderSidebar()}<main class="main"><header class="topbar"><div class="breadcrumb">${sharedMode?escape(sharedRun?.number || '共享工作区'):'研究工具'} <span>/</span> Case Study <span>/</span> <strong>${c ? escape(c.taskId) : '工作台'}</strong></div><div class="top-actions"><span class="save-state"><i></i><span id="save-label">${sharedMode?'已连接服务器':db ? '已保存到此浏览器' : '仅临时保存 · 请导出备份'}</span></span>${sharedMode?'':button('使用示例', 'demo', 'button subtle')+button(icon('upload') + '导入文件', 'import', 'button')}<details class="export-menu"><summary class="button">${icon('download')}导出</summary><div>${button('工作包 JSON · 含证据与标注', 'export-json', 'menu-item')}${button('当前 Case · Markdown', 'export-md', 'menu-item')}${button('所有 Case · CSV', 'export-csv', 'menu-item')}</div></details></div></header>
    <div class="content">${sharedMode&&c?renderSharedBanner(c):''}${startupError?`<section class="notice amber"><h2>无法打开任务</h2><p>${escape(startupError)}</p><a class="button" href="./">返回运行中心填写昵称或重试</a></section>`:c ? renderWorkspace(c) : renderEmpty()}</div><footer class="footer"><span>BCG / CASE STUDY STUDIO</span><span>轨迹阅读 · 证据引用 · ${aiReadOnly()?'AI 标注':'人工标注'}</span></footer></main>`;
  if (saveError) saveLabel('未保存 · 请导出备份', true);
  else if (pending.size || saving) saveLabel('正在保存…');
  document.querySelectorAll('[data-action="demo"]').forEach(el => { el.textContent = '真实任务 Demo'; });
  document.querySelectorAll('.welcome [data-action="guide-demo"]').forEach(el => { el.textContent = '打开真实 Demo'; });
  const editorLink=document.createElement('a');
  editorLink.href='./copy-editor.html';editorLink.target='_blank';editorLink.rel='noopener';
  editorLink.textContent='文案编辑';$('.footer').append(editorLink);
  if (c) {
    $('#evidence-content').scrollTop = state.evidenceScroll;
    $('#form-content').scrollTop = state.formScroll;
    lockReviewer();
  }
  renderedCase = state.active;
  scheduleViewSave();
}
function renderSharedBanner(c){
  if(c.remote.readOnly){
    const ai=c.remote.ai;
    return `<div class="shared-case-banner ai-review-banner"><div><strong>AI 标注 · 只读</strong><span>${escape(ai.model)} · ${escape(ai.reasoning_effort)} · LiteLLM${ai.usage?` · 标注 Token ${num(ai.usage.total_tokens)}`:''}</span></div><a class="button subtle" href="${escape(caseLink(remote.run,remote.benchmark,remote.task))}">人工标注</a></div>`;
  }
  const ready=sharedRun?.tasks.some(t=>t.benchmark===remote.benchmark&&t.id===remote.task&&canAnnotateTask(t)&&t.ai_review?.status==='completed');
  return `<div class="shared-case-banner"><span>标注者 ${escape(c.remote.user.name)} · 本人的标注独立保存，其他人不会覆盖。</span><div class="shared-review-links">${ready?`<a class="button subtle" href="${escape(caseLink(remote.run,remote.benchmark,remote.task,'ai'))}">查看 AI 标注</a>`:''}${button('加载服务器版本','reload-shared','button subtle')}</div></div>`;
}
function renderEmpty() {
  return `<section class="welcome"><div class="welcome-copy"><span class="eyebrow">CASE ANNOTATION</span><h1>BCG Case<br>标注工作台</h1><p>导入一道题的运行记录，阅读轨迹和 Graph，<br class="desktop">记录 Agent 的行为、任务结果和改进方向。</p><div class="welcome-actions">${button(icon('book') + '开始标注教程', 'help', 'button primary large')}${button('用示例练习', 'guide-demo', 'button large')}</div><p class="welcome-hint">首次使用从教程开始；已有 Case 文件可在下方直接导入。</p></div><div class="welcome-visual" aria-hidden="true"><span class="visual-caption">CASE REVIEW WORKFLOW</span><div class="mini-node n1"><small>01 / EVIDENCE</small>原始记录</div><div class="mini-line l1"></div><div class="mini-node n2"><small>02 / BELIEF</small>节点与关系<span class="mini-confidence">belief · relation · confidence</span></div><div class="mini-line l2"></div><div class="mini-node n3"><small>03 / ANNOTATION</small>行为与结果标注</div><span class="visual-note">标注流程</span></div></section>
    <section class="import-zone" id="drop-zone"><div class="upload-icon">${icon('upload')}</div><div><h2>拖入 Case 运行包，一次导入多个任务</h2><p>自动打包的 JSON.GZ 无需解压 · 可多选运行包或选择目录 · 兼容单题散文件</p></div><div class="import-zone-actions">${button('选择文件', 'import', 'button')}${button(icon('folder') + '选择目录', 'folder', 'button subtle')}</div></section>
    <section class="workflow-strip"><div><span>01</span><h3>填写基本信息</h3><p>阅读任务和报告，记录任务结果</p></div><div><span>02</span><h3>引用关键过程</h3><p>沿轨迹记录行为、依据和影响</p></div><div><span>03</span><h3>选择并评价维度</h3><p>点亮关键过程中体现的维度，再评价利用与效果</p></div><div><span>04</span><h3>总结并导出</h3><p>完成归因总结，导出标注工作包</p></div></section>
    <div class="empty-bottom"><span>${icon('lock')}浏览器本地解析 · 自动保存 · 导出分享</span><span>支持 Deep Research Agent / BCG 实验记录</span></div>`;
}
function renderWorkspace(c) {
  const p = completion(c), t = tokenMetrics(c);
  return `<section class="case-heading"><div><div class="case-kicker"><span class="eyebrow">CASE REVIEW</span><span class="badge ${c.demoKind === 'real' ? 'green' : c.demo ? 'amber' : 'neutral'}">${escape(demoLabel(c))}</span><span id="review-status" class="badge ${c.annotation.status === 'reviewed' ? 'green' : 'neutral'}">${c.annotation.status === 'reviewed' ? '已完成' : '标注中'}</span></div><h1>${escape(c.title)}</h1><div class="case-meta"><span>${icon('file')}${c.files.length} 个${c.remote?.deferredEvidence?'已加载的':''}来源文件</span><span>${num(c.sources.reduce((s, v) => s + v.records.length, 0))} 条记录</span><span>${c.remote?.deferredEvidence?'完整 Graph 按需加载':`${c.graphs.length} 份 Graph`}</span><span>${c.contexts.length} 条 Context / 请求</span></div></div><div class="heading-actions">${sharedMode?'':button(icon('plus') + '补充附件', 'attach', 'button')}${button(icon('check') + '完成标注', 'complete', 'button primary')}</div></section>
    ${c.warnings.length ? `<details class="notice amber"><summary>记录提示 · ${c.warnings.length}</summary><ul data-copy-lock>${c.warnings.map(w=>`<li>${escape(w)}</li>`).join('')}</ul></details>` : ''}
    ${renderDemoOverview(c)}
    <section class="summary-bar"><div><small>AGENT TOKENS</small><strong>${num(t.agent)}</strong></div><div title="${escape(t.graphScope)}"><small>GRAPH TOKENS <span>ⓘ</span></small><strong>${num(t.graph)}</strong></div><div><small>AGENT + GRAPH</small><strong>${num(t.combined)}</strong></div><div><small>关键过程</small><strong><b class="event-total" data-event-count data-copy-lock>${c.annotation.events.length}</b><span>个</span></strong></div><div class="progress-cell"><small>标注进度 <span id="review-progress">${dimensionProgress(c)}</span></small><progress id="progress-meter" value="${p.percent}" max="100"></progress></div></section>
    <div class="workbench"><section class="panel evidence-panel"><div class="panel-heading"><div><span class="section-number">01</span><h2>轨迹</h2></div></div><nav class="tabs" aria-label="证据视图">${[['trajectory','运行轨迹'], ['context','每轮 Belief 注入情况'], ['graph','Belief Graph'], ['report','任务和结果']].map(([id, label]) => `<button data-tab="${id}" class="${state.tab === id ? 'selected' : ''}" aria-pressed="${state.tab === id}">${label}</button>`).join('')}</nav><div id="evidence-content" class="panel-body">${renderEvidence(c)}</div></section>
    <section class="panel annotation-panel"><div class="panel-heading"><div><span class="section-number">02</span><h2>分析</h2></div></div>${renderReviewRoute(c)}<div id="form-content" class="panel-body form-body">${renderForm(c)}</div></section></div>`;
}
function renderReviewRoute(c) {
  return `<section class="review-route" aria-label="标注操作步骤"><div class="review-route-header"><strong>标注步骤</strong>${button(icon('book') + '标注教程', 'help', 'button ghost')}</div><nav aria-label="标注表单">${annotationStages.map(([stage, label], i) => `<button data-stage="${stage}" data-form="${stage}" class="${state.form === stage ? 'selected' : ''}" aria-pressed="${state.form === stage}" ${state.form === stage ? 'aria-current="step"' : ''}><span class="step-number">${i + 1}</span><span class="step-label">${label}</span></button>`).join('')}</nav></section>`;
}
function renderDemoOverview(c) {
  if(c.demoKind!=='real')return '';
  const p=c.demoProvenance||{};
  return `<section class="demo-overview"><div><span class="eyebrow">REAL TASK / BCG DEEP RESEARCH AGENT</span><p>${escape(p.runDate||'')} · Agent ${escape(c.model)} / low · Graph ${escape(p.graphModel||'')} / none · 最近 2 轮</p></div><nav aria-label="真实 Demo 阅读入口">${button('阅读任务与报告','demo-report','button')}${button('查看 Graph 注入','demo-context','button')}${button('查看最终 Graph','demo-graph','button')}</nav></section>`;
}
function blank(title, description, action = '') { return `<div class="blank-state">${icon('file')}<h3>${title}</h3><p>${description}</p>${action}</div>`; }
const refButton = (ref, label = '引用为关键过程') => button('＋ ' + label, 'cite', 'button cite-button', `data-ref="${escape(ref)}"`);
function renderEvidence(c) {
  if (state.tab === 'trajectory') return renderTrajectory(c);
  if (state.tab === 'context') return renderContext(c);
  if (state.tab === 'graph') return renderGraph(c);
  return renderReport(c);
}
function renderToolCall(call, index) {
  const args = call.arguments;
  const objectArgs = args !== null && typeof args === 'object' && !Array.isArray(args);
  const value = x => typeof x === 'string' ? x : pretty(x);
  const parameters = objectArgs
    ? Object.entries(args).map(([key, arg]) => `<div><dt>${escape(key)}</dt><dd><pre>${escape(value(arg))}</pre></dd></div>`).join('')
    : args === undefined ? '' : `<div><dt>参数</dt><dd><pre>${escape(value(args))}</pre></dd></div>`;
  return `<section class="tool-call-card"><header><span class="tool-call-label">工具调用 <span data-copy-lock>${index + 1}</span></span><code data-copy-lock>${escape(call.name)}</code></header>${parameters ? `<dl class="tool-call-args" data-copy-lock>${parameters}</dl>` : '<p class="tool-call-empty">无参数</p>'}</section>`;
}
function renderTrajectory(c) {
  const agents = agentTrajectories(c);
  if (!agents.length) return blank('尚无 Agent 运行轨迹', '补充 Agent 执行日志或包含对话历史的 session state。Belief 注入、Graph 和报告可在对应视图中查看。', button('补充运行轨迹', 'attach', 'button'));
  const agent = agents.find(a => a.id === state.source) ?? agents[0]; state.source = agent.id;
  const filtered = agent.records.map(recordPresentation).filter(r => (!state.filter || r.role === state.filter) && `${r.text} ${r.thinking} ${r.tools}`.toLowerCase().includes(state.query.toLowerCase()));
  const pageSize = 24, maxPage = Math.max(0, Math.ceil(filtered.length / pageSize) - 1); state.page = Math.min(state.page, maxPage);
  const roles = [...new Set(agent.records.map(r => r.role))];
  const selector = agents.length > 1 ? `<label class="select-label">选择 Agent<select id="agent-select">${agents.map(a => `<option value="${escape(a.id)}" ${agent.id === a.id ? 'selected' : ''}>${escape(a.name)}</option>`).join('')}</select></label>` : '';
  return `<div class="evidence-toolbar">${selector}<div class="search-row"><label class="search-input">${icon('search')}<input id="trace-search" aria-label="搜索轨迹" placeholder="搜索 query、内容或工具名" value="${escape(state.query)}"></label><select id="role-select" aria-label="按角色筛选"><option value="">所有角色</option>${roles.map(r => `<option ${state.filter === r ? 'selected' : ''}>${escape(r)}</option>`).join('')}</select></div></div>
  <div class="timeline">${filtered.slice(state.page * pageSize, (state.page + 1) * pageSize).map(r => `<article class="record"><div class="record-header"><span class="role ${['user','assistant','tool'].includes(r.role) ? r.role : 'other'}">${escape(r.title)}</span><span class="mono">${r.call !== undefined ? 'call ' + escape(r.call) + ' · ' : ''}记录 ${r.index}</span></div>${r.thinking ? `<details class="thinking"><summary>Thinking / Reasoning</summary><pre>${escape(r.thinking)}</pre></details>` : ''}${r.text ? `<pre class="record-text">${escape(r.text)}</pre>` : ''}${r.toolCalls.length ? `<div class="tool-call-list">${r.toolCalls.map(renderToolCall).join('')}</div>` : ''}${!r.text && !r.thinking && !r.toolCalls.length ? '<p class="record-empty">本轮未记录文字或工具调用。</p>' : ''}<div class="record-footer">${refButton('record:' + r.id)}<details class="raw-record"><summary>原始 JSON</summary><pre>${escape(pretty(r.raw))}</pre></details></div></article>`).join('') || blank('没有匹配的记录', '试试其他关键词或角色。')}</div><div class="pagination">${button('← 上一页', 'prev-page', 'button subtle', state.page === 0 ? 'disabled' : '')}<span>${state.page + 1} / ${maxPage + 1} 页 · ${filtered.length} 条记录</span>${button('下一页 →', 'next-page', 'button subtle', state.page >= maxPage ? 'disabled' : '')}</div>`;
}
function renderContext(c) {
  if (!c.contexts.length) return blank('尚未导入 Belief 注入记录', '导入 requests.jsonl、graph-contexts 或 Model I/O，查看每轮的注入内容。', button('补充附件', 'attach', 'button'));
  const ctx = c.contexts.find(x => x.id === state.context) ?? c.contexts[0]; state.context = ctx.id;
  const index = c.contexts.indexOf(ctx), turn = contextTurn(c, ctx), beliefText = contextBeliefText(ctx);
  const emptyText = ctx.kind === 'injection_audit' && ctx.raw?.graph_chars === 0 && !ctx.fallback
    ? '本轮未注入 Belief。' : '本次记录未包含 Graph 注入文本。';
  const outputNotice = {
    'missing-call': '这份记录没有 Agent 调用编号，暂不能对应到当轮输出。Graph turn 与 Agent 调用编号分开计算。',
    'missing-trajectory': '补充同一次运行的 Agent 轨迹，即可查看对应输出。',
    'unmatched-agent': '尚未找到这份注入记录所属 Agent 的轨迹。',
    'ambiguous-agent': '多个 Agent 使用相同的调用编号，现有记录未能确定所属 Agent。',
    'missing-output': '轨迹中尚未记录这一轮的 Assistant 输出。',
    'ambiguous-call': '同一调用编号对应多条 Assistant 记录，暂不能确定唯一输出。',
  }[turn.status] || '尚未找到对应输出。';
  return `<nav class="context-navigation" aria-label="Belief 注入轮次切换">
    ${button('← 上一轮', 'prev-context', 'button subtle', index === 0 ? 'disabled' : '')}
    <div class="context-current" role="status" aria-live="polite" aria-atomic="true"><strong id="context-label" data-copy-lock>${escape(ctx.label)}</strong><span id="context-position">${index + 1} / ${c.contexts.length} 条记录</span></div>
    ${button('下一轮 →', 'next-context', 'button subtle', index === c.contexts.length - 1 ? 'disabled' : '')}
  </nav>
  ${ctx.fallback ? `<div class="notice amber">本轮发生回退：<span data-copy-lock>${escape(ctx.fallback)}</span></div>` : ''}
  <div class="context-section-heading"><h3>本轮注入的 Belief</h3><span data-copy-lock>${beliefText !== null && ctx.nodeIds.length ? `${ctx.nodeIds.length} 个节点 · ` : ''}${num((beliefText ?? '').length)} 字符</span></div>
  <article class="record context-record">${beliefText === null ? `<p class="record-empty context-content">${emptyText}</p>` : `<div class="report-markdown context-content" data-copy-lock>${renderMarkdown(beliefText || '本轮 Graph 没有 Belief 内容。')}</div>`}</article>
  <section class="context-output" aria-label="当轮 Agent 输出" data-call="${escape(turn.call ?? '')}">
    <div class="context-section-heading"><h3>本轮 Agent 输出</h3>${turn.agent ? `<span data-copy-lock>${escape(turn.agent.name)} · call ${escape(turn.call)}</span>` : ''}</div>
    ${turn.assistant ? renderContextOutput(turn.assistant) : `<p class="notice context-output-missing">${outputNotice}</p>`}
    ${turn.results.length ? `<details class="context-tool-results"><summary>本轮工具结果 <span data-copy-lock>· ${turn.results.length} 条</span></summary>${turn.results.map(renderContextOutput).join('')}</details>` : ''}
  </section>`;
}
function renderContextOutput(record) {
  const r = recordPresentation(record);
  return `<article class="record context-output-record" data-record-id="${escape(r.id)}" data-call="${escape(r.call ?? '')}">
    <div class="record-header"><span class="role ${r.role === 'assistant' ? 'assistant' : 'tool'}">${escape(r.title)}</span><span class="mono">记录 ${r.index}</span></div>
    ${r.thinking ? `<details class="thinking" open><summary>Thinking / Reasoning</summary><div class="report-markdown" data-copy-lock>${renderMarkdown(r.thinking)}</div></details>` : ''}
    ${r.text ? `<div class="report-markdown context-output-text" data-copy-lock>${renderMarkdown(r.text)}</div>` : ''}
    ${r.toolCalls.length ? `<div class="tool-call-list">${r.toolCalls.map(renderToolCall).join('')}</div>` : ''}
    ${!r.text && !r.thinking && !r.toolCalls.length ? '<p class="record-empty">本轮未记录文字或工具调用。</p>' : ''}
    <details class="raw-record"><summary>查看原始输出记录</summary><pre>${escape(pretty(r.raw))}</pre></details>
  </article>`;
}
function renderGraph(c) {
  if(c.remote?.deferredEvidence){
    const loading=c.remote.graphLoading;
    return `<div class="deferred-graph" role="status" aria-live="polite" aria-busy="${!!loading}">${loading?'<span class="evidence-loading-spinner" aria-hidden="true"></span>':''}${blank(loading?'正在加载完整 Graph…':'完整 Graph',
      c.remote.graphError||'轨迹和每轮 Belief 已加载。完整图和历史快照在查看时加载。',loading?'':button(c.remote.graphError?'重新加载':'加载完整 Graph','load-full-evidence','button'))}</div>`;
  }
  if (!c.graphs.length) return blank('尚未导入 Graph', '可上传 BCG 的 state.json、result.json 或含 nodes / relations 的 Graph 快照。', button('补充 Graph 文件', 'attach', 'button'));
  const views = graphViews(c), turns = state.graphView === 'turns';
  const graphs = views[state.graphView], selection = turns ? 'graphTurn' : 'graphFull';
  const switcher = `<div class="graph-view-switch" role="group" aria-label="Graph 展示范围">${[['full','全部 Graph'],['turns','每轮 Graph']].map(([view,label]) => `<button data-graph-view="${view}" aria-pressed="${state.graphView === view}">${label}<span>${views[view].length}</span></button>`).join('')}</div>`;
  if (!graphs.length) return switcher + blank(turns ? '尚未导入每轮 Graph' : '尚未导入全部 Graph', turns ? '补充 belief_graph.jsonl 等历史快照文件，查看各轮建图结果。' : '补充 state.json 或最终 Graph 快照，查看完整图。', button('补充 Graph 文件', 'attach', 'button'));
  const graph = graphs.find(g => g.id === state[selection]) ?? graphs[0]; state[selection] = graph.id;
  const index = graphs.indexOf(graph);
  const turn = / · Graph turn (-?\d+)$/.exec(graph.label)?.[1];
  const label = !turns ? '完整 Graph' : turn === undefined ? '历史快照' : `Graph turn ${turn}`;
  const query = state.query.toLowerCase();
  const matching = graph.nodes.filter(n => `${n.id} ${n.belief ?? n.decision ?? n.content ?? ''}`.toLowerCase().includes(query));
  const node = graph.nodes.find(n => String(n.id) === state.node) ?? matching[0];
  const adjacent = node ? graph.relations.filter(e => e.from_id === node.id || e.to_id === node.id) : [];
  const position = turns ? `${index + 1} / ${graphs.length} 份快照` : graphs.length > 1 ? `${index + 1} / ${graphs.length} 份完整 Graph` : '完整 Graph';
  const currentLabel = `<div class="graph-current" role="status" aria-live="polite" aria-atomic="true"><strong id="graph-label" data-copy-lock>${escape(label)}</strong><span id="graph-position">${position}</span></div>`;
  const navigation = turns || graphs.length > 1 ? `<nav class="graph-navigation" aria-label="Belief Graph 快照切换">
    ${button('← 上一份', 'prev-graph', 'button subtle', index === 0 ? 'disabled' : '')}
    ${currentLabel}
    ${button('下一份 →', 'next-graph', 'button subtle', index === graphs.length - 1 ? 'disabled' : '')}
  </nav>` : '';
  return `${switcher}${navigation}
  <div class="notice">${turns ? '该轮建图完成时的 Graph 快照，包含截至该轮的节点和关系。' : '完整 Graph 中的全部节点和关系，与每轮快照分开展示。'}实际注入 Agent 的内容见「每轮 Belief 注入情况」。</div><div class="graph-stats"><strong>${graph.nodes.length}<span>节点</span></strong><strong>${graph.relations.length}<span>关系</span></strong><span class="mono">${escape(graph.graphId)}</span></div><label class="search-input">${icon('search')}<input id="graph-search" aria-label="搜索 Graph 节点" value="${escape(state.query)}" placeholder="搜索 belief 内容或 ID"></label>
  ${node ? `<div class="graph-neighborhood">${graphSVG(node, adjacent, graph)}<span class="graph-caption">当前节点的一跳关系 · 点击相邻节点查看</span></div><article class="belief-detail"><div class="record-header"><span class="badge green">B${escape(node.id)}</span><span class="mono">confidence ${escape(node.confidence ?? '未知')}</span></div><p>${escape(node.belief ?? node.decision ?? node.content)}</p><div class="small muted">来源：${escape(pretty(node.source ?? {}))} · stance：${escape(node.stance ?? '未知')}</div><div class="relation-list">${adjacent.map(e => `<div><span class="mono">B${escape(e.from_id)}</span><span class="edge-type ${e.type === 'contradicts' ? 'contradiction' : ''}">${escape(e.type)}</span><span class="mono">B${escape(e.to_id)}</span><p>${escape(e.note ?? '')}</p></div>`).join('') || '<p class="muted small">此节点在上传快照中没有关系。</p>'}</div><details class="raw-record"><summary>节点原始属性</summary><pre>${escape(pretty(node))}</pre></details></article>` : graph.nodes.length ? blank('没有匹配的节点', '请调整关键词。') : blank('本份 Graph 暂无节点', '该快照记录的节点数量为 0。')}
  <div class="node-list-heading">所有匹配节点 <span>${matching.length}</span></div><div class="node-list">${matching.slice(0, 300).map(n => `<button data-node="${escape(n.id)}" class="node-row ${n.id === node?.id ? 'selected' : ''}"><span class="mono">B${escape(n.id)}</span><span>${escape(n.belief ?? n.decision ?? n.content)}</span></button>`).join('')}</div>${matching.length > 300 ? '<p class="muted small">仅列出前 300 个节点，请用搜索缩小范围。</p>' : ''}`;
}
function graphSVG(node, edges, graph) {
  const neighbors = [...new Set(edges.map(e => e.from_id === node.id ? e.to_id : e.from_id))].filter(id => id !== node.id && graph.nodes.some(n => n.id === id)).slice(0, 8);
  const points = new Map([[node.id, [260, 112]]]);
  neighbors.forEach((id, i) => { const a = (i / Math.max(neighbors.length, 1)) * Math.PI * 2 - Math.PI / 2; points.set(id, [260 + Math.cos(a) * 190, 112 + Math.sin(a) * 77]); });
  return `<svg viewBox="0 0 520 224" role="img" aria-label="B${escape(node.id)} 的一跳关系图"><defs><marker id="edge-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="var(--line-strong)"/></marker></defs>${edges.filter(e => points.has(e.from_id) && points.has(e.to_id) && e.from_id !== e.to_id).map(e => {const a = points.get(e.from_id), b = points.get(e.to_id), dx = b[0]-a[0], dy = b[1]-a[1], len = Math.hypot(dx,dy); return `<line x1="${a[0] + dx/len*23}" y1="${a[1]+dy/len*23}" x2="${b[0]-dx/len*26}" y2="${b[1]-dy/len*26}" class="graph-edge ${e.type === 'contradicts' ? 'conflict-edge' : ''}" marker-end="url(#edge-arrow)"><title>B${escape(e.from_id)} ${escape(e.type)} B${escape(e.to_id)}</title></line>`;}).join('')}${[...points.entries()].map(([id,[x,y]]) => `<g class="svg-node ${id === node.id ? 'center' : ''}" data-node="${escape(id)}" role="button" tabindex="0" aria-label="查看节点 B${escape(id)}"><circle cx="${x}" cy="${y}" r="23"/><text x="${x}" y="${y+4}" text-anchor="middle">B${escape(id)}</text></g>`).join('')}</svg>`;
}
function renderReport(c) {
  return `<section class="task-result"><h3 class="subheading">任务原文</h3><pre class="plain-text">${escape(c.question || '任务文件中未提供，可在基本信息中填写分析目标。')}</pre><h3 class="subheading">最终结果</h3>${c.report ? `<article class="record report-record"><div class="report-markdown" data-copy-lock>${renderMarkdown(c.report)}</div></article>` : blank('尚无最终结果', '导入包含最终回答或报告的运行结果。')}</section>`;
}

function field(label, path, value, {type = 'textarea', options, placeholder = '', rows = 3, hint = '', help = ''} = {}) {
  if(aiReadOnly())return `<div class="field ai-review-field"><span>${label}</span><div class="ai-review-value" data-copy-lock>${escape(value||'—')}</div></div>`;
  const id = 'field-' + path.replace(/\./g, '-');
  const attrs = `id="${id}" data-field="${escape(path)}"${help ? ` aria-describedby="${id}-help"` : ''}`;
  const control = options ? `<select ${attrs}><option value="">请选择…</option>${options.map(o => `<option ${value === o ? 'selected' : ''}>${escape(o)}</option>`).join('')}</select>` : type === 'input' ? `<input ${attrs} value="${escape(value)}" placeholder="${escape(placeholder)}">` : `<textarea ${attrs} rows="${rows}" placeholder="${escape(placeholder)}">${escape(value)}</textarea>`;
  if (help) return `<div class="field field-with-help"><span class="field-label"><label for="${id}">${label}</label><span class="field-help"><button type="button" class="field-help-button" aria-label="${escape(label)}的说明" aria-describedby="${id}-help"><span aria-hidden="true" data-copy-lock>!</span></button><span class="field-tooltip" id="${id}-help" role="tooltip">${escape(help)}</span></span></span>${control}${hint ? `<small>${hint}</small>` : ''}</div>`;
  return `<label class="field" for="${id}"><span>${label}</span>${control}${hint ? `<small>${hint}</small>` : ''}</label>`;
}
function renderForm(c) {
  if (state.form === 'basic') return renderBasic(c);
  if (state.form === 'dimensions') return renderDimensions(c);
  if (state.form === 'events') return renderEvents(c);
  return renderConclusion(c);
}
function renderConclusion(c) {
  const a = c.annotation, conclusion = a.conclusion, observations = conclusionObservations(c);
  if(aiReadOnly()) return `<div class="form-intro compact"><h3>重点与改进方向</h3></div>
    <section class="conclusion-section">
      ${field('主要作用','conclusion.benefit',conclusion.benefit)}
      ${field('主要问题','conclusion.problem',conclusion.problem)}
      ${field('优先改进','conclusion.improvement',conclusion.improvement)}
      ${field('验证方式','conclusion.verification',conclusion.verification)}
    </section>
    <details class="conclusion-review"><summary>关联关键过程</summary><div class="observation-processes">${a.events.map(e=>`<button class="observation-process" data-open-event="${escape(e.id)}">${escape(eventLabel(c,e))}</button>`).join('')}</div></details>
    <details class="conclusion-comparison"><summary>证据强度与归因限制</summary>${field('证据强度','conclusion.evidence',conclusion.evidence)}${field('其他影响因素','conclusion.comparison',conclusion.comparison)}</details>`;
  const processLink = e => `<button type="button" class="observation-process" data-open-event="${escape(e.id)}"><span data-copy-lock>${escape(eventLabel(c, e))}</span>${e.fault ? `<small data-copy-lock>归因：${escape(e.fault)}</small>` : ''}</button>`;
  return `<div class="form-intro"><span class="eyebrow">OBSERVATION → ATTRIBUTION → IMPROVEMENT</span><h3>从关键过程形成结论</h3><p>结合已记录的过程和评价，总结 BCG 的实际作用，定位问题并提出下一步改进。</p></div>
    <details class="conclusion-review" open><summary><span class="conclusion-number">01</span><strong>回顾已记录的评价</strong><span class="conclusion-count" data-copy-lock>${observations.length} 个维度</span></summary>
      <div class="conclusion-review-body"><p class="conclusion-hint">以下内容随关键过程和维度评价更新。点击维度或过程可返回查看。</p>
        ${observations.length ? observations.map(d => `<article class="observation-card" data-observation="${d.id}">
          <header><button type="button" class="observation-dimension" data-open-dimension="${d.id}"><span class="dimension-id">${d.id}</span><span ${d.custom ? 'data-copy-lock' : ''}>${escape(d.title)}</span>${d.pending ? '<small>待确认</small>' : ''}</button><span class="observation-effect ${d.effect === '正向' ? 'positive' : d.effect === '负向' ? 'negative' : ''}" data-copy-lock>${escape(d.effect || '效果未评')}</span></header>
          <p class="observation-rating" data-copy-lock>利用情况：${escape(d.utilization || '未填写')}</p>
          ${d.note ? `<p class="observation-note" data-copy-lock>${escape(d.note)}</p>` : '<p class="conclusion-hint">判断依据尚未填写。</p>'}
          <div class="observation-processes">${d.events.length ? d.events.map(processLink).join('') : '<span class="conclusion-hint">尚未关联关键过程。</span>'}</div>
        </article>`).join('') : `<p class="conclusion-empty">${hasNoApplicableDimensions(c) ? '已查看轨迹，未发现相关维度。可在下方记录观察结论和后续核查方向。' : '尚未选择维度。先根据关键过程完成维度评价。'}</p>
          ${a.events.length ? `<div class="observation-processes">${a.events.map(processLink).join('')}</div>` : ''}
          <button type="button" class="button subtle" data-stage="dimensions">查看维度评价</button>`}
      </div>
    </details>
    <section class="conclusion-section"><header><span class="conclusion-number">02</span><h3>归因判断</h3></header>
      <p class="conclusion-hint">主要作用和主要问题至少填写一项。引用过程编号，说明 Graph 内容、Agent 行为与结果之间的联系。</p>
      ${field('主要作用', 'conclusion.benefit', conclusion.benefit, {placeholder:'例如：E02 中 Agent 复用 B18 的排除结论，转而验证另一个候选，避免再次调查已排除的方向。',rows:3})}
      ${field('主要问题', 'conclusion.problem', conclusion.problem, {placeholder:'记录最重要的 1–2 个问题。例如：E03 中相关 belief 已注入，Agent 仍重复查询，结果没有增加有效证据。',rows:3})}
      ${field('结论的证据强度', 'conclusion.evidence', conclusion.evidence, {options:OPTIONS.evidence, hint:'明确可见：有直接行为与结果记录；有较强迹象：部分关联仍需核验；证据不足：暂时无法确定作用。'})}
      <details class="conclusion-comparison" ${conclusion.comparison.trim() ? 'open' : ''}><summary>对照与其他信息来源 <span>选填</span></summary>
        ${field('对照条件与其他影响因素', 'conclusion.comparison', conclusion.comparison, {placeholder:'记录同题对照，以及 recent context、原始任务或其他工具是否也提供了相同信息。',rows:3})}
      </details>
    </section>
    <section class="conclusion-section"><header><span class="conclusion-number">03</span><h3>下一步改进</h3><span class="conclusion-optional">选填</span></header>
      <p class="conclusion-hint">围绕已定位的问题提出改法。有效做法也可以记录在这里，作为后续保留和验证的方向。</p>
      <fieldset class="improvement-stages"><legend>优先改进环节 <span>可多选</span></legend>
        <div class="improvement-stage-options">${IMPROVEMENT_STAGES.map(stage => `<button type="button" data-conclusion-focus="${escape(stage)}" aria-pressed="${(conclusion.focus || []).includes(stage)}">${stage}</button>`).join('')}</div>
      </fieldset>
      ${field('具体改进方案', 'conclusion.improvement', conclusion.improvement, {placeholder:'对应哪个问题，调整什么。例如：针对 E03，在注入内容中明确保留已排除的候选及原因。',rows:3})}
      ${field('验证方式与预期变化', 'conclusion.verification', conclusion.verification || '', {placeholder:'如何验证改动有效。例如：检查同类过程中是否减少重复查询，同时保留必要核验和答案正确性。',rows:3})}
    </section>
    <div id="conclusion-readiness" aria-live="polite">${renderConclusionReadiness(c)}</div>
    <div class="form-bottom">${button(icon('check')+'完成标注', 'complete', 'button primary')}${button(icon('download')+'导出 Markdown', 'export-md', 'button')}</div>`;
}
function renderConclusionReadiness(c) {
  const missing = validateReview(c);
  return missing.length ? `<details class="conclusion-checklist"><summary>还有 <span data-copy-lock>${missing.length}</span> 项待补充</summary><ul data-copy-lock>${missing.map(item => `<li>${escape(item)}</li>`).join('')}</ul></details>` : '<p class="conclusion-ready">必填内容已齐全，可以完成标注。</p>';
}
function renderDimensions(c) {
  const dimensions = selectedDimensions(c), ids = new Set(dimensions.map(d => d.id));
  const none = hasNoApplicableDimensions(c);
  if(aiReadOnly()) return `<div class="form-intro compact"><h3>维度评价</h3><p>证据与完整分析见关联的关键过程。</p></div>${dimensions.map(d=>{
    const r=c.annotation.dimensions[d.id];
    return `<section class="dimension-card ai-dimension-card"><div class="dimension-title"><span class="dimension-id">${d.id}</span><h3>${escape(d.title)}</h3></div>
      <p class="ai-dimension-verdict">${escape(r.utilization)} · ${escape(r.effect)}</p>
      <div class="observation-processes">${linkedEvents(c,d.id).map(e=>`<button class="observation-process" data-open-event="${escape(e.id)}">${escape(eventLabel(c,e))}</button>`).join('')}</div>
      ${r.note?`<details class="ai-dimension-note"><summary>评价补充</summary><p>${escape(r.note)}</p></details>`:''}</section>`;
  }).join('')||blank('未评估维度','本次没有证据充分的关键过程。')}`;
  return `<div class="form-intro"><span class="eyebrow">EVIDENCE → DIMENSIONS → EVALUATION</span><h3>评估 BCG 的实际作用</h3><p>先查看轨迹并记录关键过程，再选择这些过程中体现的维度。已复用和未复用 Graph 信息的行为都可以成为分析依据。</p></div>
    <fieldset class="dimension-picker"><legend>关键过程体现的维度</legend><p>点亮表示从关键过程中发现该维度存在影响可能，不依据任务类型预选。可多选。</p>
      <div class="dimension-options" role="group" aria-label="选择关键过程体现的维度">${allDimensions(c).map(d => `<button type="button" class="dimension-choice ${ids.has(d.id) ? 'selected' : ''} ${c.annotation.dimensions[d.id].opportunity === '无法判断' ? 'needs-confirmation' : ''}" data-toggle-dimension="${d.id}" aria-pressed="${ids.has(d.id)}" title="${escape(d.question || d.title)}" ${d.custom ? 'data-copy-lock' : ''}><span class="dimension-id">${d.id}</span><span class="dimension-choice-name">${escape(d.title)}</span><span class="dimension-choice-mark" aria-hidden="true">${icon('check')}</span></button>`).join('')}<button type="button" class="dimension-choice dimension-add" data-action="add-dimension" aria-label="添加自定义维度">${icon('plus')}<span>自定义维度</span></button></div>
      <div class="dimension-selection-status" role="status"><span data-copy-lock>${dimensions.length ? `已选择 ${dimensions.length} 个维度` : none ? '已确认未发现相关维度' : '尚未选择维度'}</span><span>下方仅展示已选维度</span></div>
      ${!dimensions.length ? `<label class="no-dimension-choice"><input type="checkbox" data-no-dimensions ${none ? 'checked' : ''}>已查看轨迹，未发现相关维度</label>` : ''}
    </fieldset>
    ${!dimensions.length ? blank(none ? '继续完成归因总结' : '从关键过程开始选择', none ? '在归因总结中记录这次观察的结论和依据。' : c.annotation.events.length ? `已记录 ${c.annotation.events.length} 个关键过程。请在上方点亮这些过程中体现的维度。` : '在运行轨迹中引用关键过程，记录 Agent 的实际行为及相关 Graph 内容，再选择维度。', `<button class="button subtle" data-stage="${none ? 'conclusion' : 'events'}">${none ? '填写归因总结' : c.annotation.events.length ? '查看关键过程' : '记录关键过程'}</button>`) : dimensions.map(d => {
      const r = c.annotation.dimensions[d.id];
      return `<section class="dimension-card" data-dimension="${d.id}"><div class="dimension-title"><span class="dimension-id">${d.id}</span><h3 ${d.custom ? 'data-copy-lock' : ''}>${escape(d.title)}</h3><span class="dimension-tag">${d.short}</span></div>${d.custom ? `<div class="dimension-manage">${button('编辑维度', 'edit-dimension', 'button ghost', `data-dimension-id="${d.id}"`)}${button(icon('trash')+'删除维度', 'delete-dimension', 'button ghost', `data-dimension-id="${d.id}"`)}</div>` : ''}${d.question ? `<p ${d.custom ? 'data-copy-lock' : ''}>${escape(d.question)}</p>` : ''}
        ${r.opportunity === '无法判断' ? `<div class="notice amber">此维度的旧标注为“无法判断”。结合关键过程后，点击上方该维度确认影响可能。</div>` : ''}
        ${renderProcessEvidence(c, d.id)}
        <div class="two-fields dimension-ratings">${field('是否利用', `dimensions.${d.id}.utilization`, r.utilization, {options:OPTIONS.utilization.filter(v => v !== '不适用'), help:'根据关联的关键过程判断 Agent 对相关 Graph 信息的使用程度，包括充分利用、部分利用、未利用或误用。'})}${field('是否产生效果', `dimensions.${d.id}.effect`, r.effect, {options:OPTIONS.effect.filter(v => v !== '不适用'), help:'根据后续行为和任务结果判断实际影响：正向、负向或无明显变化。影响尚不明确时选“待验证”。'})}</div>
        <div class="dimension-evidence">${field('判断依据', `dimensions.${d.id}.note`, r.note, {placeholder:'说明关联过程中的 Graph 内容、Agent 行为和后续影响如何支持本维度的评价。', rows:2})}</div></section>`;
    }).join('')}`;
}
function renderProcessEvidence(c, dimensionId) {
  const selected = new Set(linkedEvents(c, dimensionId).map(e => e.id));
  const legacy = c.annotation.dimensions[dimensionId].evidence;
  return `<fieldset class="process-evidence"><legend>证据来源 · 关键过程</legend><div class="process-evidence-hint"><span>选择体现此维度的关键过程，可多选。</span><span class="process-count" data-copy-lock>已关联 ${selected.size} 个</span></div>
    ${c.annotation.events.length ? `<div class="process-options">${c.annotation.events.map(e => `<div class="process-option ${selected.has(e.id) ? 'linked' : ''}"><label><input type="checkbox" data-link-dimension="${dimensionId}" data-link-event="${escape(e.id)}" ${selected.has(e.id) ? 'checked' : ''}><span data-copy-lock>${escape(eventLabel(c, e))}</span></label>${button('查看', 'view-process', 'button ghost', `data-open-event="${escape(e.id)}" aria-label="查看关键过程"`)}</div>`).join('')}</div>` : `<div class="process-empty"><p>先从运行轨迹引入关键过程，再将它关联到本维度。</p><button class="button subtle" data-stage="events">记录关键过程</button></div>`}
    ${legacy?.trim() ? `<details class="legacy-evidence"><summary>原有证据备注</summary><p data-copy-lock>${escape(legacy)}</p><small>原文已保留，关联关键过程后用于维度评价。</small></details>` : ''}
  </fieldset>`;
}
function renderProcessDimensions(c, event) {
  const dimensions = selectedDimensions(c).filter(d => linkedEvents(c, d.id).some(e => e.id === event.id));
  return `<div class="process-dimensions"><span>关联维度</span><div>${dimensions.length ? dimensions.map(d => `<button class="button subtle" data-open-dimension="${d.id}" ${d.custom ? 'data-copy-lock' : ''}>${d.id} ${escape(d.title)}</button>`).join('') : '<span class="small muted">尚未关联维度</span>'}</div></div>`;
}

function openDimensionEditor(id = '') {
  const c = current(); if (!c) return;
  const dimension = allDimensions(c).find(d => d.id === id && d.custom);
  if (id && !dimension) return;
  dimensionEditorCase = c.id; dimensionEditorId = id;
  $('#dimension-title').textContent = id ? '编辑自定义维度' : '添加自定义维度';
  $('#dimension-name').value = dimension?.title || '';
  $('#dimension-description').value = dimension?.question || '';
  $('#dimension-error').textContent = '';
  $('#dimension-submit').textContent = id ? '保存修改' : '添加维度';
  $('#dimension-dialog').showModal();
  $('#dimension-name').focus();
}

function focusDimension(id) {
  const panel = $('#form-content'), card = $(`[data-dimension="${id}"]`);
  if (!panel || !card) return;
  panel.scrollTop += card.getBoundingClientRect().top - panel.getBoundingClientRect().top - 16;
  card.querySelector('select')?.focus({preventScroll: true});
}
function renderBasic(c) {
  const a = c.annotation;
  return `<div class="form-intro"><span class="eyebrow">START WITH THE TASK</span><h3>明确任务目标与结果</h3><p>记录任务要求与完成情况。</p></div>
    <div class="two-fields">${field('标注人', 'reviewer', a.reviewer, {type:'input',placeholder:'姓名 / 编号'})}${field('Agent 形式', 'team', a.team, {options:['单 Agent','Agent Team','无法判断']})}</div>
    ${field('任务目标', 'goal', a.goal, {placeholder:c.question || '任务需要回答的问题和交付内容。'})}
    ${field(aiReadOnly()?'任务结果（AI 评估）':'任务结果', 'result', a.result, {options:OPTIONS.result})}${renderTaskAssessment(a)}`;
}
function renderTaskAssessment(annotation) {
  const assessment=annotation.resultAssessment;
  if(!assessment)return aiReadOnly()?'<p class="ai-task-assessment-legacy">这份旧版 AI 标注未评估任务结果。</p>':'';
  const quotes=(key,label)=>`<h5>${label}</h5>${(assessment[key]||[]).map(q=>`<blockquote>${escape(q)}</blockquote>`).join('')}`;
  return `<section class="ai-task-assessment" aria-label="AI 任务结果评估"><div class="ai-task-assessment-heading"><h4>AI 结果评估</h4><span>非官方评分</span></div><strong>${escape(assessment.result)}</strong><p class="ai-task-rationale">${escape(assessment.rationale)}</p><details><summary>查看判断依据原文</summary>${quotes('task_quotes','任务要求原文')}${quotes('report_quotes','最终报告原文')}</details>${assessment.limitations?.length?`<div class="ai-task-limitations"><h5>判断范围与待核实内容</h5><ul>${assessment.limitations.map(item=>`<li>${escape(item)}</li>`).join('')}</ul></div>`:''}</section>`;
}
function renderEventEvidence(c, e) {
  const view = eventEvidencePresentation(c, e), r = view.record, belief = eventBeliefContext(c, e);
  const markdown = text => `<div class="report-markdown">${renderMarkdown(text)}</div>`;
  const beliefNotice = {
    'missing-record': '未找到此关键过程的来源记录。',
    'not-agent-call': '此记录没有对应的 Agent 调用。',
    'missing-context': '未找到对应轮的 Belief 注入记录。',
    'ambiguous-context': '对应轮存在不同的注入记录，无法确定唯一版本。',
    'missing-graph': '本轮记录未包含 belief_graph 内容。',
    'empty': '本轮未注入 Belief。',
  }[belief.status];
  const content = r
    ? `${r.thinking ? `<details class="thinking" open><summary>Thinking / Reasoning</summary>${markdown(r.thinking)}</details>` : ''}${r.text ? markdown(r.text) : ''}${r.toolCalls.length ? `<div class="tool-call-list">${r.toolCalls.map(renderToolCall).join('')}</div>` : ''}${!r.text && !r.thinking && !r.toolCalls.length ? '<p class="record-empty">本轮未记录文字或工具调用。</p>' : ''}`
    : view.text ? markdown(view.text) : '<p class="record-empty">此历史过程未保存原文摘录。</p>';
  return `<details class="event-reference event-disclosure" ${aiReadOnly()?'':'open'}><summary>轨迹证据<span class="event-disclosure-note">${r ? '引用记录' : '引用摘录'}</span></summary>
    <section class="event-belief-context" aria-label="关键过程对应轮的 Belief" data-call="${escape(belief.call ?? '')}">
      <div class="event-evidence-heading"><h4>本轮注入的 Belief</h4><span data-copy-lock>${belief.agent ? escape(belief.agent.name) + ' · ' : ''}${belief.call !== null ? 'call ' + escape(belief.call) : ''}</span></div>
      ${belief.context?.fallback ? `<p class="notice amber" data-copy-lock>本轮发生回退：${escape(belief.context.fallback)}</p>` : ''}
      ${belief.status === 'matched' ? `<div class="report-markdown event-belief-content" data-copy-lock>${renderMarkdown(belief.text)}</div>` : `<p class="record-empty event-belief-missing">${beliefNotice}</p>`}
    </section>
    <div class="event-evidence-heading"><h4>${!r ? '引用的轨迹摘录' : r.role === 'assistant' ? '本轮 Agent 输出' : ['tool', 'function'].includes(r.role) ? '引用的工具结果' : '引用的轨迹记录'}</h4></div>
    ${r ? `<div class="record-header" data-copy-lock><span class="role ${['user','assistant','tool'].includes(r.role) ? r.role : 'other'}">${escape(r.title)}</span><span class="mono">${r.call !== undefined ? 'call ' + escape(r.call) + ' · ' : ''}记录 ${r.index}</span></div>` : ''}
    <div class="event-evidence-content" data-copy-lock>${content}</div>
    ${e.reportQuote?`<section class="event-report-quote"><div class="event-evidence-heading"><h4>最终报告原文节选</h4></div><div class="report-markdown" data-copy-lock>${renderMarkdown(e.reportQuote)}</div></section>`:''}
    <details class="event-source-details"><summary>来源与原始摘录</summary><p data-copy-lock>${escape(e.reference || '此历史过程未保存来源位置。')}</p>${e.quote ? `<pre class="event-quote" data-copy-lock>${escape(e.quote)}</pre>` : ''}</details>
  </details>`;
}
function renderEvents(c) {
  const events = c.annotation.events;
  const e = events.find(e => e.id === state.event) ?? (aiReadOnly()?events[0]:events.at(-1)); if (e) state.event = e.id;
  const intro = `<div class="form-intro compact"><div><span class="eyebrow">CRITICAL MOMENTS</span><h3>${aiReadOnly()?'预期作用与实际表现':'记录预期作用与实际差距'}</h3></div>${button('从运行轨迹引入', 'choose-event-source', 'button')}</div>`;
  if (!e) return intro + (aiReadOnly()?blank('未选出关键过程','本次 Review 未找到证据充分的过程，可在归因总结中查看分析说明。'):blank('还没有关键过程', '在运行轨迹中点击「引用为关键过程」，再记录 BCG 应发挥的作用与实际表现。'));
  const optional = [
    ['当前子问题', 'objective', '本轮正在解决的具体问题。'],
    ['原始证据与来源', 'original', '补充其他相关记录的位置与内容，无需重抄已保存的摘录。'],
    ['入图情况', 'ingestion', '需要定位建图问题时，记录 belief / relation ID 和缺失内容。'],
    ['当轮实际可见内容', 'visible', '需要区分未入选和未利用时，补充当轮注入情况。'],
    ['其他来源与影响因素', 'alternatives', '例如近期上下文仍有相同信息，或搜索额度、网页访问影响了结果。'],
  ];
  const filled = optional.filter(([, key]) => e[key]?.trim()).length;
  return `${intro}<div class="event-switcher">${events.map((v,i) => `<button data-event="${escape(v.id)}" class="${v.id === e.id ? 'active' : ''}" title="${escape(v.label || '关键过程')}">E${String(i+1).padStart(2,'0')} ${escape(v.label || '关键过程')}</button>`).join('')}</div>
    ${renderEventEvidence(c, e)}
    ${renderSearchAssessment(e)}
    <section class="event-core" aria-label="关键过程简要记录">
      ${eventField('BCG 应发挥的作用', e, 'need', {rows:2,placeholder:'例如：复用已找到的证据，避免再次搜索相同问题。'}, events)}
      ${eventField('实际发生的情况', e, 'behavior', {rows:2,placeholder:'例如：B71 已入图但未被注入，Agent 又搜索了相同内容。'}, events)}
      ${eventField(aiReadOnly()&&c.annotation.reviewFormat==='focused-v3'?'影响':'影响与改进线索', e, 'impact', {rows:2,placeholder:'例如：重复搜索未带来新证据。可优先召回已有结果；效果不明时注明待验证。'}, events)}
      <div class="two-fields">${eventField('归因环节', e, 'fault', {options:OPTIONS.fault}, events)}${eventField('证据强度', e, 'strength', {options:OPTIONS.evidence}, events)}</div>
    </section>
    ${renderRelatedOccurrences(e)}
    <details class="event-supplement event-disclosure"><summary>补充证据检查<span class="event-disclosure-note">可选</span><span class="event-detail-count" data-copy-lock>${filled ? `已填写 ${filled} 项` : ''}</span></summary><p class="event-detail-hint">只补充判断所需的信息。原有详细标注完整保留在这里。</p>${optional.map(([label, key, placeholder]) => eventField(label, e, key, {rows:2,placeholder}, events)).join('')}</details>
    <details class="event-meta event-disclosure"><summary>过程信息<span class="event-disclosure-note">可修改</span></summary>${eventField('过程名称 / 位置', e, 'label', {type:'input',placeholder:'名称已从轨迹生成，可改为一句话概括。'}, events)}${eventField('所属 Agent', e, 'agent', {type:'input',placeholder:'单 Agent / coordinator / job-id'}, events)}</details>
    ${renderProcessDimensions(c, e)}${button('删除此过程', 'delete-event', 'button danger')}`;
}
function renderSearchAssessment(e) {
  const s=e.searchAssessment;if(!s)return '';
  return `<details class="ai-search-assessment"><summary>搜索判断 · ${escape(s.verdict)}</summary><dl>${[
    ['当前信息需求',s.information_need],['Graph 是否已满足',s.graph_sufficiency],['核验目的',s.verification_reason],
    ['预期新增内容',s.expected_gain],['判断依据',s.basis],['实际新增证据',s.observed_gain],
  ].map(([label,value])=>`<dt>${label}</dt><dd>${escape(value)}</dd>`).join('')}</dl>
    ${s.result_quote?`<h5>当轮工具结果</h5><blockquote>${escape(s.result_quote)}</blockquote>`:''}
    ${s.unavailable_quote?`<h5>调用前已知的工具状态</h5><blockquote>${escape(s.unavailable_quote)}</blockquote>`:''}</details>`;
}
function renderRelatedOccurrences(e) {
  const items=e.relatedOccurrences||[];if(!items.length)return '';
  return `<details class="ai-related-occurrences"><summary>同类过程与关联轮次 · ${items.length}</summary><p>${escape(e.groupReason)}</p>${items.map(item=>`<details><summary>${escape(item.agent)} · call ${escape(item.call)}</summary><h5>本轮注入的 Belief</h5><div class="report-markdown">${renderMarkdown(item.graphEvidence.text)}</div><h5>本轮 Agent 输出</h5><div class="report-markdown">${renderMarkdown(item.quote)}</div></details>`).join('')}</details>`;
}
function eventField(label, e, key, opts, events) { return field(label, `events.${events.indexOf(e)}.${key}`, e[key], opts); }
function renderEvidenceOnly() { $('#evidence-content').innerHTML = renderEvidence(current()); lockReviewer(); scheduleViewSave(); }
function openEvidence(tab) {
  if (state.tab === tab) return;
  state.tab = tab;
  state.query = '';
  state.page = 0;
  document.querySelectorAll('.evidence-panel [data-tab]').forEach(el => {
    const selected = el.dataset.tab === tab;
    el.classList.toggle('selected', selected);
    el.setAttribute('aria-pressed', String(selected));
  });
  renderEvidenceOnly();
  $('#evidence-content').scrollTop = 0;
}
async function ensureFullEvidence() {
  const c=current();
  if(!c?.remote?.deferredEvidence)return c;
  if(fullEvidenceTask)return fullEvidenceTask;
  c.remote.graphLoading=true;c.remote.graphError='';
  if(state.tab==='graph')renderEvidenceOnly();
  fullEvidenceTask=(async()=>{
    try{
      const full=await loadFullSharedEvidence(c);
      // Parser-generated IDs are different in the full import. Keep the reader
      // on the same original file/record when swapping the immutable evidence.
      const context=c.contexts.find(v=>v.id===state.context);
      const source=c.sources.find(v=>v.id===state.source);
      const path=id=>c.files.find(f=>f.id===id)?.path;
      if(context){
        const file=full.files.find(f=>f.path===path(context.fileId));
        const next=full.contexts.find(v=>v.fileId===file?.id&&v.index===context.index&&v.call===context.call);
        state.context=next?.id||'';
      }
      if(source)state.source=full.sources.find(v=>v.label===source.label&&v.kind===source.kind)?.id||'';
      const index=state.cases.findIndex(v=>v.id===c.id);
      if(index>=0)state.cases[index]=full;
      if(current()?.id===c.id)renderShell();
      return full;
    }catch(error){
      c.remote.graphError=error.message;c.remote.graphLoading=false;
      if(current()?.id===c.id&&state.tab==='graph')renderEvidenceOnly();
      toast('完整 Graph 加载失败：'+error.message,true);
      return null;
    }finally{c.remote.graphLoading=false;fullEvidenceTask=null;}
  })();
  return fullEvidenceTask;
}
function renderFormOnly() {
  const c = current();
  $('#form-content').innerHTML = renderForm(c);
  document.querySelectorAll('.annotation-panel [data-form]').forEach(el => {
    const selected = el.dataset.form === state.form;
    el.classList.toggle('selected', selected);
    el.setAttribute('aria-pressed', String(selected));
    if (selected) el.setAttribute('aria-current', 'step');
    else el.removeAttribute('aria-current');
  });
  document.querySelectorAll('[data-event-count]').forEach(el => { el.textContent = c.annotation.events.length; });
  lockReviewer();
  scheduleViewSave();
}

function lockReviewer() {
  const field = $('[data-field="reviewer"]');
  if (sharedMode && field) field.readOnly = true;
  if(aiReadOnly()){
    document.querySelectorAll(reviewMutations).forEach(el=>{
      el.disabled=true;
      if(el.matches('[data-action]'))el.hidden=true;
    });
    $('.annotation-panel')?.classList.add('ai-readonly');
    const chip=$('#review-status');if(chip){chip.textContent='AI 标注';chip.className='badge green';}
    saveLabel('AI 标注 · 已保存到服务器');
  }
}

async function readUploads(list, attach = false) {
  if (sharedMode) return toast('共享任务已包含运行证据，无需导入文件。', true);
  if (state.busy) return;
  let accepted = [...list].filter(f => /\.(json(?:\.gz)?|jsonl|ndjson|md|txt)$/i.test(f.name) && !/(^|\/)(node_modules|\.git|\.venv)\//.test(f.webkitRelativePath || ''));
  // Selecting a run directory imports its generated bundles, not loose files
  // from many unrelated tasks. Exact duplicates are skipped after validation.
  const packed = accepted.filter(f => /^case-stud(?:y|ies)(?:-\d+)?\.json(?:\.gz)?$/i.test(f.name));
  if (!attach && packed.length && accepted.some(f=>f.webkitRelativePath)) {
    const batches=packed.filter(f=>/^case-studies(?:-\d+)?\.json(?:\.gz)?$/i.test(f.name));
    accepted=batches.length ? batches : packed;
  }
  if (!accepted.length) return toast('请选择 Case 运行包（JSON / JSON.GZ）或 JSONL、Markdown、TXT 记录。ZIP 请先解压。', true);
  if (accepted.length > 300 || accepted.some(f => f.size > 200 * 1024 * 1024) || accepted.reduce((s,f) => s+f.size,0) > 250*1024*1024) return toast('一次最多 300 个文件、单文件 200 MB、合计 250 MB。请选择单题目录，排除无关日志。', true);
  state.busy = true; toast('正在本地解析文件…');
  try {
    const files = [];
    let remaining=MAX_IMPORT_BYTES;
    for (const f of accepted) {
      const file=await readImportFile(f,remaining);
      remaining-=file.size; files.push(file);
    }
    const bundles=files.filter(f=>/\.json$/i.test(f.path)).map(f=>JSON.parse(f.text)).filter(p=>p?.format===BUNDLE_FORMAT);
    if (bundles.length) {
      if (attach) throw new Error('Case 运行包请通过「导入 Case」导入；「补充附件」用于当前 Case 的散文件。');
      if (bundles.length !== files.length) throw new Error('请分开导入 Case 运行包和散文件，避免把不同任务混在一起。');
      const imported=await importBundles(bundles,state.cases);
      if (imported.cases.length) {
        await flush();
        await enqueueWrite(()=>dbPutCases(imported.cases));
        state.cases.unshift(...imported.cases);
        activateCase(imported.cases[0].id); renderShell();
      }
      const warnings=imported.cases.reduce((n,c)=>n+c.warnings.length,0);
      toast(`已导入 ${imported.cases.length} 个 Case${imported.skipped ? `，跳过 ${imported.skipped} 个重复 Case` : ''}${imported.revisions ? '；新证据版本单独保存，原标注保留' : ''}${warnings ? `；${warnings} 条记录提示可在 Case 标题下查看` : ''}。`);
      return;
    }
    if (files.length === 1 && /\.json$/i.test(files[0].path)) {
      const payload = JSON.parse(files[0].text);
      if (payload?.format === 'bcg-case-study-workspace') {
        const restored = restoreWorkspace(payload);
        const overlap = restored.some(c => state.cases.some(old => old.id === c.id));
        if (overlap && !confirm('导入版本将替换已有 Case 的本地标注。点击“确定”继续，或取消后先导出备份。')) return;
        for (const c of restored) { const idx = state.cases.findIndex(v=>v.id===c.id); if(idx<0) state.cases.push(c); else state.cases[idx]=c; pending.set(c.id, structuredClone(c)); }
        activateCase(restored[0]?.id || state.active); await flush(); renderShell(); toast(`已恢复 ${restored.length} 个 Case。`); return;
      }
    }
    const c = ingestFiles(files, attach ? current() : null);
    if (attach) state.cases[state.cases.findIndex(v=>v.id===c.id)] = c; else state.cases.unshift(c);
    activateCase(c.id); dirty(c); await flush(); renderShell();
    toast(`已${attach ? '补充' : '导入'} ${files.length} 个文件${c.warnings.length ? '，有 ' + c.warnings.length + ' 条导入提醒' : ''}。`);
  } catch(e) { toast('导入失败：' + e.message, true); }
  finally {state.busy=false; $('#file-input').value=''; $('#folder-input').value='';}
}
function cite(ref) {
  const c=current(), e=eventFromTrajectory(c,ref,window.getSelection()?.toString().trim());
  if(!e)return;
  c.annotation.noApplicableDimensions=false;
  c.annotation.events.push(e);state.event=e.id;state.form='events';dirty();renderFormOnly();$('#form-content').scrollTop=0;toast('轨迹证据已保存，补充预期作用、实际情况与影响即可。');
}
function download(name, text, type) {
  const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
async function exportData(kind) {
  if(!state.cases.length)return toast('请先导入 Case。');
  // A workspace export must remain complete, even if only its preview was read.
  if(kind==='json'&&current()?.remote?.deferredEvidence){
    toast('正在准备包含完整 Graph 的工作包…');
    if(!await ensureFullEvidence())return;
  }
  const c=current(), safe=(c?.taskId || 'cases').replace(/[^\p{L}\p{N}._-]/gu,'_');
  if(kind==='json')download('bcg-case-study-workspace.json',pretty(workspaceExport(state.cases)),'application/json');
  if(kind==='md' && c)download(`case-study-${safe}.md`,markdownReport(c),'text/markdown;charset=utf-8');
  if(kind==='csv')download('bcg-case-study-summary.csv',csvSummary(state.cases),'text/csv;charset=utf-8');
  document.querySelectorAll('.export-menu[open]').forEach(e=>e.removeAttribute('open'));
}

async function demo() {
  if(state.busy)return false;
  await flush();
  const old=state.cases.find(c=>c.demoKind==='real'&&c.taskId===REAL_DEMO_ID);
  if(old){activateCase(old.id);renderShell();return true;}
  state.busy=true;toast('正在载入真实运行记录…');
  try {
    const response=await fetch('./demo-case.json', {cache:'no-store'});
    if(!response.ok)throw new Error('Demo 文件暂时无法读取，请稍后重试。');
    const c=await buildDemoCase(await response.json());
    state.cases.unshift(c);activateCase(c.id);dirty(c);await flush();renderShell();
    toast('已载入真实任务。标注留空，可从报告、轨迹和 Graph 开始分析。');
    return true;
  } catch(e) {toast(`Demo 加载失败：${e.message}`,true);return false;}
  finally {state.busy=false;}
}

async function deleteCase(id) {
  const index = state.cases.findIndex(c => c.id === id);
  if (index < 0 || state.busy) return;
  const c = state.cases[index];
  if (!confirm(`删除「${c.title}」？\nTask ID：${c.taskId}\n\n仅删除当前浏览器中的这条 Case、导入内容和标注，不影响原始文件。\n删除后无法直接撤销，如需保留请先导出工作包。`)) return;
  state.busy = true;
  deletingCases.add(id);
  pending.delete(id);
  document.querySelectorAll('.case-delete').forEach(el => { el.disabled = true; });
  try {
    // Older autosaves must finish before deletion; subsequent edits cannot requeue this case.
    await enqueueWrite(() => db ? dbOperation('delete', id) : Promise.resolve());
    pending.delete(id);
    state.cases = state.cases.filter(item => item.id !== id);
    if (!state.cases.length) state.sidebarSearch = '';
    if (state.active === id) {
      activateCase(state.cases[Math.min(index, state.cases.length - 1)]?.id || '');
      renderShell();
    } else {
      const list = $('.case-list'), top = list.scrollTop, left = list.scrollLeft;
      $('.sidebar').outerHTML = renderSidebar();
      $('.case-list').scrollTop = top;
      $('.case-list').scrollLeft = left;
      rememberView();
    }
    const next = [...document.querySelectorAll('.case-delete')][Math.min(index, state.cases.length - 1)];
    (next || $('.sidebar [data-action="import"]'))?.focus({preventScroll: true});
    toast('已删除此 Case，原始文件未改动。');
  } catch (error) {
    pending.set(id, c);
    saveError = error.message;
    saveLabel('删除失败 · 记录已保留', true);
    toast(`删除失败，记录和标注已保留：${error.message}`, true);
  } finally {
    deletingCases.delete(id);
    state.busy = false;
    document.querySelectorAll('.case-delete').forEach(el => { el.disabled = false; });
  }
}

document.addEventListener('click', async e => {
  const target=e.target.closest('button, [data-node]');if(!target || target.disabled)return;
  if(aiReadOnly() && target.matches(reviewMutations))return;
  if (deletingCases.size) return;
  if (target.dataset.deleteCase) { await deleteCase(target.dataset.deleteCase); return; }
  if (target.dataset.toggleDimension) {
    const c=current(), id=target.dataset.toggleDimension;
    if (!c || !allDimensions(c).some(d=>d.id===id)) return;
    const top=$('#form-content').scrollTop;
    setDimensionSelected(c,id,c.annotation.dimensions[id].opportunity!=='有');
    dirty(c);renderFormOnly();$('#form-content').scrollTop=top;
    $(`[data-toggle-dimension="${id}"]`)?.focus({preventScroll:true});return;
  }
  if (target.dataset.conclusionFocus) {
    const c=current(), stage=target.dataset.conclusionFocus;
    const selected=target.getAttribute('aria-pressed')!=='true';
    if (!c || !setConclusionFocus(c,stage,selected)) return;
    target.setAttribute('aria-pressed',String(selected));
    dirty(c);return;
  }
  if(target.dataset.guideStep !== undefined){showGuide(Number(target.dataset.guideStep));return;}
  if(target.dataset.guideOpen){openStage(target.dataset.guideOpen);return;}
  if(target.dataset.stage){openStage(target.dataset.stage);return;}
  if(target.dataset.case){await flush();activateCase(target.dataset.case);renderShell();return;}
  if(target.dataset.tab){openEvidence(target.dataset.tab);if(target.dataset.tab==='graph')await ensureFullEvidence();return;}
  if(target.dataset.form){openStage(target.dataset.form);return;}
  if(target.dataset.openEvent){
    if(!current()?.annotation.events.some(v=>v.id===target.dataset.openEvent))return;
    state.event=target.dataset.openEvent;state.form='events';renderFormOnly();$('#form-content').scrollTop=0;return;
  }
  if(target.dataset.openDimension){
    const id=target.dataset.openDimension;if(!current() || !allDimensions(current()).some(d=>d.id===id))return;
    state.form='dimensions';renderFormOnly();
    focusDimension(id);return;
  }
  if(target.dataset.event){state.event=target.dataset.event;renderFormOnly();return;}
  if(target.dataset.node!==undefined){state.node=target.dataset.node;renderEvidenceOnly();return;}
  if(['full','turns'].includes(target.dataset.graphView)){
    state.graphView=target.dataset.graphView;state.node='';state.query='';renderEvidenceOnly();$('#evidence-content').scrollTop=0;
    $(`[data-graph-view="${state.graphView}"]`)?.focus({preventScroll:true});return;
  }
  const action=target.dataset.action,c=current();
  if (action==='reload-shared' && c?.remote) {
    if (!confirm('加载服务器上的最新标注会替换本页草稿。若有未保存内容，请先导出工作包备份。')) return;
    clearTimeout(saveTimer);await writeQueue;pending.clear();
    if (db) await dbOperation('delete',c.id);
    location.reload();return;
  }
  if(sharedMode && ['import','attach','folder','guide-import','guide-demo','demo','delete'].includes(action)) return toast('共享任务无需导入，请从运行列表选择「人工标注」。');
  if(action==='add-dimension'){openDimensionEditor();return;}
  if(action==='edit-dimension'){openDimensionEditor(target.dataset.dimensionId);return;}
  if(action==='cancel-dimension'){$('#dimension-dialog').close();return;}
  if(action==='delete-dimension' && c){
    const id=target.dataset.dimensionId;
    if(!allDimensions(c).some(d=>d.id===id && d.custom))return;
    if(confirm('删除此自定义维度及其评价和关联。关键过程本身和其他维度保持不变。') && removeCustomDimension(c,id)){
      const top=$('#form-content').scrollTop;
      dirty(c);renderFormOnly();$('#form-content').scrollTop=top;
      $('[data-action="add-dimension"]')?.focus({preventScroll:true});
    }
    return;
  }
  if(action==='import'||action==='attach'){state.attach=action==='attach';$('#file-input').click();}
  if(action==='folder'){state.attach=false;$('#folder-input').click();}
  if(action==='help')showGuide();
  if(action==='guide-prev')showGuide(guideStep-1);
  if(action==='guide-next'){if(guideStep===4)$('#help-dialog').close();else showGuide(guideStep+1);}
  if(action==='guide-import'){$('#help-dialog').close();state.attach=false;$('#file-input').click();}
  if(action==='guide-demo'){if(await demo())openStage('basic');}
  if(action==='demo-report')openEvidence('report');
  if(action==='demo-context'){
    state.tab='context';
    state.context=(c.contexts.find(ctx=>ctx.nodeIds.length&&!ctx.fallback)||c.contexts[0])?.id||'';
    renderShell();
  }
  if(action==='demo-graph'){state.tab='graph';state.graphView='full';state.graphFull=(graphViews(c).full.find(g=>g.label==='agent/bcg/state.json')||graphViews(c).full[0])?.id||'';state.node='';state.query='';renderShell();}
  if(action==='demo')await demo();
  if(action==='cite' && state.tab==='trajectory' && target.closest('.timeline'))cite(target.dataset.ref);
  if(action==='choose-event-source')openEvidence('trajectory');
  if(action==='delete-event'){
    const linked=allDimensions(c).filter(d=>linkedEvents(c,d.id).some(v=>v.id===state.event)).map(d=>d.id);
    const message=linked.length?`此过程被 ${linked.join('、')} 引用。删除后会移除这些关联，需要重新选择证据。`:'将删除此关键过程，保留其他标注。';
    if(confirm(message)&&removeEvent(c,state.event)){state.event='';dirty();renderFormOnly();}
  }
  if(action==='prev-page'||action==='next-page'){state.page+=action==='prev-page'?-1:1;renderEvidenceOnly();$('#evidence-content').scrollTop=0;}
  if((action==='prev-context'||action==='next-context') && c?.contexts.length){
    const index=Math.max(0,c.contexts.findIndex(ctx=>ctx.id===state.context));
    const next=Math.max(0,Math.min(c.contexts.length-1,index+(action==='prev-context'?-1:1)));
    state.context=c.contexts[next].id;renderEvidenceOnly();$('#evidence-content').scrollTop=0;
    const same=$(`[data-action="${action}"]`);
    (same?.disabled?$('.context-navigation button:not(:disabled)'):same)?.focus({preventScroll:true});
  }
  if((action==='prev-graph'||action==='next-graph') && c?.graphs.length){
    const graphs=graphViews(c)[state.graphView],selection=state.graphView==='turns'?'graphTurn':'graphFull';
    if(!graphs.length)return;
    const index=Math.max(0,graphs.findIndex(graph=>graph.id===state[selection]));
    const next=Math.max(0,Math.min(graphs.length-1,index+(action==='prev-graph'?-1:1)));
    state[selection]=graphs[next].id;state.node='';renderEvidenceOnly();$('#evidence-content').scrollTop=0;
    const same=$(`[data-action="${action}"]`);
    (same?.disabled?$('.graph-navigation button:not(:disabled)'):same)?.focus({preventScroll:true});
  }
  if(action==='complete'){
    const missing=validateReview(c);if(missing.length){toast('还需填写：'+missing.join('、'),true);return;}
    c.annotation.status='reviewed';c.updatedAt=new Date().toISOString();if(sharedMode)c.remote.dirty=true;pending.set(c.id,c);await flush();renderShell();if(!saveError)toast(sharedMode?'已完成标注并保存到服务器。':'已完成标注。建议导出工作包备份。');
  }
  if(action==='load-full-evidence')await ensureFullEvidence();
  if(action?.startsWith('export-'))await exportData(action.slice(7));
  if(action==='delete' && c) await deleteCase(c.id);
});
document.addEventListener('pointerover', e => {
  const help = e.target.closest('.field-help');
  if (help && !help.contains(e.relatedTarget)) help.classList.remove('help-dismissed');
});
document.addEventListener('focusin', e => {
  e.target.closest('.field-help')?.classList.remove('help-dismissed');
});
document.addEventListener('keydown',e=>{
  if (e.key === 'Escape') document.querySelectorAll('.field-help:hover,.field-help:focus-within').forEach(el => el.classList.add('help-dismissed'));
  if((e.key==='Enter'||e.key===' ')&&e.target.matches('[data-node][role="button"]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
});
document.addEventListener('input',e=>{
  const el=e.target;
  if(aiReadOnly() && el.matches(reviewMutations))return;
  if(el.dataset.field){
    if(sharedMode && el.dataset.field==='reviewer')return;
    const c=current();let dest=c.annotation;const keys=el.dataset.field.split('.');const last=keys.pop();for(const key of keys)dest=dest[key];dest[last]=el.value;
    const supplement = el.closest('.event-supplement');
    if (supplement) {
      const filled = [...supplement.querySelectorAll('[data-field]')].filter(field => field.value.trim()).length;
      supplement.querySelector('.event-detail-count').textContent = filled ? `已填写 ${filled} 项` : '';
    }
    if (el.dataset.field.startsWith('events.') && last === 'label') {
      const index = Number(keys[1]), event = c.annotation.events[index];
      const tab = [...document.querySelectorAll('.event-switcher [data-event]')].find(button => button.dataset.event === event.id);
      if (tab) {tab.textContent = `E${String(index + 1).padStart(2, '0')} ${event.label || '关键过程'}`;tab.title = event.label || '关键过程';}
    }
    dirty(c);return;
  }
  if(el.id==='case-search'){state.sidebarSearch=el.value;const pos=el.selectionStart;$('.sidebar').outerHTML=renderSidebar();$('#case-search').focus();$('#case-search').setSelectionRange(pos,pos);}
  if(el.id==='trace-search'||el.id==='graph-search'){state.query=el.value;state.page=0;const id=el.id,pos=el.selectionStart;renderEvidenceOnly();$('#'+id).focus();$('#'+id).setSelectionRange(pos,pos);}
});
document.addEventListener('change',e=>{
  const el=e.target;
  if(aiReadOnly() && el.matches(reviewMutations))return;
  if (el.matches('[data-no-dimensions]')) {
    const c=current();if (!c || !confirmNoApplicableDimensions(c,el.checked)) return;
    const top=$('#form-content').scrollTop;
    dirty(c);renderFormOnly();$('#form-content').scrollTop=top;
    $('[data-no-dimensions]')?.focus({preventScroll:true});return;
  }
  if(el.dataset.linkDimension){
    const c=current();
    if(!c||!setDimensionEvent(c,el.dataset.linkDimension,el.dataset.linkEvent,el.checked)){el.checked=false;return;}
    el.closest('.process-option').classList.toggle('linked',el.checked);
    el.closest('.process-evidence').querySelector('.process-count').textContent=`已关联 ${linkedEvents(c,el.dataset.linkDimension).length} 个`;
    dirty(c);return;
  }
  if(el.id==='agent-select'){state.source=el.value;state.query='';state.filter='';state.page=0;renderEvidenceOnly();$('#agent-select')?.focus({preventScroll:true});}
  if(el.id==='role-select'){state.filter=el.value;state.page=0;renderEvidenceOnly();}
});
$('#file-input').addEventListener('change',e=>readUploads(e.target.files,state.attach));
$('#dimension-form').addEventListener('submit',e=>{
  e.preventDefault();
  const c=current();if(!c || c.id!==dimensionEditorCase)return;
  try{
    const title=$('#dimension-name').value, description=$('#dimension-description').value;
    const dimension=dimensionEditorId ? updateCustomDimension(c,dimensionEditorId,title,description) : addCustomDimension(c,title,description);
    $('#dimension-dialog').close();dirty(c);state.form='dimensions';renderFormOnly();focusDimension(dimension.id);
    toast(dimensionEditorId ? '维度已更新。' : '已添加维度，请关联关键过程并填写评价。');
  }catch(error){$('#dimension-error').textContent=error.message;$('#dimension-name').focus();}
});
$('#folder-input').addEventListener('change',e=>readUploads(e.target.files,false));
$('#close-help').addEventListener('click',()=>$('#help-dialog').close());
document.addEventListener('dragover',e=>{e.preventDefault();$('#drop-zone')?.classList.add('dragging');});
document.addEventListener('dragleave',()=>$('#drop-zone')?.classList.remove('dragging'));
document.addEventListener('drop',e=>{e.preventDefault();$('#drop-zone')?.classList.remove('dragging');if(e.dataTransfer?.files.length)readUploads(e.dataTransfer.files,false);});
document.addEventListener('scroll',e=>{
  if (['evidence-content','form-content'].includes(e.target.id)) scheduleViewSave();
},true);
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'){rememberView();flush();}});
window.addEventListener('pagehide',rememberView);
window.addEventListener('beforeunload',e=>{rememberView();if(pending.size || saving){flush();e.preventDefault();e.returnValue='';}});

let savedView=null;
try{savedView=localStorage.getItem(viewStorageKey);}catch{}
try {
  // Read-only AI results never read personal drafts. Human shared pages load
  // only this user's task draft after session resolution, not the entire DB.
  if(!sharedMode||remote?.review!=='ai')db=await openDB();
  if(!sharedMode)state.cases=(await dbOperation('getAll')).filter(c=>!c.remote);
  if(!sharedMode){
  for(const c of state.cases)if(updateDemoTitle(c))pending.set(c.id,c);
  if(pending.size)await flush();
  }
  state.cases.sort((a,b)=>(b.updatedAt||'').localeCompare(a.updatedAt||''));
  navigation=readViewState(savedView,state.cases.map(c=>c.id));
  state.active=navigation.active;
  Object.assign(state,normalizeView(navigation.cases[state.active]));
}
catch(e){toast('浏览器持久化存储不可用。仍可标注，请及时导出工作包。',true);}
if(sharedMode) {
  try {
    if(startupError)throw new Error(startupError);
    const loaded=await loadSharedCase(remote,id=>dbOperation('get',id).catch(()=>null));
    sharedRun=loaded.run;state.cases=[loaded.c];state.active=loaded.c.id;
    navigation=readViewState(savedView,[loaded.c.id]);
    Object.assign(state,normalizeView(navigation.cases[state.active]));
    if(loaded.c.remote.readOnly && !navigation.cases[state.active]){state.form='events';state.event=loaded.c.annotation.events[0]?.id||'';}
    if(loaded.c.remote.dirty){pending.set(loaded.c.id,loaded.c);if(loaded.c.remote.conflict){saveError='服务器标注已更新，请先导出本地草稿备份，再加载服务器版本';toast(saveError,true);}else saveTimer=setTimeout(flush,450);}
  }catch(error){state.cases=[];state.active='';startupError=error.message;}
}
renderShell();

function copyFragments() {
  const before={...state}, sample=ingestFiles([]);
  const fragments=[renderEmpty()];
  try {
    for(const form of ['basic','dimensions','events','conclusion']){
      state.form=form;fragments.push(renderForm(sample));
    }
    sample.annotation.events.push(newEvent());
    state.form='events';fragments.push(renderForm(sample));
    fragments.push(renderTrajectory(sample),renderContext(sample),renderGraph(sample),renderReport(sample));
    setDimensionSelected(sample,'A1',true);
    state.form='dimensions';fragments.push(renderWorkspace(sample));
  } finally {Object.assign(state,before);}
  return fragments;
}
await mountCopyRuntime({
  fragments:new URLSearchParams(location.search).has('copy-editor')?copyFragments():[],
  async navigate(target){
    await flush();
    if(target==='home'){$('#help-dialog').close();activateCase('');renderShell();}
    if(target==='demo'){$('#help-dialog').close();await demo();}
    if(target==='guide')showGuide(0);
  },
});

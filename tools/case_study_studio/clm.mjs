import {renderMarkdown} from './markdown.mjs';

export const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const parseJSON = value => {try{return JSON.parse(value || 'null');}catch{return null;}};
export const formatNumber = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
export const duration = value => Number.isFinite(value) ? `${Math.floor(Math.round(value)/60)} 分 ${Math.round(value)%60} 秒` : '—';
export function turnLabel(turn) {
  if(turn.edited) return 'Context 编辑';
  if(turn.submitted) return '提交报告';
  const cmd=turn.command||'';
  if(cmd.includes('web_search'))return '搜索';
  if(cmd.includes('web_fetch'))return '抓取';
  if(cmd.includes('read_source')||cmd.includes('find_in_source'))return '读取来源';
  if(cmd.includes('report.md'))return '报告';
  return cmd?'Bash':'模型输出';
}
export function diffHTML(text) {
  return `<pre class="clm-code">${String(text).split('\n').map(line=>{
    const cls=line.startsWith('@@')?'heading':line.startsWith('+')&&!line.startsWith('+++')?'add':line.startsWith('-')&&!line.startsWith('---')?'del':'';
    return `<span class="clm-diff-line${cls?' clm-diff-'+cls:''}">${escape(line)||' '}</span>`;
  }).join('')}</pre>`;
}
const code = text => `<pre class="clm-code">${escape(text)}</pre>`;
const empty = text => `<p class="clm-empty">${escape(text)}</p>`;
const block = (title,body) => `<section class="clm-output-block"><h4>${escape(title)}</h4>${body}</section>`;
const markdown = text => `<div class="report-markdown">${renderMarkdown(text)}</div>`;
function reasoning(message) {return message.reasoning ?? message.reasoning_content ?? '';}
function messageText(content) {
  if(typeof content==='string')return content;
  return Array.isArray(content)?content.map(p=>p.text??JSON.stringify(p)).join('\n'):content?JSON.stringify(content):'';
}
function toolCalls(message) {
  return (message.tool_calls||[]).map(tc=>{
    const fn=tc.function||{},args=parseJSON(fn.arguments);
    return block(`Tool call · ${fn.name||tc.type||'tool'}`,`<div class="clm-command">${code(args?.command??(args?JSON.stringify(args,null,2):fn.arguments||''))}</div>`);
  }).join('');
}
export function messageHTML(message) {
  const thought=reasoning(message), text=messageText(message.content);
  return (thought?block('Thinking（原始返回）',markdown(thought)):'')+(text?block(message.role==='tool'?'工具结果':'正文',markdown(text)):'')+toolCalls(message);
}
export function messagesHTML(messages) {
  if(!Array.isArray(messages)||!messages.length)return empty('没有 messages 记录。');
  return messages.map((m,i)=>{
    const text=messageText(m.content),preview=text.replace(/\s+/g,' ').slice(0,75);
    return `<details class="clm-message"${i>=messages.length-3?' open':''}><summary>${i+1} · ${escape(m.role||'message')}<span class="clm-message-preview">${escape(preview)}</span></summary><div class="clm-message-content">${messageHTML(m)||empty('空消息')}</div></details>`;
  }).join('');
}

async function start() {
  const $=selector=>document.querySelector(selector), all=selector=>[...document.querySelectorAll(selector)];
  const state={run:null,detail:null,number:0,turn:null,context:'after',model:'output',section:'turns',cache:new Map(),ticket:0,runTicket:0,report:null};
  const endpoint=()=>`/api/clm/runs/${encodeURIComponent(state.run)}`;
  async function get(url) {const response=await fetch(url,{cache:'no-store'});const data=await response.json();if(!response.ok)throw new Error(data.error||'读取失败');return data;}
  function fail(error) {$('#clm-error').textContent=error.message||String(error);$('#clm-error').hidden=false;}
  function url() {const q=new URLSearchParams({run:state.run,turn:state.number});history.replaceState(null,'',location.pathname+'?'+q);}
  const visibleTurns=()=>state.detail?.turns.filter(t=>!$('#edits-only').checked||t.edited)||[];
  function turnList() {
    const turns=visibleTurns();$('#turn-count').textContent=`${turns.length} 轮`;
    $('#clm-turn-list').innerHTML=turns.length?turns.map(t=>`<button class="clm-turn${t.edited?' edited':''}" data-turn="${t.number}" ${state.number===t.number?'aria-current="step"':''} title="第 ${t.number} 次调用 · ${escape(turnLabel(t))}"><b>${String(t.number).padStart(2,'0')}</b><span>${escape(turnLabel(t))}</span></button>`).join(''):empty('没有已应用的编辑。');
    const index=turns.findIndex(t=>t.number===state.number);$('#prev-turn').disabled=index<=0;$('#next-turn').disabled=index<0||index>=turns.length-1;
    const list=$('#clm-turn-list'),active=list.querySelector('[aria-current="step"]');
    if(active){const a=active.getBoundingClientRect(),b=list.getBoundingClientRect();if(a.top<b.top||a.bottom>b.bottom)list.scrollTop+=a.top-b.top-(list.clientHeight-a.height)/2;}
  }
  function download(name,text) {
    if(text==null)return;
    const link=document.createElement('a'),href=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));link.href=href;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(href),1000);
  }
  const contextFile=()=>({before:'context.before.txt',after:'context.after.txt',diff:'context.diff',parsed:'context.parsed.json'}[state.context]);
  const modelFile=()=>state.model.includes('input')?'input.json':'output.json';
  function renderContext() {
    if(!state.turn)return;
    const files=state.turn.files,text=files[contextFile()],step=parseJSON(files['step.json'])||{};
    all('[data-context]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.context===state.context)));
    let html;
    if(state.context==='diff')html=text!=null?diffHTML(text):empty(step.file_changed?'这一轮没有单独保存差异文件，可对照编辑前后原文。':'这一轮 context 文件没有变化。');
    else if(state.context==='parsed')html=messagesHTML(parseJSON(text));
    else html=text==null?empty('这一轮未记录 context 文件。'):text===''?empty('可编辑历史为空，system 和初始任务保留在模型输入中。'):code(text);
    $('#clm-context').innerHTML=html;
    const note=state.context==='parsed'?'这是编辑校验后的历史，尚未追加本轮输出与工具结果。':step.context_edit_applied?'文件修改已应用。下一轮输入使用这份历史，并追加当前轮交互。':step.file_changed?'文件发生变化，但修改未被 CLM 接受。':'本轮没有应用 context 编辑。';
    $('#clm-context-note').textContent=note;
    $('[data-download="context"]').disabled=text==null;
  }
  function renderModel() {
    if(!state.turn)return;
    const files=state.turn.files,raw=files[modelFile()],data=parseJSON(raw);
    all('[data-model]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.model===state.model)));
    let html;
    if(state.model.startsWith('raw-'))html=raw!=null?code(raw):empty('记录尚未保存。');
    else if(state.model==='input')html=messagesHTML(data?.messages)+(data?.tools?.length?`<details class="clm-message"><summary>工具定义 · ${data.tools.length} 个</summary><div class="clm-message-content">${code(JSON.stringify(data.tools,null,2))}</div></details>`:'');
    else {
      const choices=data?.choices||[],step=parseJSON(files['step.json']);
      html=choices.map(c=>messageHTML(c.message||{})).join('')||empty('没有输出记录。');
      if(step)html+=block('命令执行结果',code(`${step.stdout||''}${step.stderr?'\n'+step.stderr:''}\n(exit_code=${step.return_code})`));
      if(step?.notes)html+=block('CLM 编辑回执',code(step.notes.trim()));
    }
    $('#clm-model').innerHTML=html; $('[data-download="model"]').disabled=raw==null;
    $('#clm-input-note').textContent=state.model.includes('input')?'这是本轮编辑之前发送至模型服务的实际请求。':'Thinking、正文与 Tool call 均来自原始 API 返回，没有改写。';
    $('#show-next-input').disabled=!state.detail.turns.some(t=>t.number===state.number+1);
  }
  async function selectTurn(number) {
    const item=state.detail.turns.find(t=>t.number===Number(number));if(!item)return;
    state.number=item.number;state.turn=null;url();turnList();const ticket=++state.ticket;
    $('#clm-turn-title').textContent=`第 ${item.number} 次模型调用 · ${turnLabel(item)}`;
    $('#clm-turn-meta').textContent=`Input ${formatNumber(item.usage.prompt_tokens)} · Output ${formatNumber(item.usage.completion_tokens)} · Total ${formatNumber(item.usage.total_tokens)} tokens`;
    $('#clm-panels').setAttribute('aria-busy','true');
    for(const pane of ['#clm-context','#clm-model'])$(pane).innerHTML='<div class="clm-loading" role="status"><i class="clm-spinner"></i>正在读取这一轮…</div>';
    all('[data-download]').forEach(b=>b.disabled=true);
    try {
      const data=state.cache.get(item.number)||await get(endpoint()+'/turns/'+item.number);
      if(ticket!==state.ticket)return;
      state.cache.set(item.number,data);state.turn=data;renderContext();renderModel();
    } catch(error){if(ticket===state.ticket)fail(error);}
    finally{if(ticket===state.ticket)$('#clm-panels').setAttribute('aria-busy','false');}
  }
  function settings() {
    const d=state.detail,s=d.settings;
    const rows=[['Agent model',s.model_name],['Thinking',d.reasoning_effort||(s.enable_thinking?'开启':'关闭')],['CLM 有效 context 预算',`${formatNumber(s.context_budget_tokens-s.context_budget_reserve_tokens)} tokens`],['最大输出',`${formatNumber(s.max_tokens)} tokens`],['Temperature / top-p',`${s.temperature??'默认'} / ${s.top_p??'默认'}`],['Context 编辑规则',s.allow_edit_growth==='shrink'?'shrink · 只接受缩短历史的编辑':s.allow_edit_growth],['搜索 / 抓取上限',`${d.limits?.search_calls??50} / ${d.limits?.fetch_calls??50}`],['模型调用上限',s.lm_call_cap],['CLM commit',d.clm_commit],['本次状态',d.status==='submitted'?'已提交报告':d.status]];
    $('#clm-settings-section').innerHTML=`<h2>运行设置</h2><dl class="clm-settings-grid">${rows.map(([k,v])=>`<div><dt>${escape(k)}</dt><dd>${escape(v??'—')}</dd></div>`).join('')}</dl><p class="clm-settings-note">使用官方 CLM 推理框架与所选 Agent 模型，没有使用论文训练权重。System 和初始用户任务固定保留，其余历史由模型编辑文件后交给 CLM 解析。界面只读取原始记录，不重新调用模型。${d.managed?'报告评分可在运行中心查看。':''}</p><p class="clm-settings-note">Output 使用服务端的总输出 Token 口径，包含 Thinking。</p>`;
  }
  async function section(name) {
    state.section=name;
    all('[data-section]').forEach(b=>{if(b.dataset.section===name)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    $('#clm-turn-section').hidden=name!=='turns';$('#clm-report-section').hidden=name!=='report';$('#clm-settings-section').hidden=name!=='settings';
    if(name==='settings'){settings();return;}
    if(name!=='report')return;
    $('#clm-report-section').innerHTML='<div class="clm-loading"><i class="clm-spinner"></i>正在读取任务与报告…</div>';
    const run=state.run;
    try {
      const data=state.report||await get(endpoint()+'/report');if(run!==state.run)return;state.report=data;
      $('#clm-report-section').innerHTML=`<details class="clm-task"><summary>${escape(state.detail.task_id)} · 原始任务要求</summary>${data.task?markdown(data.task):empty('未保存任务文本。')}</details><div class="clm-report-heading"><h2>最终报告</h2><button class="button" id="download-report"${data.report==null?' disabled':''}>下载 Markdown ↓</button></div>${data.report!=null?markdown(data.report):empty('未保存最终报告。')}`;
      $('#download-report').onclick=()=>download(`clm-${state.run}-report.md`,data.report);
    }catch(error){fail(error);}
  }
  async function selectRun(id,number) {
    const ticket=++state.runTicket;++state.ticket;state.run=id;state.cache.clear();state.turn=null;state.report=null;$('#clm-error').hidden=true;
    try{
      const data=await get(endpoint());if(ticket!==state.runTicket)return;state.detail=data;
      $('#run-subtitle').textContent=`DeepResearch Bench II · ${data.task_id} · ${(data.settings.model_name||'').replace(/^openai\//,'')} · ${data.status==='submitted'?'已提交报告':data.status}`;
      const stats=[['模型调用',data.turns.length,'每一轮均可查看原始 I/O'],['Context 编辑',data.accepted_edits,'已被 CLM 接受的文件修改'],['运行用时',duration(data.wall_seconds),data.started_at?new Date(data.started_at).toLocaleString('zh-CN',{hour12:false}):''],['Total tokens',formatNumber(data.usage.total_tokens),`Input ${formatNumber(data.usage.prompt_tokens)} · Output ${formatNumber(data.usage.completion_tokens)}`]];
      $('#clm-stats').innerHTML=stats.map(([k,v,sub])=>`<div class="clm-stat"><small>${escape(k)}</small><strong>${escape(v)}</strong><span>${escape(sub)}</span></div>`).join('');
      $('#edits-only').checked=false;await selectTurn(data.turns.some(t=>t.number===Number(number))?number:data.turns[0]?.number);await section(state.section);
    }catch(error){fail(error);}
  }
  $('#clm-run').onchange=e=>selectRun(e.target.value,1);
  $('#clm-refresh').onclick=async()=>{
    const button=$('#clm-refresh');button.disabled=true;button.textContent='刷新中…';
    try{if(state.run)await selectRun(state.run,state.number);}finally{button.disabled=false;button.textContent='刷新记录';}
  };
  $('#clm-turn-list').onclick=e=>{const b=e.target.closest('[data-turn]');if(b)selectTurn(b.dataset.turn);};
  $('#edits-only').onchange=()=>{const turns=visibleTurns();if(turns.length&&!turns.some(t=>t.number===state.number))selectTurn(turns[0].number);else turnList();};
  const move=delta=>{const turns=visibleTurns(),i=turns.findIndex(t=>t.number===state.number);if(turns[i+delta])selectTurn(turns[i+delta].number);};
  $('#prev-turn').onclick=()=>move(-1);$('#next-turn').onclick=()=>move(1);
  $('#show-next-input').onclick=()=>{state.model='input';$('#edits-only').checked=false;selectTurn(state.number+1);};
  all('[data-context]').forEach(b=>b.onclick=()=>{state.context=b.dataset.context;renderContext();});
  all('[data-model]').forEach(b=>b.onclick=()=>{state.model=b.dataset.model;renderModel();});
  all('[data-section]').forEach(b=>b.onclick=()=>section(b.dataset.section));
  all('[data-download]').forEach(b=>b.onclick=()=>{if(!state.turn)return;const file=b.dataset.download==='context'?contextFile():modelFile();download(`call-${String(state.number).padStart(4,'0')}-${file}`,state.turn.files[file]);});
  document.addEventListener('keydown',e=>{if(state.section!=='turns'||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,a,[contenteditable]'))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();move(e.key==='ArrowLeft'?-1:1);}});
  try {
    const data=await get('/api/clm/runs'),q=new URLSearchParams(location.search);
    if(!data.runs.length)throw new Error('当前没有发布的 CLM 记录。');
    $('#clm-run').innerHTML=data.runs.map(r=>`<option value="${escape(r.id)}">${r.managed?escape(r.id)+' · ':''}${escape(r.task_id)} · ${escape((r.settings.model_name||'').replace(/^openai\//,''))} · ${escape((r.started_at||'').slice(0,16).replace('T',' '))}</option>`).join('');
    const id=data.runs.some(r=>r.id===q.get('run'))?q.get('run'):data.runs[0].id;$('#clm-run').value=id;await selectRun(id,q.get('turn')||1);
  }catch(error){fail(error);$('#run-subtitle').textContent='记录暂时不可用';}
}

if(typeof document!=='undefined')start();

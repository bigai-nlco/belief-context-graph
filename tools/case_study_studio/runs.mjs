import {api, storedToken, storeToken, logout, caseLink, benchmarkLabel, contextModeLabel, defaultRunTitle, taskAnnotationStatus, canAnnotateTask, annotationProgress, reportScoreLabel, runDisplayStatus, runScoreSummary} from './shared.mjs';
import {mountTaskPicker} from './task-picker.mjs';
import {mountScoreDetails} from './score-details.mjs';
import {renderVersion, versionLabel} from './run-version.mjs';

const $=s=>document.querySelector(s);
const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>typeof x==='number'?x.toLocaleString('en-US'):'—';
const time=x=>x?new Date(x).toLocaleString('zh-CN',{hour12:false}):'—';
const names={queued:'排队中',starting:'正在启动',running:'运行中',finished:'已结束',finished_with_task_errors:'已结束 · 有失败',completed:'已产出报告',timeout:'超时',cancelled:'已停止',interrupted:'已中断',failed:'启动 / 执行失败',not_run:'未执行',agent_error:'Agent 错误',worker_error:'执行错误',empty_report:'无报告',graph_fallback:'Graph 回退',score_error:'评分错误'};
Object.assign(names,{judging:'报告评分中',finished_with_judge_errors:'已结束 · 评分未完成'});
const badge=status=>`<span class="run-status ${escape(status)}">${escape(names[status]||status)}</span>`;
let user=null,settings=null,selected=Number(new URLSearchParams(location.search).get('run'))||0,refreshTask=null,cancelTarget=0,noticeTimer,refreshFeedbackTimer;
let suggestedRunTitle='';
let taskPicker=null;
const pendingReviews=new Set();
const scoreDetails=mountScoreDetails($('#score-dialog'));

function reviewAction(run,task){
  const ai=task.ai_review||{},key=`${run.id}/${task.benchmark}/${task.id}`;
  const attributes=`data-ai-review="${escape(task.id)}" data-benchmark="${escape(task.benchmark)}" data-run-id="${run.id}"`;
  if(!canAnnotateTask(task))return `<button class="button" ${attributes} disabled title="仅正常完成的任务可以标注">AI 标注</button>`;
  if(ai.status==='completed')return `<a class="button ai-review-link" href="${escape(caseLink(run.id,task.benchmark,task.id,'ai'))}" title="查看已完成的 AI 标注">查看 AI 标注</a>`;
  if(pendingReviews.has(key))return '<button class="button" disabled>正在提交…</button>';
  if(['queued','running'].includes(ai.status))return `<button class="button ai-review-busy" disabled aria-busy="true"><span class="ai-review-label">${ai.status==='queued'?'AI 标注排队中':'AI 标注中'}</span><span class="ai-review-spinner" aria-hidden="true"></span></button>${ai.batches?`<small class="ai-review-progress"><span>已分析 ${ai.completed_batches} / ${ai.batches} 批</span>${ai.completed_batches===ai.batches?'<span>正在汇总</span>':''}</small>`:''}`;
  const failed=['failed','interrupted'].includes(ai.status);
  return `<button class="button" ${attributes} ${failed?'data-retry="true"':''} ${task.report?'':'disabled'} title="${failed?'重试 · ':''}LiteLLM · GPT-6.1-Sol · medium">${failed?'重试 AI 标注':'AI 标注'}</button>${failed?`<small class="ai-review-error">${escape(ai.error||'AI 标注未完成')}</small>`:''}`;
}

function annotationStatus(task){
  const status=taskAnnotationStatus(task);
  const reviewers=(task.reviews||[]).map(r=>`${r.name} · ${r.status==='reviewed'?'已完成':'草稿'}`).join('；');
  return `<span class="run-status ${status.completed?'completed':status.inProgress?'running':'not_run'}" data-annotation-status="${status.completed?'completed':status.inProgress?'in_progress':'not_started'}" title="${escape(reviewers)}">${status.label}</span>`;
}

function notice(text){$('#run-notice').textContent=text;$('#run-notice').hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>$('#run-notice').hidden=true,7000);}
function hideIdentityToken(){
  $('#identity-token').value='';$('#identity-token-panel').hidden=true;
  $('#show-token').textContent='查看身份码';$('#show-token').setAttribute('aria-expanded','false');
}
function identity(){
  $('#identity-form').hidden=!!user;$('#identity-current').hidden=!user;
  $('#identity-title').textContent=user?'标注者信息':'标注身份';
  $('#identity-name').textContent=user?.name||'';
  $('#identity-error').textContent='';hideIdentityToken();
  if(!$('#identity-dialog').open)$('#identity-dialog').showModal();
  if(!user)$('#identity-form [name=name]').focus();
}
// Patch these read-only, escaped templates in place. Replacing a scroll container
// on every poll and restoring scrollLeft wakes the browser's overlay scrollbar,
// even when the data is unchanged. Stable keys also preserve rows and focus.
const liveKey=node=>node.nodeType===1?(node.getAttribute('data-live-key')||node.id):'';
function patchLiveChildren(parent,nextParent){
  const keyed=new Map([...parent.childNodes].filter(liveKey).map(node=>[liveKey(node),node]));
  const nextKeys=new Set([...nextParent.childNodes].map(liveKey).filter(Boolean));
  let cursor=parent.firstChild;
  for(const next of [...nextParent.childNodes]){
    const key=liveKey(next);
    let node=key?keyed.get(key):cursor&&!liveKey(cursor)?cursor:null;
    if(!node||node.nodeType!==next.nodeType||node.nodeName!==next.nodeName){
      node=next.cloneNode(true);parent.insertBefore(node,cursor);
    }else{
      // Remove obsolete siblings instead of unnecessarily detaching a retained
      // keyed section (for example when a run-level error notice disappears).
      if(key)while(cursor&&cursor!==node&&!nextKeys.has(liveKey(cursor))){
        const after=cursor.nextSibling;cursor.remove();cursor=after;
      }
      if(node!==cursor)parent.insertBefore(node,cursor);
      if(!node.isEqualNode(next)){
        if(node.nodeType===1){
          for(const {name} of [...node.attributes])if(!next.hasAttribute(name)&&!(name==='open'&&node.matches('details[data-preserve-open]')))node.removeAttribute(name);
          for(const {name,value} of next.attributes)if(node.getAttribute(name)!==value)node.setAttribute(name,value);
          patchLiveChildren(node,next);
        }else if(node.nodeValue!==next.nodeValue)node.nodeValue=next.nodeValue;
      }
    }
    cursor=node.nextSibling;
  }
  while(cursor){const next=cursor.nextSibling;cursor.remove();cursor=next;}
}
function renderLive(parent,html,{reset=false}={}){
  const template=document.createElement('template');template.innerHTML=html;
  if(reset)parent.replaceChildren(template.content);
  else patchLiveChildren(parent,template.content);
}
function renderList(runs){
  renderLive($('#run-list'),runs.length?runs.map(run=>`<button class="run-item ${run.id===selected?'selected':''}" data-run="${run.id}" data-live-key="run-${run.id}"><div><strong>${escape(run.number)}</strong>${badge(runDisplayStatus(run))}</div><p class="run-name">${escape(run.title)}</p><div><small>${escape(contextModeLabel(run.mode))} · ${run.ended} / ${run.total} 已结束</small><small>${run.reports} 份报告</small></div><small class="run-version-label">${escape(versionLabel(run))}</small></button>`).join(''):'<p class="muted">暂无测试。点击右上角启动测试。</p>');
}
function renderRun(run){
  const current=$('#run-content');
  const agent=run.tasks.reduce((n,t)=>n+(t.agent_tokens||0),0),graph=run.tasks.reduce((n,t)=>n+(t.graph_tokens||0),0);
  const tokenLabel=run.tasks.some(t=>t.calls>0&&(t.agent_tokens==null||t.graph_tokens==null))?'已记录模型 Token · 用量不完整':'累计模型 Token';
  const annotations=annotationProgress(run.tasks);
  const scores=runScoreSummary(run.tasks);
  const scoreTotal=scores.points===null?'—':scores.points.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
  const annotationText=annotations.total?`已标注 ${annotations.completed} / ${annotations.total} 题`:'暂无可标注任务';
  const seconds=run.started_at?Math.max(0,(new Date(run.finished_at||Date.now())-new Date(run.started_at))/1000):0;
  renderLive(current,`<section class="run-heading" data-live-key="heading">
    <span class="eyebrow">${escape(run.number)}</span>
    <div class="run-heading-row">
      <h1>${escape(run.title)}</h1>
      <div class="run-heading-actions"><a class="button" href="./annotation-stats.html?run=${run.id}">标注统计</a>${badge(runDisplayStatus(run))}${run.managed&&['queued','starting','running'].includes(run.status)?`<button class="button" id="stop-run" data-run-id="${run.id}" ${run.cancel_requested?'disabled':''}>${run.cancel_requested?'正在停止…':'停止测试'}</button>`:''}</div>
    </div>
    <p>模式：${escape(contextModeLabel(run.mode))} · Agent：${escape(run.agent||'待启动')} · Graph：${run.mode==='default'?'不启用':escape(run.graph)} · 并发 ${run.workers||'—'}<br>启动时间：${time(run.started_at||run.created_at)}${run.finished_at?` · 结束：${time(run.finished_at)}`:''}</p>
    ${run.judge?.enabled?`<p>Report 评分：${run.tasks.filter(t=>t.scoring_status==='official_protocol_scored').length} / ${run.tasks.filter(t=>t.status==='completed').length} 份报告${run.judge.model?` · Judge ${escape(run.judge.model)}`:''}</p>`:''}
    ${renderVersion(run)}
    </section>
    ${run.error?`<div class="notice amber" data-live-key="run-error">${escape(run.error)}</div>`:''}
    <section class="run-stats" data-live-key="stats">
      <div class="run-stat"><span>已产出报告 / 任务数</span><strong>${run.reports} / ${run.total}</strong></div>
      <div class="run-stat run-score-stat" title="已完成 Judge 评分的单题分数之和，每题满分 100。未评分不计入总分和均分。"><span>Judge 总分</span><strong class="run-score-total">${scoreTotal}</strong><small class="run-score-average">${scores.average===null?'暂无评分':`已评分题均分 ${scores.average.toFixed(1)} / 100`}</small><small class="run-score-coverage">已评分 ${scores.scored} / ${scores.total} 题</small></div>
      <div class="run-stat"><span>超时 / 错误 / 中断</span><strong>${run.failed}</strong></div>
      <div class="run-stat"><span>${tokenLabel}</span><strong>${num(agent+graph)}</strong></div>
      <div class="run-stat"><span>批次耗时</span><strong>${Math.floor(seconds/60)} <small>分钟</small></strong></div>
    </section>
    <section class="run-progress" aria-label="测试与标注进度" data-live-key="progress">
      <div class="progress-track progress-test"><div class="progress-track-heading"><label for="run-test-progress">测试进度</label><span>已结束 ${run.ended} / ${run.total} 题</span></div><progress id="run-test-progress" value="${run.ended}" max="${run.total||1}" aria-valuetext="已结束 ${run.ended} / ${run.total} 题"></progress></div>
      <div class="progress-track progress-annotation"><div class="progress-track-heading"><label for="run-annotation-progress" title="人工或 AI 任一完成即计入，每题只计一次">标注进度 <small>仅统计可标注任务</small></label><span>${annotationText}</span></div><progress id="run-annotation-progress" value="${annotations.completed}" max="${annotations.total||1}" aria-valuetext="${annotationText}"></progress></div>
      <div class="progress-usage">Agent ${num(agent)} · Graph ${num(graph)}</div>
    </section>
    <section class="run-table-wrap" data-live-key="task-table"><table class="run-table"><colgroup><col class="run-col-task"><col class="run-col-status"><col class="run-col-score"><col class="run-col-calls"><col class="run-col-search"><col class="run-col-agent"><col class="run-col-graph"><col class="run-col-reviews"><col class="run-col-result"></colgroup><thead><tr><th>Benchmark / Task</th><th>运行状态</th><th title="LLM Judge 对最终 Report 的评分，满分 100 分">分数</th><th>模型调用 / 轮次</th><th>搜索 / 抓取</th><th>Agent Token</th><th>Graph Token</th><th class="review-names">标注情况</th><th>结果</th></tr></thead><tbody>${run.tasks.map(task=>{
      const mine=task.reviews.find(r=>r.mine);
      const scoreButton=task.score_details_available?`<button type="button" class="button score-view-button" data-score-task="${escape(task.id)}" data-score-benchmark="${escape(task.benchmark)}" data-score-run="${run.id}" data-score-number="${escape(run.number)}" aria-label="查看 Task ${escape(task.id)} 的 Report 评分详情">查看</button>`:'';
      return `<tr data-live-key="${escape(task.benchmark)}:${escape(task.id)}"><td><strong>${escape(benchmarkLabel(task.benchmark))}</strong><small>${escape(task.id)}</small></td><td class="task-status">${badge(task.status)}${task.duration?`<small>${(task.duration/60).toFixed(1)} 分钟</small>`:''}</td><td class="task-score" data-scoring-status="${escape(task.scoring_status||'official_pending')}"><span class="report-score">${escape(reportScoreLabel(task))}</span>${scoreButton}</td><td>${num(task.calls)} 次<small>第 ${num(task.turn)} 轮</small></td><td>${num(task.searches)} / ${num(task.fetches)}${task.blocked?`<small>${task.blocked} 次来源拦截</small>`:''}</td><td class="task-agent-tokens"><span class="token-total">${num(task.agent_tokens)}</span><small class="token-breakdown"><span>输入 ${num(task.agent_input)}</span><span>输出 ${num(task.agent_output)}</span></small></td><td class="task-graph-tokens"><span class="token-total">${num(task.graph_tokens)}</span><small class="token-breakdown"><span>输入 ${num(task.graph_input)}</span><span>输出 ${num(task.graph_output)}</span></small></td><td class="review-names">${annotationStatus(task)}</td><td><div class="task-review-actions">${canAnnotateTask(task)?`<a class="button ${mine?'':'primary'}" href="${escape(caseLink(run.id,task.benchmark,task.id))}">人工标注</a>`:'<button class="button" disabled title="仅正常完成的任务可以标注">人工标注</button>'}${reviewAction(run,task)}</div></td></tr>`;
    }).join('')}</tbody></table></section>`,{reset:current.dataset.liveRun!==String(run.id)});
  current.dataset.liveRun=String(run.id);
}
function refresh(){
  if(!user)return Promise.resolve(false);
  // A manual click can join an in-flight automatic refresh instead of reporting
  // success before the outstanding request has actually finished.
  if(refreshTask)return refreshTask;
  refreshTask=(async()=>{
  try{
    const {runs}=await api('runs');
    if(!runs.some(r=>r.id===selected))selected=runs[0]?.id||0;
    renderList(runs);
    if(selected)renderRun(await api(`runs/${selected}`));
    else $('#run-content').innerHTML='<section class="run-heading"><div><span class="eyebrow">BCG DEEP RESEARCH</span><h1>启动测试，直接查看与标注</h1><p>同一个运行编号下查看任务进度、最终报告、运行轨迹和 Belief Graph。</p></div></section>';
    $('#connection').textContent='已更新 '+new Date().toLocaleTimeString('zh-CN',{hour12:false});
    return true;
  }catch(error){$('#connection').textContent='连接中断 · 正在重试';if(error.status===401){user=null;identity();}else notice(error.message);return false;}
  finally{refreshTask=null;}
  })();
  return refreshTask;
}
async function manualRefresh(){
  const button=$('#refresh-button');
  if(button.disabled)return;
  if(!user)return identity();
  clearTimeout(refreshFeedbackTimer);
  const setState=state=>{
    button.dataset.state=state;
    button.querySelector('.refresh-label').textContent={idle:'刷新',loading:'刷新中…',success:'已刷新',error:'刷新失败'}[state];
  };
  button.disabled=true;
  button.setAttribute('aria-busy','true');
  setState('loading');
  // Keep very fast local requests perceptible; do not delay rendering the data.
  const [ok]=await Promise.all([refresh(),new Promise(resolve=>setTimeout(resolve,700))]);
  setState(ok?'success':'error');
  button.disabled=false;
  button.setAttribute('aria-busy','false');
  refreshFeedbackTimer=setTimeout(()=>setState('idle'),1600);
}
document.addEventListener('click',async event=>{
  const stop=event.target.closest('#stop-run');
  if(stop&&!stop.disabled){cancelTarget=Number(stop.dataset.runId);$('#cancel-error').textContent='';$('#cancel-dialog').showModal();return;}
  const score=event.target.closest('[data-score-task]');
  if(score){
    await scoreDetails.open({run:Number(score.dataset.scoreRun),number:score.dataset.scoreNumber,benchmark:score.dataset.scoreBenchmark,task:score.dataset.scoreTask});
    return;
  }
  const review=event.target.closest('[data-ai-review]');
  if(review && !review.disabled){
    const key=`${review.dataset.runId}/${review.dataset.benchmark}/${review.dataset.aiReview}`;
    if(pendingReviews.has(key))return;
    pendingReviews.add(key);review.disabled=true;review.textContent='正在提交…';
    try{
      const result=await api(`runs/${key.split('/')[0]}/tasks/${review.dataset.benchmark}/${encodeURIComponent(review.dataset.aiReview)}/ai-review`,{method:'POST',body:{retry:review.dataset.retry==='true'}});
      notice(result.status==='completed'?'AI 标注已完成，可点击查看':'AI 标注已加入后台队列，关闭页面不影响执行');
    }catch(error){notice(error.message);}
    finally{pendingReviews.delete(key);await refresh();}
    return;
  }
  const close=event.target.closest('[data-close]');if(close)$('#'+close.dataset.close).close();
  const run=event.target.closest('[data-run]');if(run){selected=Number(run.dataset.run);history.replaceState(null,'','?run='+selected);await refresh();}
});
$('#refresh-button').addEventListener('click',manualRefresh);
$('#identity-button').addEventListener('click',identity);
$('#identity-dialog').addEventListener('close',hideIdentityToken);
$('#show-token').addEventListener('click',()=>{
  if(!$('#identity-token-panel').hidden)return hideIdentityToken();
  $('#identity-token').value=storedToken();$('#identity-token-panel').hidden=false;
  $('#show-token').textContent='隐藏身份码';$('#show-token').setAttribute('aria-expanded','true');
  $('#identity-token').focus();$('#identity-token').select();
});
$('#logout-button').addEventListener('click',()=>{logout();user=null;$('#identity-button').textContent='标注身份';$('#identity-token').value='';$('#identity-form').reset();identity();});
$('#identity-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=event.submitter;button.disabled=true;$('#identity-error').textContent='';
  try{const form=new FormData(event.target),result=await api('session',{method:'POST',body:{name:form.get('name'),token:form.get('token')}});storeToken(result.token);user=result.user;$('#identity-dialog').close();$('#identity-button').textContent=user.name;notice('标注会保存到服务器，个人身份码可在右上角查看。');await refresh();}
  catch(error){$('#identity-error').textContent=error.message;}finally{button.disabled=false;}
});
$('#launch-button').addEventListener('click',()=>{
  if(!user)return identity();
  if(!settings?.profiles?.length)return notice('服务器尚未配置可启动的模型');
  const title=$('#launch-form').elements.title;
  if(!title.value.trim() || title.value===suggestedRunTitle){
    suggestedRunTitle=defaultRunTitle();
    title.value=suggestedRunTitle;
  }
  $('#launch-error').textContent='';
  $('#launch-dialog').showModal();
});
$('#launch-profile').addEventListener('change',()=>{const profile=settings.profiles.find(p=>p.id===$('#launch-profile').value);$('#launch-form').elements.workers.max=profile.max_workers;$('#launch-form').elements.workers.value=Math.min(Number($('#launch-form').elements.workers.value),profile.max_workers);});
$('#launch-mode').addEventListener('change',()=>{
  $('#launch-context-info').innerHTML=$('#launch-mode').value==='default'
    ?'单 Agent Default · 按顺序保留系统提示、用户输入、Assistant 回复和工具结果<br>不建图、不注入 Graph、不按最近两轮裁剪；仍受模型上下文长度限制<br>报告生成后自动进行同一套 LLM Judge 评分'
    :$('#launch-mode').value==='bcg_no_retrieval'
      ?'单 Agent BCG w/o Retrieval · 最近 2 轮 · 永久保留用户输入<br>注入当前 Graph 的全部节点和边，不检索、不按 6,600 字符裁剪；仍受模型上下文长度限制<br>建图：gpt-6-luna / none · LiteLLM · 报告生成后自动进行同一套 LLM Judge 评分'
      :'单 Agent BCG · 最近 2 轮 · Graph 6,600 字符<br>建图：gpt-6-luna / none · LiteLLM<br>报告生成后自动进行 LLM Judge 评分';
});
$('#launch-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=event.submitter;button.disabled=true;$('#launch-error').textContent='';
  try{
    const form=new FormData(event.target);
    if(!taskPicker)throw new Error('题目选择器尚未加载，请稍后再试。');
    const ids=taskPicker.value();
    const tasks={drb2:ids};
    const body={title:form.get('title'),profile:form.get('profile'),mode:form.get('mode'),tasks};
    for(const key of ['workers','timeout','max_turns','search_calls','fetch_calls'])body[key]=Number(form.get(key));
    const run=await api('runs',{method:'POST',body});selected=run.id;history.replaceState(null,'','?run='+selected);event.target.elements.title.value='';suggestedRunTitle='';$('#launch-dialog').close();notice(`${run.number} 已加入队列`);await refresh();
  }catch(error){$('#launch-error').textContent=error.message;}finally{button.disabled=false;}
});
$('#cancel-form').addEventListener('submit',async event=>{
  event.preventDefault();event.submitter.disabled=true;$('#cancel-error').textContent='';
  try{await api(`runs/${cancelTarget}/cancel`,{method:'POST',body:{}});event.target.reset();$('#cancel-dialog').close();notice('已请求停止；正在保留记录并关闭本次专用 Graph 服务');await refresh();}
  catch(error){$('#cancel-error').textContent=error.message;}finally{event.submitter.disabled=false;}
});
try{
  settings=await api('settings');
  for(const option of $('#launch-mode').options)option.disabled=!settings.context_modes?.includes(option.value);
  taskPicker=mountTaskPicker($('#launch-task-picker'),{
    defaults:settings.default_tasks.drb2,limit:settings.max_tasks,catalog:settings.task_catalog?.drb2,
  });
  $('#launch-max-tasks').textContent=String(settings.max_tasks);
  $('#launch-profile').innerHTML=settings.profiles.map(p=>`<option value="${escape(p.id)}">${escape(p.label)}</option>`).join('');
  if(settings.profiles.length)$('#launch-profile').dispatchEvent(new Event('change'));
  if(storedToken()){try{({user}=await api('session'));}catch{logout();}}
  if(user){$('#identity-button').textContent=user.name;await refresh();}
  else{$('#run-content').innerHTML='<section class="run-heading"><div><h1>测试与标注，集中在一个工作区</h1><p>填写昵称后即可启动或停止测试、查看运行结果、阅读轨迹并独立标注。</p></div></section>';identity();}
}catch(error){$('#connection').textContent='服务不可用';notice(error.message);}
setInterval(()=>{if(!document.hidden)refresh();},3000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});

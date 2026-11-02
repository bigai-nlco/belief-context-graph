import {api,benchmarkLabel} from './shared.mjs';

const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)} / 100`:'—';
const verdicts={'1':'满足标准','0':'未满足标准','-1':'命中受限引用'};

function evidenceMarkup(item){
  // Keep recorded quotations verbatim, even if a legacy zero-score item has one.
  if(typeof item.evidence==='string'&&item.evidence.trim())
    return `<blockquote data-evidence-status="provided">${escape(item.evidence)}</blockquote>`;
  if(item.score===0)
    return '<p class="score-no-evidence" data-evidence-status="not-required">0 分项按评分规则不提供原文证据。</p>';
  if(item.score===1||item.score===-1)
    return '<p class="score-evidence-missing" data-evidence-status="missing">证据缺失：该项按评分规则应提供报告原句，但评分记录中未提供。</p>';
  return '<p class="score-no-evidence" data-evidence-status="unrecorded">评分记录中未提供原文证据。</p>';
}

export function scoreItemsMarkup(data,dimension='all',score='all'){
  let count=0;
  const groups=data.dimensions.filter(d=>dimension==='all'||d.id===dimension).map(d=>{
    const items=d.items.map((item,index)=>({...item,index:index+1}))
      .filter(item=>score==='all'||String(item.score)===score);
    count+=items.length;
    if(!items.length)return '';
    return `<section class="score-dimension"><h3>${escape(d.label)} <span>${items.length} 项</span></h3>${items.map(item=>`
      <article class="score-criterion">
        <div class="score-criterion-heading"><span class="score-item-index">${item.index}</span><h4>${escape(item.criterion)}</h4><span class="score-verdict" data-score="${escape(item.score)}">${escape(item.score)} · ${escape(verdicts[item.score]||'未评分')}</span></div>
        <div class="score-explanation"><h5>评分理由</h5><p>${escape(item.reason||'未记录理由')}</p></div>
        <div class="score-explanation"><h5>报告原文证据</h5>${evidenceMarkup(item)}</div>
      </article>`).join('')}</section>`;
  }).join('');
  return {count,html:groups||'<p class="score-empty">没有符合当前筛选条件的评分项。</p>'};
}

export function scoreDetailsMarkup(data){
  const total=data.dimensions.reduce((n,d)=>n+d.items.length,0);
  const date=data.judged_at?new Date(data.judged_at).toLocaleString('zh-CN',{hour12:false}):'—';
  return `
    ${data.status==='judge_error'?'<p class="score-error" role="status">本次评分未完成，未产生完整分数。这里只展示已保存的评分项，不会自动重新评分。</p>':''}
    <p class="score-metadata">Judge ${escape(data.model||'未记录')} · 思考强度 ${escape(data.reasoning_effort||'未记录')}<br>评分时间 ${escape(date)}${Number.isFinite(data.duration_seconds)?` · 耗时 ${data.duration_seconds.toFixed(1)} 秒`:''}</p>
    <div class="score-overview"><div><span>总分</span><strong>${percent(data.score)}</strong></div>${data.dimensions.map(d=>`<div><span>${escape(d.label)}</span><strong>${percent(d.score)}</strong></div>`).join('')}</div>
    <p class="score-legend">单项评分：1 满足标准 · 0 未满足标准 · −1 命中受限引用。总分为得 1 分的项目占全部评分项的比例。<br>原文证据由 Judge 从 Agent 最终报告中摘录，按每条评分项记录。0 分项按规则留空；1 分和 −1 分项应提供报告原句。</p>
    <div class="score-filters">
      <label><span>评分维度</span><select data-score-dimension><option value="all">全部维度（${total}）</option>${data.dimensions.map(d=>`<option value="${escape(d.id)}">${escape(d.label)}（${d.items.length}）</option>`).join('')}</select></label>
      <label><span>单项得分</span><select data-score-filter><option value="all">全部得分</option><option value="1">1 · 满足标准</option><option value="0">0 · 未满足标准</option><option value="-1">−1 · 命中受限引用</option></select></label>
      <span class="score-filter-count" role="status" aria-live="polite">显示 ${total} 项</span>
    </div>
    <div class="score-items">${scoreItemsMarkup(data).html}</div>`;
}

export function mountScoreDetails(dialog,{load=api}={}){
  const content=dialog.querySelector('#score-dialog-content');
  const subtitle=dialog.querySelector('#score-dialog-subtitle');
  let request=0,current=null;
  dialog.addEventListener('close',()=>{request++;current=null;content.replaceChildren();content.setAttribute('aria-busy','false');});
  const fetchDetails=async()=>{
    const selection=current,ticket=++request;
    if(!selection)return;
    content.setAttribute('aria-busy','true');
    content.innerHTML='<div class="score-loading" role="status"><span class="ai-review-spinner" aria-hidden="true"></span>正在读取评分详情…</div>';
    try{
      const data=await load(`runs/${selection.run}/tasks/${encodeURIComponent(selection.benchmark)}/${encodeURIComponent(selection.task)}/score`);
      if(ticket!==request||!dialog.open)return;
      content.innerHTML=scoreDetailsMarkup(data);
      const filter=()=>{
        const result=scoreItemsMarkup(data,content.querySelector('[data-score-dimension]').value,content.querySelector('[data-score-filter]').value);
        content.querySelector('.score-items').innerHTML=result.html;
        content.querySelector('.score-filter-count').textContent=`显示 ${result.count} 项`;
      };
      content.querySelectorAll('select').forEach(select=>select.addEventListener('change',filter));
    }catch(error){
      if(ticket!==request||!dialog.open)return;
      content.innerHTML=`<div class="score-error" role="alert"><p>${escape(error.message||'无法读取评分详情')}</p><button type="button" class="button" data-reload-score>重新加载</button></div>`;
      content.querySelector('[data-reload-score]').addEventListener('click',fetchDetails);
    }finally{
      if(ticket===request)content.setAttribute('aria-busy','false');
    }
  };
  return {open(selection){
    current=selection;
    subtitle.textContent=`${selection.number} · ${benchmarkLabel(selection.benchmark)} · Task ${selection.task}`;
    content.scrollTop=0;
    if(!dialog.open)dialog.showModal();
    return fetchDetails();
  }};
}

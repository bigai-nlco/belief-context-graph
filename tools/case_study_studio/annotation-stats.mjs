import {api,storedToken} from './shared.mjs';

const run=Number(new URLSearchParams(location.search).get('run'));
const $=selector=>document.querySelector(selector);
let snapshotHash='',busy=false;
let viewVersion=0,summaryTimer=null,summaryJob=null,summaryMarkup='',summaryPosting=false,hasAnnotations=false;
const status=$('#stats-feedback'),placeholder=$('#stats-placeholder'),frame=$('#stats-report');
for(const id of ['#stats-back','#stats-return'])$(id).href=`./?run=${run}`;

function message(text){status.textContent=text;status.hidden=!text;}
function putSummary(){
  const doc=frame.contentDocument,main=doc?.querySelector('main');
  if(!main)return;
  let host=doc.querySelector('#run-summary-host');
  if(!host){host=doc.createElement('div');host.id='run-summary-host';main.insertBefore(host,main.querySelector('.toc'));}
  if(host.innerHTML!==summaryMarkup)host.innerHTML=summaryMarkup;
  if(summaryJob?.summary_style){
    let style=doc.querySelector('#run-summary-style');
    if(!style){style=doc.createElement('style');style.id='run-summary-style';doc.head.append(style);}
    style.textContent=summaryJob.summary_style;
  }
}
frame.addEventListener('load',putSummary);
function summaryStatus(text,{pending=false,retry=false,generate=false}={}){
  $('#stats-summary-status').hidden=false;$('#summary-status-text').textContent=text;
  $('#summary-spinner').hidden=!pending;$('#summary-retry').hidden=!retry;$('#summary-retry').disabled=summaryPosting;
  $('#summary-generate').hidden=!generate;$('#summary-generate').disabled=summaryPosting;
}
function showSummaryJob(job){
  summaryJob=job;
  const pending=['queued','running'].includes(job.status);
  if(job.status==='completed'&&job.summary_html){
    summaryMarkup=job.summary_html;
    summaryStatus('AI 问题总结已完成 · GPT-6.1-Sol / high · 相同输入复用已保存结果');
  }else if(pending){
    summaryMarkup='<section class="rules"><h2>Agent 未充分利用 BCG 的问题总结</h2><p class="note">GPT-6.1-Sol / high 正在后台分析当前标注。统计图表已经可以查看，关闭页面不影响总结生成。</p></section>';
    summaryStatus(`AI 问题总结${job.status==='queued'?'排队中':'生成中'} · 已完成 ${job.completed_cohorts} / ${job.cohorts} 个标注组 · GPT-6.1-Sol / high`,{pending:true});
  }else if(['failed','interrupted'].includes(job.status)){
    summaryMarkup='';summaryStatus(job.error||'AI 总结未完成，确定性统计不受影响。',{retry:true});
  }
  putSummary();
}
async function pollSummary(key,version){
  if(version!==viewVersion)return;
  try{
    const job=await api(`runs/${run}/annotation-summary/${key}`);
    if(version!==viewVersion)return;
    showSummaryJob(job);
    if(['queued','running'].includes(job.status))summaryTimer=setTimeout(()=>pollSummary(key,version),3000);
  }catch(error){
    if(version!==viewVersion)return;
    summaryStatus('暂时无法读取 AI 总结进度，稍后自动重试；统计仍可查看。');
    summaryTimer=setTimeout(()=>pollSummary(key,version),5000);
  }
}
async function loadSummary(version=viewVersion){
  if(version!==viewVersion||!hasAnnotations)return;
  summaryStatus('正在读取已有 AI 总结…',{pending:true});
  try{
    const result=await api(`runs/${run}/annotation-summary?statistics_sha=${encodeURIComponent(snapshotHash)}`);
    if(version!==viewVersion)return;
    if(result.job){
      showSummaryJob(result.job);
      if(['queued','running'].includes(result.job.status))summaryTimer=setTimeout(()=>pollSummary(result.job.input_key,version),3000);
    }else{
      summaryJob=null;summaryMarkup='';putSummary();
      summaryStatus('统计已生成。点击“生成 AI 总结”后调用 GPT-6.1-Sol / high，消耗 API 额度。',{generate:true});
    }
  }catch(error){
    if(version!==viewVersion)return;
    summaryStatus(`无法读取已有 AI 总结：${error.message}。请刷新统计重试。`);
  }
}
async function startSummary(retry=false){
  if(!hasAnnotations||summaryPosting||busy)return;
  clearTimeout(summaryTimer);const version=viewVersion,sha=snapshotHash;
  summaryPosting=true;summaryStatus('正在准备固定格式的标注 Context · GPT-6.1-Sol / high',{pending:true});
  try{
    const job=await api(`runs/${run}/annotation-summary`,{method:'POST',body:{statistics_sha:sha,retry}});
    if(version!==viewVersion)return;
    showSummaryJob(job);
    if(job.status==='completed'||['queued','running'].includes(job.status))await pollSummary(job.input_key,version);
  }catch(error){
    if(version!==viewVersion)return;
    summaryStatus(error.message,{retry:true});
  }finally{
    summaryPosting=false;$('#summary-retry').disabled=false;$('#summary-generate').disabled=false;
    if(version!==viewVersion&&hasAnnotations)void loadSummary(viewVersion);
  }
}
$('#summary-generate').addEventListener('click',()=>startSummary());
$('#summary-retry').addEventListener('click',()=>startSummary(true));
async function refresh(){
  if(busy)return;
  busy=true;$('#stats-refresh').disabled=true;$('#stats-download').disabled=true;
  $('#stats-refresh').textContent='统计中…';$('.stats-body').setAttribute('aria-busy','true');message('');
  try{
    if(!Number.isSafeInteger(run)||run<1)throw new Error('运行编号无效，请从运行页面进入。');
    const result=await api(`runs/${run}/annotation-stats`);
    viewVersion++;clearTimeout(summaryTimer);summaryJob=null;summaryMarkup='';
    snapshotHash=result.stats.input_sha256;
    hasAnnotations=result.stats.cohorts.length>0;
    frame.srcdoc=result.report_html;frame.hidden=false;placeholder.hidden=true;
    document.title=`${result.stats.run.number} · 标注统计`;
    if(hasAnnotations)void loadSummary(viewVersion);
    else summaryStatus('暂无已完成标注，不调用 AI 总结模型。');
  }catch(error){
    message(error.status===401?'请先返回运行中心填写昵称，再打开标注统计。':error.message);
    if(frame.hidden){
      placeholder.querySelector('.stats-spinner').hidden=true;
      placeholder.querySelector('h1').textContent='暂时无法显示统计';
      placeholder.querySelector('p').textContent='已有标注不受影响。可以返回运行中心或点击刷新重试。';
    }
  }finally{
    busy=false;$('#stats-refresh').disabled=false;$('#stats-refresh').textContent='刷新统计';
    $('#stats-download').disabled=!snapshotHash;$('.stats-body').setAttribute('aria-busy','false');
  }
}
$('#stats-refresh').addEventListener('click',refresh);
$('#stats-download').addEventListener('click',async()=>{
  if(!snapshotHash||busy)return;
  busy=true;$('#stats-download').disabled=true;$('#stats-refresh').disabled=true;
  $('#stats-download').textContent='正在打包…';message('');
  try{
    const summaryQuery=summaryJob?.status==='completed'?'?summary='+encodeURIComponent(summaryJob.input_key):'';
    const response=await fetch(`/api/runs/${run}/annotation-stats/${snapshotHash}/download${summaryQuery}`,{
      headers:{Authorization:'Bearer '+storedToken()},cache:'no-store'});
    if(!response.ok)throw new Error((await response.json()).error||'下载失败');
    const url=URL.createObjectURL(await response.blob()),link=document.createElement('a');
    link.href=url;link.download=`RUN-${String(run).padStart(4,'0')}-annotation-stats.zip`;
    document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
  }catch(error){message(error.message);}
  finally{busy=false;$('#stats-download').disabled=false;$('#stats-refresh').disabled=false;$('#stats-download').textContent='下载图表与数据';}
});
await refresh();

import {loadEvidence} from './evidence-loader.mjs';
import {previewCacheKey,readPreviewCache,writePreviewCache,deletePreviewCache} from './evidence-cache.mjs';

const TOKEN_KEY = 'bcg-studio-personal-token';
const owners = new Map();
export const storedToken = () => {try {return localStorage.getItem(TOKEN_KEY) || '';} catch {return '';}};
export function storeToken(token) {localStorage.setItem(TOKEN_KEY, token);}
export function logout() {localStorage.removeItem(TOKEN_KEY);}
export const benchmarkLabel = id => id === 'drb2' ? 'DeepResearch Bench II' : id;
export const canAnnotateTask = task => task?.status === 'completed' && task.annotatable === true;
export function taskAnnotationStatus(task) {
  const reviews=task.reviews||[],ai=task.ai_review?.status;
  const humanDone=reviews.some(r=>r.status==='reviewed'),aiDone=ai==='completed';
  const completed=humanDone||aiDone;
  const inProgress=!completed&&(reviews.some(r=>r.status==='draft')||['queued','running'].includes(ai));
  const sources=completed?[...(humanDone?['人工']:[]),...(aiDone?['AI']:[])]:[];
  const label=humanDone&&aiDone?'AI 和人工已标注':aiDone?'AI 已标注':humanDone?'人工已标注':inProgress?'标注中':'未标注';
  return {completed,inProgress,label,sources};
}
export function annotationProgress(tasks) {
  const eligible=tasks.filter(canAnnotateTask);
  return {completed:eligible.filter(t=>taskAnnotationStatus(t).completed).length,total:eligible.length};
}
export function defaultRunTitle(date=new Date()) {
  const pad=value=>String(value).padStart(2,'0');
  const stamp=`${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  return `${benchmarkLabel('drb2')} 测试 · ${stamp}`;
}
export async function api(path, {method='GET', body, token=storedToken()}={}) {
  const headers = {};
  if(token) headers.Authorization = 'Bearer ' + token;
  if(body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch('/api/' + path, {method,headers,body:body===undefined?undefined:JSON.stringify(body),cache:'no-store'});
  const data = await response.json();
  if(!response.ok) {const error=new Error(data.error || '请求失败');error.status=response.status;throw error;}
  return data;
}
export function remoteLocation(search=location.search) {
  const q=new URLSearchParams(search);
  if(!q.has('run')) return null;
  const run=q.get('run'),benchmark=q.get('benchmark'),task=q.get('task');
  if(benchmark !== 'drb2') throw new Error('共享页面仅保留 DeepResearch Bench II；其他 Benchmark 的历史文件和标注仍保留在服务器');
  if(!/^\d+$/.test(run) || !/^[A-Za-z0-9_-]{1,80}$/.test(task||'')) throw new Error('运行或任务链接无效');
  if(q.has('review') && q.get('review') !== 'ai') throw new Error('标注类型无效');
  return {run:Number(run),benchmark,task,...(q.get('review')==='ai'?{review:'ai'}:{})};
}
export const casePath = remote => `runs/${remote.run}/tasks/${remote.benchmark}/${encodeURIComponent(remote.task)}`;
export const caseLink = (run,benchmark,task,review) => './index.html?' + new URLSearchParams({run,benchmark,task,...(review==='ai'?{review}:{})});

// Browser source IDs change on every import. Bind only by original file hash,
// path, JSONL line and call, never by task ID or call alone.
export function bindAIAnnotation(c, annotation, model) {
  const result=structuredClone(annotation);
  if(model) result.reviewer=model;
  for(const event of result.events.flatMap(e=>[e,...(e.relatedOccurrences||[])])) {
    const p=event.sourceLocator;
    const files=c.files.filter(f=>f.path===p?.path && f.digest===p?.digest);
    const records=c.sources.filter(s=>files.some(f=>f.id===s.fileId)).flatMap(s=>s.records)
      .filter(r=>r.index===p?.index && String(r.call)===String(p?.call) && r.role==='assistant');
    if(records.length!==1) throw new Error('AI 标注与原始证据不匹配，未关联到其他轮次');
    event.sourceRef='record:'+records[0].id;
  }
  return result;
}

function validatePreview(value,remote,sha) {
  const p=value.projection;
  if(!/^[a-f0-9]{64}$/.test(sha||'') || p?.kind!=='review-preview' || p.version!==1 || p.source_sha256!==sha || !Array.isArray(p.omitted_files))
    throw new Error('AI 标注的证据版本已变化，未加载不同版本的结果');
  validateTask(value.c,remote);
}
function validateTask(c,remote) {
  // Routes use the runner's short name; exported results use the adapter's
  // canonical name. Only this explicit pair identifies the same benchmark.
  const benchmarkKey=name=>name==='drb2'?'deepresearch_bench_ii':name;
  if(String(c.taskId)!==String(remote.task)||benchmarkKey(c.benchmark)!==benchmarkKey(remote.benchmark))
    throw new Error('证据与当前任务不匹配');
}
async function fetchEvidence(remote,preview=false) {
  const response=await fetch('/api/'+casePath(remote)+(preview?'/evidence-preview':'/evidence'),{
    headers:{Authorization:'Bearer '+storedToken()},cache:'no-store'});
  if(!response.ok)throw new Error((await response.json()).error||'证据加载失败');
  return response;
}

export async function loadSharedCase(remote, cached=[]) {
  const token=storedToken();
  const isAI=remote.review==='ai';
  const [{user},run,review]=await Promise.all([api('session'),api(`runs/${remote.run}`),api(casePath(remote)+(isAI?'/ai-review':'/annotation'))]);
  if(isAI && review.status!=='completed') throw new Error('AI 标注尚未完成，请返回运行页面查看进度');
  const key=isAI?previewCacheKey(remote,review.evidence_sha):null;
  let value=isAI?await readPreviewCache(key):null,sha=review.evidence_sha;
  if(value) {
    try{validatePreview(value,remote,sha);bindAIAnnotation(value.c,review.annotation,review.model);}
    catch{await deletePreviewCache(key);value=null;}
  }
  if(!value) {
    const response=await fetchEvidence(remote,isAI);
    sha=response.headers.get('X-Evidence-SHA256');
    if(isAI&&sha!==review.evidence_sha)throw new Error('AI 标注的证据版本已变化，未加载不同版本的结果');
    value=await loadEvidence(await response.blob());
    validateTask(value.c,remote);
    if(isAI){
      validatePreview(value,remote,sha);
      bindAIAnnotation(value.c,review.annotation,review.model);
      // Cache evidence only; personal identity and annotations are attached below.
      void writePreviewCache(key,value);
    }
  }
  let c=value.c;
  c.id=`shared-${remote.run}-${remote.benchmark}-${remote.task}-${isAI?'ai':user.id}`;
  c.remote={...remote,user,revision:review.revision,evidence_sha:sha,dirty:false,readOnly:isAI,
    deferredEvidence:isAI&&value.projection.omitted_files.length>0};
  if(isAI) {
    c.annotation=bindAIAnnotation(c,review.annotation,review.model);
    c.remote.ai={model:review.model,reasoning_effort:review.reasoning_effort,usage:review.usage,coverage:review.coverage,finished_at:review.finished_at};
    return {c,run,user};
  }
  if(review.annotation) c.annotation=review.annotation;
  c.annotation.reviewer=user.name;
  const local=typeof cached==='function'?await cached(c.id):cached.find(x=>x.id===c.id);
  const draft=local?.remote?.dirty?local:null;
  if(draft) {
    c=draft;
    c.remote.conflict=c.remote.revision!==review.revision || c.remote.evidence_sha!==sha;
  }
  owners.set(c.id,token);
  return {c,run,user};
}

export async function loadFullSharedEvidence(c) {
  if(!c.remote?.deferredEvidence)return c;
  const response=await fetchEvidence(c.remote);
  if(response.headers.get('X-Evidence-SHA256')!==c.remote.evidence_sha)
    throw new Error('完整 Graph 与标注的证据版本不匹配');
  const value=await loadEvidence(await response.blob()),full=value.c;
  if(value.projection)throw new Error('未获得完整证据，请重试');
  validateTask(full,c.remote);
  if(c.files.some(f=>!full.files.some(g=>f.path===g.path&&f.digest===g.digest)))
    throw new Error('完整 Graph 与已加载的证据不匹配');
  full.id=c.id;
  full.remote={...c.remote,deferredEvidence:false,graphLoading:false,graphError:''};
  full.annotation=bindAIAnnotation(full,c.annotation,c.remote.ai?.model);
  return full;
}

export async function saveSharedCase(c) {
  if(c.remote?.readOnly) throw new Error('AI 标注只读，不会覆盖人工标注');
  if(!owners.has(c.id) || owners.get(c.id)!==storedToken()) throw new Error('登录身份已变化。请先导出草稿备份，再重新打开任务');
  const annotation=structuredClone(c.annotation),before=JSON.stringify(annotation);
  const result=await api(casePath(c.remote)+'/annotation',{method:'PUT',body:{
    revision:c.remote.revision,evidence_sha:c.remote.evidence_sha,annotation}});
  c.remote.revision=result.revision;
  c.remote.dirty=JSON.stringify(c.annotation)!==before;
  c.remote.conflict=false;
  return result;
}

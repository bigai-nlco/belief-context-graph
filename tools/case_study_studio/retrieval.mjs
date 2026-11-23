// Read-only retrieval inspection. Scores always come from the recorded invocation.
import {overviewGraph, trajectoryLayout, timelineSize, taskSetIndex, memberLabels, taskRootId, taskDiagramLayout, workflowDetails} from './model.mjs';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fmt = value => finite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 4}) : '未记录';
const text = node => node.belief ?? node.decision ?? node.content ?? '';
const parent = path => (path || '').replace(/\\/g, '/').split('/').slice(0, -1).join('/');

export function retrievalRounds(c) {
  return c.contexts.filter(ctx => ctx.kind === 'injection_audit' && ctx.raw?.call !== undefined);
}

export function retrievalSnapshot(c, ctx) {
  if (!ctx) return {graph: null, status: 'missing-audit'};
  const path = id => c.files.find(file => file.id === id)?.path;
  const owner = parent(path(ctx.fileId));
  let candidates = c.graphs.filter(graph => graph.scope === 'turns' && graph.snapshotScope === 'acknowledged_update');
  // A graph identity is required across directories; never substitute a final graph.
  candidates = candidates.filter(graph => ctx.graphId ? graph.graphId === ctx.graphId : parent(path(graph.fileId)) === owner);
  const owned = candidates.filter(graph => parent(path(graph.fileId)) === owner);
  if (owned.length) candidates = owned;
  const turns = ctx.raw?.graph_n_turns_ingested;
  candidates = candidates.filter(graph => finite(graph.call) && graph.call <= Number(ctx.call) && (
    finite(turns) ? graph.nTurns === turns : graph.call === Number(ctx.call) || unchangedSince(c, ctx, graph.call)
  ));
  // Duplicate exports are fine; distinct snapshots with the same identity are not.
  const unique = new Map(candidates.map(graph => [JSON.stringify([graph.nTurns, graph.nodes, graph.relations]), graph]));
  if (unique.size !== 1) return {graph: null, status: unique.size ? 'ambiguous' : 'missing'};
  const graph = [...unique.values()][0];
  const nodes = new Set(graph.nodes.map(n => n.id)), edges = new Set(graph.relations.map(e => e.id));
  if (ctx.nodeIds.some(id => !nodes.has(id)) || ctx.relationIds.some(id => !edges.has(id))) return {graph: null, status: 'mismatch'};
  return {graph, status: 'matched'};
}

function unchangedSince(c, ctx, call) {
  // Early legacy audits omitted the turn count. Reuse an acknowledged snapshot
  // only if every intervening call explicitly records zero graph writes.
  for (let next = call + 1; next <= Number(ctx.call); next++) {
    const events = retrievalRounds(c).filter(row => row.fileId === ctx.fileId && Number(row.call) === next);
    if (events.length !== 1 || events[0].graphId !== ctx.graphId || events[0].raw.graph_turns_written !== 0 || events[0].fallback) return false;
  }
  return true;
}

export function retrievalRows(graph, ctx) {
  if (!graph) return [];
  const selected = new Set(ctx.nodeIds), selection = ctx.raw?.selector_result || {};
  return graph.nodes.map(node => {
    const recorded = selection.node_scores?.[String(node.id)], replay = selection.score_replay?.node_scores?.[String(node.id)];
    const score = recorded || replay;
    const recordedPrize = recorded?.prize ?? selection.selected_prizes?.[String(node.id)];
    return {node, selected: selected.has(node.id), score, scoreOrigin: recorded ? 'recorded' : replay ? 'replay' : 'missing',
      confidence: score?.confidence ?? node.confidence,
      relevance: recorded?.relevance ?? replay?.relevance,
      prize: recordedPrize ?? replay?.prize,
      relevanceOrigin: finite(recorded?.relevance) ? 'recorded' : finite(replay?.relevance) ? 'replay' : 'missing',
      prizeOrigin: finite(recordedPrize) ? 'recorded' : finite(replay?.prize) ? 'replay' : 'missing',
      scoringSkipped: selection.strategy === 'all' || (!selection.strategy && !selection.node_scores && !ctx.nodeIds.length && !ctx.fallback),
      relevanceApplicable: selection.strategy !== 'confidence_pcst'};
  }).sort((a, b) => Number(b.selected) - Number(a.selected) || (b.prize ?? -1) - (a.prize ?? -1) || a.node.id - b.node.id);
}

function scoreText(row, field) {
  if (field === 'relevance' && !row.relevanceApplicable) return '不适用';
  if (!finite(row[field]) && row.scoringSkipped) return '未计算';
  return fmt(row[field]) + (row[`${field}Origin`] === 'replay' ? '（补算）' : '');
}

const layouts = new WeakMap();
export function graphLayout(graph, ids, iterations = 150) {
  let cache = layouts.get(graph);
  if (!cache) {cache = new Map(); layouts.set(graph, cache);}
  const key = JSON.stringify([iterations, ids]);
  if (cache.has(key)) return cache.get(key);
  const points = ids.map((id, i) => ({id, x: 500 + Math.cos(i * 2.39996) * 24 * Math.sqrt(i), y: 310 + Math.sin(i * 2.39996) * 24 * Math.sqrt(i), vx: 0, vy: 0}));
  const byId = new Map(points.map(p => [p.id, p]));
  const edges = graph.relations.filter(e => e.from_id !== e.to_id && byId.has(e.from_id) && byId.has(e.to_id));
  // Full graph is never truncated. Large graphs use a spatial grid for repulsion.
  for (let step = 0; step < iterations; step++) {
    const grid = new Map();
    for (const p of points) {const k = `${Math.floor(p.x / 100)},${Math.floor(p.y / 100)}`; if (!grid.has(k)) grid.set(k, []); grid.get(k).push(p);}
    for (const p of points) {
      const gx = Math.floor(p.x / 100), gy = Math.floor(p.y / 100);
      for (let x = gx - 1; x <= gx + 1; x++) for (let y = gy - 1; y <= gy + 1; y++) {
        for (const q of grid.get(`${x},${y}`) || []) {
          if (p === q) continue;
          const dx = p.x - q.x, dy = p.y - q.y, d = Math.max(1, Math.hypot(dx, dy));
          const force = Math.min(8, 1700 / (d * d));
          p.vx += dx / d * force; p.vy += dy / d * force;
        }
      }
    }
    for (const edge of edges) {
      const a = byId.get(edge.from_id), b = byId.get(edge.to_id), dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(1, Math.hypot(dx, dy)), force = (d - 85) * .012;
      a.vx += dx / d * force; a.vy += dy / d * force; b.vx -= dx / d * force; b.vy -= dy / d * force;
    }
    for (const p of points) {p.vx = (p.vx + (500 - p.x) * .001) * .65; p.vy = (p.vy + (310 - p.y) * .001) * .65; p.x += p.vx; p.y += p.vy;}
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y);}
  const scale = Math.min(860 / Math.max(1, maxX - minX), 490 / Math.max(1, maxY - minY), 2);
  for (const p of points) {p.x = 500 + (p.x - (maxX + minX) / 2) * scale; p.y = 300 + (p.y - (maxY + minY) / 2) * scale;}
  const result = new Map(points.map(p => [p.id, p])); cache.set(key, result);
  return result;
}

function graphSVG(graph, ctx, rows, state) {
  const displayed = state.retrievalScope === 'selected' ? rows.filter(r => r.selected) : rows;
  if (!displayed.length) return '<p class="notice">本轮没有实际注入的节点。</p>';
  const ids = displayed.map(r => r.node.id).sort((a,b) => a-b), points = graphLayout(graph, ids);
  const selectedEdges = new Set(ctx.relationIds);
  const edges = graph.relations.filter(e => points.has(e.from_id) && points.has(e.to_id) && (state.retrievalScope !== 'selected' || selectedEdges.has(e.id)));
  const radius = displayed.length > 100 ? 7 : 12;
  return `<svg class="retrieval-svg ${displayed.length>100 && state.retrievalLabels!=='on'?'dense':''}" viewBox="0 0 1000 620" role="group" aria-label="本轮图，绿色表示实际注入节点，可点击查看内容">
    <defs><marker id="retrieval-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 10 5 0 10z" fill="#a7b5c0"/></marker><marker id="retrieval-arrow-selected" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 10 5 0 10z" fill="#147d64"/></marker></defs>
    ${edges.sort((a,b)=>Number(selectedEdges.has(a.id))-Number(selectedEdges.has(b.id))).map(e => {
      const a = points.get(e.from_id), b = points.get(e.to_id), dx=b.x-a.x, dy=b.y-a.y, d=Math.max(1,Math.hypot(dx,dy)), selected=selectedEdges.has(e.id);
      return `<line class="retrieval-edge ${selected?'is-selected':''}" data-retrieval-edge="${esc(e.id)}" x1="${a.x+dx/d*radius}" y1="${a.y+dy/d*radius}" x2="${b.x-dx/d*(radius+3)}" y2="${b.y-dy/d*(radius+3)}" marker-end="url(#retrieval-arrow${selected?'-selected':''})"><title>${esc(`B${e.from_id} ${e.type} B${e.to_id}\n${e.note || ''}`)}</title></line>`;
    }).join('')}
    ${displayed.slice().sort((a,b)=>Number(a.selected)-Number(b.selected)).map(row => {const n=row.node, p=points.get(n.id); return `<g class="retrieval-node ${row.selected?'is-selected':''} ${String(n.id)===state.retrievalNode?'is-active':''}" data-retrieval-node="${esc(n.id)}" role="button" tabindex="0" aria-label="${esc(`B${n.id} ${row.selected?'已选':'未选'} ${text(n)}`)}" transform="translate(${p.x},${p.y})"><title>${esc(`B${n.id} ${row.selected?'已选':'未选'}\nconfidence ${fmt(row.confidence)} · relevance ${scoreText(row,'relevance')} · prize ${scoreText(row,'prize')}\n${text(n)}`)}</title><circle r="${radius}"/><text y="${-radius-5}" text-anchor="middle">B${esc(n.id)}</text>${state.retrievalLabels==='on'?`<text class="retrieval-snippet" y="${radius+15}" text-anchor="middle">${esc(String(text(n)).slice(0,26))}${String(text(n)).length>26?'…':''}</text>`:''}</g>`;}).join('')}
  </svg>`;
}

function detail(row, graph, ctx) {
  if (!row) return '<p class="notice">点击节点查看完整正文和评分。</p>';
  const n=row.node, score=row.score, selectedEdges=new Set(ctx.relationIds);
  const edges=graph.relations.filter(e=>e.from_id===n.id || e.to_id===n.id);
  return `<article class="retrieval-detail" data-detail-node="${esc(n.id)}">
    <div class="retrieval-detail-heading"><strong>B${esc(n.id)}</strong><span class="retrieval-badge ${row.selected?'selected':''}">${row.selected?'已注入':'未选中'}</span></div>
    <dl class="retrieval-node-metrics"><div><dt>confidence</dt><dd>${fmt(row.confidence)}</dd></div><div><dt>relevance</dt><dd>${scoreText(row,'relevance')}</dd></div><div><dt>prize</dt><dd>${scoreText(row,'prize')}</dd></div></dl>
    ${score?.pinned_input?'<p class="notice">原始输入已保留，节点奖励置零；仍可用于桥接。</p>':''}
    ${score?.feasible===false?'<p class="notice amber">该节点单独渲染已超过本轮预算。</p>':''}
    <p class="retrieval-node-text">${esc(text(n))}</p>
    <div class="small muted">confidence 原值：${esc(n.confidence ?? '未记录')} · 奖励中的 confidence 系数：${fmt(score?.confidence_factor)}${row.scoreOrigin==='replay'?'（补算）':''}</div>
    ${row.relevanceOrigin==='replay'||row.prizeOrigin==='replay'?'<p class="notice">标注“补算”的数值由当轮快照和问题事后重算；已有原始 prize 优先展示，选中状态仍来自当时记录。</p>':''}
    <details><summary>来源与原始属性</summary><pre>${esc(JSON.stringify(n,null,2))}</pre></details>
    <details open><summary>关联关系 · ${edges.length}</summary><div class="retrieval-relations">${edges.map(e=>`<p class="${selectedEdges.has(e.id)?'selected':''}"><strong>${selectedEdges.has(e.id)?'已选 · ':''}B${esc(e.from_id)} ${esc(e.type)} B${esc(e.to_id)}</strong><br>${esc(e.note || '')}</p>`).join('') || '<p>没有关联关系。</p>'}</div></details>
  </article>`;
}

export function renderRetrieval(c, state) {
  const rounds=retrievalRounds(c);
  if (!rounds.length) return '<div class="notice">尚无逐轮选图记录。请导入同次运行的 requests.jsonl、graph_snapshots.jsonl 和 Agent 轨迹，或完整 Case 运行包。</div>';
  const ctx=rounds.find(r=>r.id===state.context) || rounds[0]; state.context=ctx.id;
  const index=rounds.indexOf(ctx), {graph,status}=retrievalSnapshot(c,ctx), selection=ctx.raw.selector_result || {};
  const rows=retrievalRows(graph,ctx), active=rows.find(r=>String(r.node.id)===state.retrievalNode) || rows[0];
  state.retrievalNode=active?String(active.node.id):'';
  const budget=ctx.raw.graph_chars_budget ?? selection.graph_chars_budget;
  const noRetrieval=(selection.strategy || ctx.raw.retrieval_strategy)==='all';
  const embedding=selection.embedding || {}, cache=embedding.persistent_cache || {};
  const file=c.files.find(f=>f.id===ctx.fileId)?.path || '';
  const filter=(state.retrievalQuery || '').toLowerCase();
  const filtered=rows.filter(r=>(state.retrievalFilter!=='selected'||r.selected)&&(state.retrievalFilter!=='unselected'||!r.selected)&&`B${r.node.id} ${text(r.node)}`.toLowerCase().includes(filter));
  const metric=(label,value)=>`<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`;
  const notices={missing:'缺少对应时刻的已确认快照，不使用最终图代替。', ambiguous:'同一调用对应多份不同快照，无法确定当时图。', mismatch:'快照缺少审计中选中的节点或关系，请核对文件是否来自同次运行。'};
  return `<section class="retrieval-view" data-copy-lock>
    <div class="retrieval-heading"><div><span class="eyebrow">RETRIEVAL INSPECTOR</span><h3>沿轨迹查看每次选图</h3><p>完整图、实际注入子图与当轮评分。</p></div><span class="retrieval-badge">${esc(noRetrieval?'BCG w/o Retrieval':selection.strategy || ctx.raw.retrieval_strategy || '旧版检索')}</span></div>
    <nav class="retrieval-navigation" aria-label="逐轮选图切换"><button class="button" data-retrieval-step="-1" ${index===0?'disabled':''}>← 上一轮</button><label>模型调用<select id="retrieval-round">${rounds.map((r,i)=>`<option value="${esc(r.id)}" ${r.id===ctx.id?'selected':''}>${i+1}. call ${esc(r.call)} · ${esc(r.graphId || c.files.find(f=>f.id===r.fileId)?.path || '')}</option>`).join('')}</select></label><button class="button" data-retrieval-step="1" ${index===rounds.length-1?'disabled':''}>下一轮 →</button></nav>
    <p class="retrieval-provenance">${index+1} / ${rounds.length} 次调用 · ${esc(file)} · Graph 已确认 turns：${esc(graph?.nTurns ?? ctx.raw.graph_n_turns_ingested ?? '未记录')}</p>
    ${ctx.fallback?`<p class="notice amber">本轮回退到原始上下文：${esc(ctx.fallback)}。下方高亮以实际注入 ID 为准。</p>`:!ctx.nodeIds.length?'<p class="notice">本轮未注入子图，可能尚未发生历史裁剪。</p>':''}
    ${noRetrieval?'<p class="retrieval-legacy">全图注入模式：跳过检索与候选评分，不按字符预算筛选节点和边。</p>':selection.score_replay?`<p class="retrieval-legacy">已补算 ${Object.keys(selection.score_replay.node_scores || {}).length} 个候选节点的评分，标记为“补算”。选点、边、子图收益和注入文本仍为原始记录。${selection.score_replay.comparison?.matches_within_tolerance===false?' 补算 prize 与部分原记录有差异，不能当作历史原值。':''}</p>`:selection.strategy&&!selection.node_scores?'<p class="retrieval-legacy">旧记录未保存全部候选评分。缺失值显示“未记录”，不会重新计算后冒充当时结果。</p>':!selection.strategy?'<p class="retrieval-legacy">本轮没有评分记录；未发生检索的调用显示“未计算”。</p>':''}
    <dl class="retrieval-metrics">${metric('完整图',graph?`${graph.nodes.length} 点 / ${graph.relations.length} 边`:'快照未匹配')}${metric('实际选中',`${ctx.nodeIds.length} 点 / ${ctx.relationIds.length} 边`)}${metric('子图收益',fmt(selection.objective))}${metric('图字符 / 上限',`${fmt(ctx.raw.graph_chars)} / ${noRetrieval?'无上限':fmt(budget)}`)}${metric('每条边成本',fmt(selection.parameters?.edge_cost))}${metric('上下文准备（秒）',fmt(finite(ctx.raw.duration_ms) ? ctx.raw.duration_ms / 1000 : null))}</dl>
    <details class="retrieval-calculation"><summary>查看计算参数、问题与向量缓存</summary><p>query_pcst：prize = relevance × [ε + (1 − ε) × confidence]；子图收益 = 节点奖励之和 − 边成本 × 所选边数。值来自该轮审计。</p><dl class="retrieval-metrics">${metric('ε',fmt(selection.parameters?.confidence_floor))}${metric('求解次数',fmt(selection.solver_calls))}${metric('长度惩罚',fmt(selection.size_penalty))}${metric('裁掉叶子',fmt(selection.leaves_pruned))}${metric('读回缓存文本',fmt(cache.loaded_texts))}${metric('新编码节点分块',fmt(embedding.new_node_chunks_encoded))}</dl><strong>问题</strong><pre>${esc(selection.query?.question || '未记录')}</pre><strong>当前焦点</strong><pre>${esc(selection.query?.focus || '未记录 / 使用原问题')}</pre><details><summary>原始选图审计</summary><pre>${esc(JSON.stringify(ctx.raw,null,2))}</pre></details></details>
    ${graph?`<div class="retrieval-controls"><div role="group" aria-label="图显示范围"><button class="button ${state.retrievalScope!=='selected'?'primary':''}" data-retrieval-scope="all">完整图 + 高亮选点</button><button class="button ${state.retrievalScope==='selected'?'primary':''}" data-retrieval-scope="selected">只看选中子图</button></div><label><input id="retrieval-labels" type="checkbox" ${state.retrievalLabels==='on'?'checked':''}> 在图上显示正文摘要</label></div>
    <div class="retrieval-legend"><span class="selected">● 已注入</span><span>○ 未选中</span><span class="active">◎ 当前查看</span><span>悬停读正文 · 点击看详情 · 拖动平移</span></div>
    <div class="retrieval-graph-layout"><div class="retrieval-canvas"><div class="retrieval-zoom"><button class="button" data-retrieval-zoom="in" aria-label="放大图">＋</button><button class="button" data-retrieval-zoom="out" aria-label="缩小图">−</button><button class="button" data-retrieval-zoom="reset">复位</button></div>${graphSVG(graph,ctx,rows,state)}</div>${detail(active,graph,ctx)}</div>
    <div class="retrieval-table-controls"><label class="search-input"><input id="retrieval-search" aria-label="搜索选图节点" placeholder="搜索节点编号或正文" value="${esc(state.retrievalQuery)}"></label><select id="retrieval-filter" aria-label="按是否选中筛选">${[['all','全部节点'],['selected','已选节点'],['unselected','未选节点']].map(([v,label])=>`<option value="${v}" ${state.retrievalFilter===v?'selected':''}>${label}</option>`).join('')}</select><span>${filtered.length} 个节点</span></div>
    <div class="retrieval-table-wrap"><table class="retrieval-table"><thead><tr><th>节点</th><th>状态</th><th>confidence</th><th>relevance</th><th>prize</th><th>节点内容</th></tr></thead><tbody>${filtered.slice(0,500).map(r=>`<tr class="${r.selected?'is-selected':''}"><td><button data-retrieval-node="${esc(r.node.id)}">B${esc(r.node.id)}</button></td><td>${r.selected?'已选':'未选'}</td><td>${fmt(r.confidence)}</td><td>${scoreText(r,'relevance')}</td><td>${scoreText(r,'prize')}</td><td><button class="retrieval-text-button" data-retrieval-node="${esc(r.node.id)}">${esc(text(r.node))}</button></td></tr>`).join('')}</tbody></table></div>${filtered.length>500?'<p class="small muted">列表显示前 500 个匹配节点，可搜索缩小范围；图中保留全部节点。</p>':''}`:`<p class="notice amber">${notices[status] || '尚无对应图快照。'}</p>`}
    <details class="retrieval-injected"><summary>当轮实际注入文本</summary><pre>${esc(ctx.text || '本轮没有图注入文本')}</pre></details>
  </section>`;
}

export function bindRetrievalPan(root) {
  const svg=root.querySelector('.retrieval-svg'); if(!svg || svg.dataset.panBound)return;
  svg.dataset.panBound='true';
  let drag=null, moved=false;
  svg.addEventListener('pointerdown',event=>{if(event.button!==0 || event.target.closest('[data-retrieval-node]'))return;drag={x:event.clientX,y:event.clientY,box:[svg.viewBox.baseVal.x,svg.viewBox.baseVal.y,svg.viewBox.baseVal.width,svg.viewBox.baseVal.height]};moved=false;svg.setPointerCapture(event.pointerId);});
  svg.addEventListener('pointermove',event=>{if(!drag)return;const rect=svg.getBoundingClientRect(),scale=Math.max(drag.box[2]/rect.width,drag.box[3]/rect.height);const dx=(event.clientX-drag.x)*scale,dy=(event.clientY-drag.y)*scale;moved ||= Math.abs(dx)+Math.abs(dy)>3;svg.setAttribute('viewBox',`${drag.box[0]-dx} ${drag.box[1]-dy} ${drag.box[2]} ${drag.box[3]}`);});
  svg.addEventListener('pointerup',()=>{drag=null;});svg.addEventListener('pointercancel',()=>{drag=null;});
  svg.addEventListener('click',event=>{if(moved){event.stopPropagation();moved=false;}});
}

export function zoomRetrieval(root, action) {
  const svg=root.querySelector('.retrieval-svg');if(!svg)return;
  if(action==='reset'){svg.setAttribute('viewBox','0 0 1000 620');return;}
  const b=svg.viewBox.baseVal, scale=action==='in'?.75:1/.75, width=Math.max(80,Math.min(5000,b.width*scale)), height=width*.62;
  svg.setAttribute('viewBox',`${b.x+(b.width-width)/2} ${b.y+(b.height-height)/2} ${width} ${height}`);
}

// The BCG keeps the Studio's original point/edge encoding. Only its spatial
// arrangement uses trajectory rows. TSG membership is rendered separately.
const preparedTimelines=new WeakMap();
function timelineDrawing(model) {
  if(preparedTimelines.has(model))return preparedTimelines.get(model);
  const layout=trajectoryLayout(model), groups=new Map(), routes=new Map(), half=timelineSize.radius;
  for(const e of model.relations){
    const a=layout.positions.get(e.from_id),b=layout.positions.get(e.to_id);
    let x,y,u,v,cx,cy,du,dv;
    if(a.x===b.x){
      const sign=(a.y<=b.y?1:-1),out=34+Math.min(18,Math.abs(b.y-a.y)*.12)+Math.abs(e.lane)*7;
      x=a.x+sign*half;u=x;y=a.y;v=b.y;
      if(a===b){x=a.x+half*.8;u=x;y-=half*.6;v+=half*.6;}
      cx=x+sign*out;du=u+sign*out;cy=y+e.lane*8;dv=v+e.lane*8;
    }else{
      const sign=Math.sign(b.x-a.x);
      x=a.x+sign*half;u=b.x-sign*half;y=a.y;v=b.y;
      const reach=(u-x)*.43;
      const bow=-(Math.abs(b.x-a.x)>timelineSize.column?25:(y===v?9:0))+e.lane*sign*12;
      cx=x+reach;du=u-reach;cy=y+bow;dv=v+bow;
    }
    const p={x,y,cx,cy,du,dv,u,v,left:Math.min(x,cx,du,u),right:Math.max(x,cx,du,u),
      top:Math.min(y,cy,dv,v),bottom:Math.max(y,cy,dv,v)};
    routes.set(e,p);
    const style=e.type==='contradicts';
    if(!groups.has(style))groups.set(style,[]);
    groups.get(style).push(p);
  }
  const scene={...layout,groups,routes};
  preparedTimelines.set(model,scene);return scene;
}

function paintTimeline(ctx,scene,model,view,width,height,k,palette){
  const {radius,column,axis}=timelineSize,plot=Math.max(1,height-axis);
  const left=view.x-width/(2*k),right=view.x+width/(2*k),top=view.y-plot/(2*k),bottom=view.y+plot/(2*k);
  const active=model.byId.get(view.selected);
  ctx.save();ctx.beginPath();ctx.rect(0,0,width,plot);ctx.clip();
  ctx.translate(-left*k,-top*k);ctx.scale(k,k);
  const path=(p,arrows)=>{
    if(p.right<left-8||p.left>right+8||p.bottom<top-8||p.top>bottom+8)return;
    ctx.moveTo(p.x,p.y);ctx.bezierCurveTo(p.cx,p.cy,p.du,p.dv,p.u,p.v);
    if(arrows && k>=.48){
      const length=Math.hypot(p.u-p.du,p.v-p.dv)||1,dx=(p.u-p.du)/length,dy=(p.v-p.dv)/length;
      ctx.moveTo(p.u-6*dx+3*dy,p.v-6*dy-3*dx);ctx.lineTo(p.u,p.v);ctx.lineTo(p.u-6*dx-3*dy,p.v-6*dy+3*dx);
    }
  };
  function stroke(contradicts,curves,selected=false){
    ctx.strokeStyle=selected?palette.focus:contradicts?palette.red:palette.line;
    ctx.setLineDash(contradicts && k>=.34?[4,3]:[]);ctx.lineWidth=Math.max(.55/k,selected?1.6:1);
    ctx.beginPath();for(const p of curves)path(p,true);ctx.stroke();
  }
  ctx.globalAlpha=active?.id ? .24 : .88;
  for(const [style,curves] of scene.groups)stroke(style,curves);
  if(active){
    ctx.globalAlpha=1;
    for(const e of model.incident.get(active.id))stroke(e.type==='contradicts',[scene.routes.get(e)],true);
  }
  ctx.globalAlpha=1;
  const point=(p,selected)=>{
    const r=Math.max(.75/k,radius);
    if(p.x+60+r<left||p.x-r>right||p.y+r<top||p.y-r>bottom)return;
    ctx.fillStyle=model.byId.get(p.id).kind==='missing'?palette.line:palette.ink;
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();
    if(selected){ctx.strokeStyle=palette.focus;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,r+3,0,Math.PI*2);ctx.stroke();}
    if(k>=.64){
      ctx.font=(selected?'bold ':'')+'12px system-ui';ctx.lineWidth=3;
      ctx.strokeStyle=palette.surface;ctx.strokeText(p.id,p.x+r+4,p.y+4);
      ctx.fillStyle=palette.ink;ctx.fillText(p.id,p.x+r+4,p.y+4);
    }
  };
  ctx.textAlign='left';ctx.setLineDash([]);
  for(const p of scene.positions.values())if(p.id!==view.selected)point(p,false);
  if(active)point(scene.positions.get(active.id),true);
  ctx.restore();
  // Keep the trajectory ruler visible when scrolling down a tall column.
  ctx.fillStyle=palette.surface;ctx.fillRect(0,plot,width,axis);ctx.strokeStyle=palette.line;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,plot+.5);ctx.lineTo(width,plot+.5);ctx.stroke();
  ctx.fillStyle=palette.muted;ctx.font='11px ui-monospace, monospace';ctx.textAlign='center';
  const stride=Math.max(1,Math.ceil(78/(column*k)));
  scene.columns.forEach((c,i)=>{
    if(i%stride!==0||c.x<left||c.x>right)return;
    const x=(c.x-left)*k;ctx.fillText(c.label,x,plot+18,Math.min(width,Math.max(70,stride*column*k)));
  });
}

const atlasViews = new WeakMap();
let atlasMountId=0;
const freshCamera = () => ({x:500, y:310, zoom:1, selected:''});
export function overviewMarkup() {
  return `<div class="atlas-slot"><section class="graph-atlas" aria-label="全图浏览器">
    <header class="atlas-head"><div role="group" aria-label="图层"><button type="button" data-layer="bcg">BCG 全图</button><button type="button" data-layer="tsg">TSG 全图</button></div>
      <button class="button" type="button" data-map="screen" aria-pressed="false">全屏查看</button></header>
    <div class="atlas-body">
      <p class="atlas-count" aria-live="polite"></p>
      <div class="atlas-controls"><label class="search-input"><input type="search" class="atlas-query" aria-label="在全图中定位节点" placeholder="搜索节点 ID 或正文"></label>
        <button class="button" type="button" data-map="locate">定位</button>
        <div role="group" aria-label="画布缩放"><button class="button" type="button" data-map="out" aria-label="缩小全图">−</button><output class="atlas-scale">100%</output><button class="button" type="button" data-map="in" aria-label="放大全图">＋</button>
        <button class="button" type="button" data-map="fit">适应全图</button></div></div>
      <div class="atlas-task-options" hidden><button type="button" class="button" data-map="root">定位总任务</button><label><input type="checkbox" data-shared-wires> 显示共享 Belief 分组连线</label></div>
      <div class="atlas-stage" tabindex="0" aria-label="图谱画布：方向键平移，加减键缩放，Home 适应全图。也可通过上方搜索定位节点。"><canvas></canvas><div class="atlas-set-board" hidden><div class="atlas-set-world" role="group" aria-label="从用户原始输入展开的任务树"></div></div><p class="atlas-empty" hidden></p></div>
      <div class="atlas-timeline-nav" hidden><label>轨迹导航 <input type="range" min="0" step="1" value="0" aria-label="跳转到轨迹列"></label><output aria-label="当前画面中的轨迹列"></output></div>
      <div class="atlas-overlaps" hidden></div>
      <div class="atlas-key"><span class="atlas-legend"></span><span>拖拽平移 · 滚轮或双指缩放</span></div>
      <div class="atlas-extra"></div>
      <div class="atlas-inspector" aria-live="polite"></div>
    </div></section><dialog class="atlas-modal" aria-label="全屏图谱"></dialog></div>`;
}

const statusText = row => row?.status_provenance === 'not_reported' ? '未报告进展' :
  ({open:'已报告开放',in_progress:'进行中',blocked:'受阻',reported_complete:'已报告完成（未核实）',cancelled:'已取消'})[row?.status] || row?.status || '未记录';

const memberPageSize=12;
const memberTitle=key=>memberLabels[key]||({all:'全部成员',shared:'跨集合共享'})[key]||'全部成员';
const setButton=(id,bucket,label,cls='')=>`<button type="button" data-set-open="${esc(id)}" data-bucket="${esc(bucket)}" class="${cls}">${label}</button>`;
const taskLink=(id,context='')=>setButton(id,'all',esc(context+id),'atlas-task-ref');
const overlapOf=(sets,bucket)=>sets && typeof bucket==='string' && /^overlap-\d+$/.test(bucket)?sets.overlaps[Number(bucket.slice(8))]:null;
function renderOverlaps(sets){
  const label=o=>'仅 '+o.tasks.join('、')+'（'+o.ids.length+'）';
  return '<span>共享归属</span>'+sets.overlaps.slice(0,3).map((o,i)=>`<button type="button" data-overlap="${i}" title="${esc(label(o))}">${esc(label(o))}</button>`).join('')+
    (sets.overlaps.length>3?`<select aria-label="其他共享归属" data-overlap-picker><option value="">其他分组（${sets.overlaps.length-3}）</option>${sets.overlaps.slice(3).map((o,i)=>`<option value="${i+3}">${esc(label(o))}</option>`).join('')}</select>`:'');
}
function setCard(g,expanded){
  const buckets=Object.entries(memberLabels).flatMap(([key,label])=>{
    const rows=g.buckets.get(key);if(!rows?.length)return [];
    return [setButton(g.id,key,`<span>${label}</span><small>${esc(rows.slice(0,2).map(m=>m.id).join(' · '))}</small><b>${rows.length} ›</b>`,'atlas-bucket')];
  }).join('');
  return `<article class="atlas-set-card" data-set-id="${esc(g.id)}">
    <button type="button" class="atlas-set-mini" data-set-open="${esc(g.id)}" data-bucket="all" aria-label="${esc(g.id+'，'+g.rows.length+' 个成员，查看任务集合')}"><b>${esc(g.id)}</b><small>${g.rows.length} 个成员</small></button>
    <header>${setButton(g.id,'all',`<i aria-hidden="true"></i>${esc(g.id)}`,'atlas-set-name')}<span class="atlas-status">${esc(statusText(g.node.raw))}</span></header>
    <p class="atlas-set-title" title="${esc(g.node.text)}">${esc(g.node.text || '任务文本未记录')}</p>
    <div class="atlas-set-totals">${setButton(g.id,'all',`全部成员 <b>${g.rows.length}</b>`)}${g.shared.length?setButton(g.id,'shared',`其中共享 <b>${g.shared.length}</b>`):''}</div>
    <details class="atlas-set-fold"${expanded?' open':''}><summary>成员分类</summary><div class="atlas-buckets">${buckets||'<span class="atlas-set-note">该任务没有记录成员引用。</span>'}</div>
    ${g.parents.size||g.children.size?`<footer>${[['父任务',g.parents],['子任务',g.children]].filter(([,ids])=>ids.size).map(([label,ids])=>
      `<span>${label}：${[...ids].slice(0,2).map(id=>taskLink(id)).join(' ')}${ids.size>2?setButton(g.id,'all','+'+(ids.size-2),'atlas-task-ref'):''}</span>`).join(' ')}</footer>`:''}
    </details></article>`;
}
function taskRootCard(question){
  return `<button class="atlas-task-root" type="button" data-task-root data-copy-lock title="${esc(question)}" aria-label="User input · 总任务，查看原文">
    <strong>User input <small>总任务</small></strong><span>${esc(question)}</span></button>`;
}
function roundedRoute(points){
  const p=points.filter(([x,y],i)=>!i||x!==points[i-1][0]||y!==points[i-1][1]);
  let d='M'+p[0].join(' ');
  for(let i=1;i<p.length-1;i++){
    const [x,y]=p[i],[ax,ay]=p[i-1],[bx,by]=p[i+1],a=Math.hypot(ax-x,ay-y),b=Math.hypot(bx-x,by-y);
    const r=Math.min(8,a/2,b/2);
    d+=` L${x+(ax-x)*r/a} ${y+(ay-y)*r/a} Q${x} ${y} ${x+(bx-x)*r/b} ${y+(by-y)*r/b}`;
  }
  return d+' L'+p.at(-1).join(' ');
}
function taskWires(scene,marker){
  const defs=`<defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 Z"/></marker></defs>`;
  const path=e=>{
    const shared=e.kind==='shared', label=e.record?`${e.from} → 子任务 ${e.to}（原始记录：${e.record.from_id} part_of ${e.record.to_id}）`:
      shared?`${e.from} 引用本组共享 Belief（不是任务间的边）`:`User input → 一级任务 ${e.to}（展示归属，非原图关系）`;
    const source=e.record?`data-task-relation="part_of" data-record-from="${esc(e.record.from_id)}" data-record-to="${esc(e.record.to_id)}"`:'';
    return `<path class="${shared?'atlas-shared-wire':'atlas-task-wire'}" data-wire-kind="${e.kind}" data-wire-from="${esc(e.from)}" data-wire-to="${esc(e.to)}" ${source} ${shared?'data-shared-reference=""':`marker-end="url(#${marker})"`} d="${roundedRoute(e.points)}"><title>${esc(label)}</title></path>`;
  };
  return defs+scene.links.filter(e=>e.kind==='shared').map(path).join('')+scene.links.filter(e=>e.kind!=='shared').map(path).join('');
}
function memberList(g,sets,view){
  const overlap=overlapOf(sets,view.bucket);
  const rows=overlap?overlap.ids:view.bucket==='shared'?g.shared:g.buckets.get(view.bucket)??g.rows;
  const pages=Math.max(1,Math.ceil(rows.length/memberPageSize));
  const page=Math.max(0,Math.min(pages-1,view.page||0));view.page=page;
  const relatives=(label,ids)=>ids.size?`<span>${label}：${[...ids].map(id=>taskLink(id)).join(' ')}</span>`:'';
  const filters=['all',...Object.keys(memberLabels).filter(k=>g.buckets.has(k)),...(g.shared.length?['shared']:[])];
  return `<div class="atlas-members">
    ${overlap?`<div class="atlas-owner-list">所属集合 ${overlap.tasks.map(id=>taskLink(id)).join(' ')}</div>`:g.parents.size||g.children.size?`<div class="atlas-owner-list">${relatives('父任务',g.parents)}${relatives('子任务',g.children)}</div>`:''}
    <div class="atlas-member-head"><strong>${overlap?'仅由以上集合共同引用':esc(g.id)+' · '+esc(memberTitle(view.bucket))}（${rows.length}）</strong>${overlap?'':`<label>成员范围 <select aria-label="选择成员范围" data-member-filter>${filters.map(k=>`<option value="${k}" ${k===view.bucket?'selected':''}>${esc(memberTitle(k))}</option>`).join('')}</select></label>`}</div>
    <ul class="atlas-member-list">${rows.slice(page*memberPageSize,(page+1)*memberPageSize).map(entry=>{
      const m=overlap?g.members.get(entry):entry;
      const owners=sets.owners.get(m.id),others=[...owners].filter(id=>id!==g.id);
      return `<li><button type="button" class="atlas-member-open" data-set-member="${esc(m.id)}"><b>${esc(m.id)}</b>${overlap?'':`<small>${esc(memberLabels[m.bucket])}</small>`}<span>${esc(m.node.text||'本快照未收录正文')}</span></button>
        ${!overlap && others.length?`<span class="atlas-shared-by">也属于 ${others.map(id=>taskLink(id)).join(' ')}</span>`:''}</li>`;
    }).join('')}</ul>
    ${pages>1?`<div class="atlas-pagination"><button type="button" data-member-page="-1" ${page===0?'disabled':''}>上一页</button><span>${page+1} / ${pages}</span><button type="button" data-member-page="1" ${page+1>=pages?'disabled':''}>下一页</button></div>`:''}</div>`;
}

export function mountOverview(slot, graph, initialLayer, saveLayer = () => {}, userInput = '') {
  const $ = s => slot.querySelector(s), panel = $('.graph-atlas'), canvas = $('canvas');
  const modal = $('.atlas-modal'), ctx = canvas.getContext('2d'), search = $('.atlas-query');
  const events = new AbortController(), opts = {signal:events.signal};
  const on = (el,type,fn,extra={}) => el.addEventListener(type,fn,{...opts,...extra});
  const colors = getComputedStyle(panel);
  const theme=key=>colors.getPropertyValue('--'+key).trim();
  const palette={ink:theme('navy'),red:theme('red'),focus:theme('focus'),line:theme('line-strong'),surface:theme('surface'),muted:theme('muted')};
  const stage=$('.atlas-stage'),setBoard=$('.atlas-set-board'),worldEl=$('.atlas-set-world');
  const taskOptions=$('.atlas-task-options'),sharedWires=$('[data-shared-wires]'),marker='atlas-task-arrow-'+(++atlasMountId);
  const question=userInput || '未记录原始 User input';
  let layer, model, timeline, sets, view, frame=0, width=0, height=0, matches=[], matchIndex=-1;
  let extent={width:1,height:1},cardEls=new Map(),hubEls=new Map(),wireSvg=null,diagram=null;
  let layoutDirty=true,layoutWidth=-1;
  const nodeResize=new ResizeObserver(()=>{layoutDirty=true;repaint();});
  const timelineNav=$('.atlas-timeline-nav'), scrub=timelineNav.querySelector('input'), rangeLabel=timelineNav.querySelector('output');
  const pointers = new Map();
  let drag=false, pinch=null, lastDrag=-Infinity;
  const plotHeight = () => timeline ? Math.max(1,height-timelineSize.axis) : height;
  const hasRoot=()=>!!sets && model.available;
  const canNavigate=()=>hasRoot()||model.nodes.length>0;
  const screen = p => [(p.x-view.x)*view.zoom+width/2, (p.y-view.y)*view.zoom+plotHeight()/2];
  const world = (x,y) => [view.x+(x-width/2)/view.zoom, view.y+(y-plotHeight()/2)/view.zoom];
  const fitScale = () => Math.min(1,width/extent.width,plotHeight()/extent.height);
  function frameView(){
    const z=fitScale();
    const readableOverview=sets && sets.groups.size<=32 && z>=.2;
    if(view.framing==='fit'||readableOverview||z>=(sets ? .56 : .78)){view.zoom=z;view.x=extent.width/2;view.y=extent.height/2;}
    else if(sets && diagram?.focus){
      const w=width/2,h=plotHeight()/2;
      view.zoom=1;view.x=extent.width<=width?extent.width/2:Math.max(w,Math.min(extent.width-w,diagram.focus.x));
      view.y=extent.height<=2*h?extent.height/2:Math.max(h,Math.min(extent.height-h,diagram.focus.y));
    }
    else{view.zoom=1;view.x=Math.min(extent.width,width)/2;view.y=Math.min(extent.height,plotHeight())/2;}
  }
  function sizeBoard(){
    if(!layoutDirty && layoutWidth===width)return;
    layoutWidth=width;layoutDirty=false;
    if(!hasRoot()){extent={width:1,height:1};return;}
    const cardWidth=Math.max(170,Math.min(220,width-26));
    worldEl.style.setProperty('--atlas-task-width',cardWidth+'px');
    const visible=new Map(cardEls);
    for(const [id,el] of hubEls){el.hidden=!view.showShared;if(view.showShared)visible.set(id,el);}
    const sizes=new Map([...visible].map(([id,el])=>[id,{width:el.offsetWidth,height:el.offsetHeight}]));
    diagram=taskDiagramLayout(sets,sizes,view.showShared);extent=diagram;
    worldEl.style.width=diagram.width+'px';worldEl.style.height=diagram.height+'px';
    for(const [id,b] of diagram.boxes){const el=visible.get(id);el.style.left=b.x+'px';el.style.top=b.y+'px';}
    wireSvg.setAttribute('viewBox',`0 0 ${diagram.width} ${diagram.height}`);
    wireSvg.innerHTML=taskWires(diagram,marker);highlightTaskWires();
  }
  function highlightTaskWires(){
    if(!sets||!wireSvg)return;
    const overlap=overlapOf(sets,view.bucket);
    const ids=cardEls.has(view.selected)?new Set(overlap?.tasks||[view.selected]):sets.owners.get(view.selected);
    wireSvg.classList.toggle('has-selection',!!ids?.size);
    for(const el of wireSvg.querySelectorAll('[data-wire-from]')){
      const near=ids?.has(el.dataset.wireFrom)||ids?.has(el.dataset.wireTo);
      el.classList.toggle('is-near',!!near);
    }
  }
  function updateScrub(){
    const cols=timeline.columns,n=cols.length;if(!n)return;
    const index=x=>(x-cols[0].x)/timelineSize.column,clamp=i=>Math.max(0,Math.min(n-1,i));
    const current=clamp(Math.round(index(view.x))),half=width/(2*view.zoom);
    const start=Math.max(0,Math.ceil(index(view.x-half-timelineSize.radius)));
    const end=Math.min(n-1,Math.floor(index(view.x+half+timelineSize.radius)));
    scrub.value=current;scrub.setAttribute('aria-valuetext',cols[current].label);
    rangeLabel.textContent=start>end?'当前视野没有轨迹列':cols[start].label+(start===end?'':' – '+cols[end].label);
  }
  function draw(){
    frame=0;
    if(!slot.isConnected)return;
    width=stage.clientWidth;height=stage.clientHeight;if(!width||!height)return;
    if(sets)sizeBoard();else extent=timeline;
    if(view.framing)frameView();
    if(sets){
      // Switch the label, not the measured card box. Zooming must not cause
      // re-layout, move the arrows, or produce unreadable miniature paragraphs.
      worldEl.dataset.density=view.zoom<(sets.groups.size>4 ? .72 : .5)?'compact':'detail';
      worldEl.style.setProperty('--atlas-mini-font',Math.min(44,13/view.zoom)+'px');
      worldEl.style.transform=`translate(${width/2-view.x*view.zoom}px,${height/2-view.y*view.zoom}px) scale(${view.zoom})`;
    }else if(ctx){
      const dpr=Math.min(window.devicePixelRatio||1,2),w=Math.round(width*dpr),h=Math.round(height*dpr);
      if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
      paintTimeline(ctx,timeline,model,view,width,height,view.zoom,palette);
      updateScrub();
    }
    $('.atlas-scale').textContent=Math.round(view.zoom*100)+'%';
  }
  function repaint(){if(!frame)frame=requestAnimationFrame(draw);}
  function showDetail() {
    const n=model.byId.get(view.selected), el=$('.atlas-inspector');
    const overlap=overlapOf(sets,view.bucket),showOverlap=overlap && sets.groups.has(view.selected);
    if(sets)for(const [id,card] of cardEls){
      card.classList.toggle('current-set',!showOverlap && id===view.selected);
      card.classList.toggle('contains-selection',showOverlap?overlap.tasks.includes(id):sets.owners.get(view.selected)?.has(id)??false);
    }
    highlightTaskWires();
    const workflowHTML=record=>workflowDetails(graph,record).map(item=>
      `<p class="atlas-workflow-line"><strong>${esc(item.label)}</strong> · ${esc(item.id)}<br>${esc(item.text)}</p>`).join('');
    if(hasRoot() && view.selected===taskRootId){
      el.innerHTML=`<div class="atlas-detail" data-copy-lock><div class="atlas-root-heading"><strong>User input · 总任务</strong><button type="button" class="atlas-task-ref" data-root-close>收起原文</button></div><p>${esc(question)}</p>${workflowHTML(graph.taskState?.root)}</div>`;
      el.scrollTop=0;return;
    }
    if(!n){el.innerHTML='';return;}
    if(showOverlap){el.innerHTML=memberList(sets.groups.get(n.id),sets,view);el.scrollTop=0;return;}
    const edges=model.incident.get(n.id),r=n.raw,p=timeline?.positions.get(n.id);
    const group=sets?.groups.get(n.id),owners=sets?.owners.get(n.id);
    const meta=n.kind==='task'?`任务状态：${statusText(r)}${r?.status_source_id!=null?' · 状态来源 B'+r.status_source_id:''}`:
      n.kind==='missing'?'快照只记录了这个端点的 ID': [p && (p.turn==null?'轮次未记录':`traj[${p.turn}]`),p?.role,p?.decision && p.caption+' 决策',r?.task_role,r?.stance,r?.confidence!=null?'confidence '+r.confidence:''].filter(Boolean).join(' · ');
    el.innerHTML=`${owners?.size && view.origin?`<button type="button" class="atlas-task-ref" data-member-back>← 返回${overlap?'共享分组':' '+esc(view.origin)+' 成员'}</button>`:''}
      <div class="atlas-detail" data-copy-lock><strong>${esc(n.id)}</strong> <span>${esc(meta)}</span><p>${esc(n.text)}</p>${n.kind==='task'?workflowHTML(r):''}</div>
      ${owners?.size?`<div class="atlas-owner-list">所属集合（${owners.size}） ${[...owners].map(id=>taskLink(id)).join(' ')}</div>`:''}
      ${group?memberList(group,sets,view):''}
      <details class="atlas-links"><summary>查看原始关系（${edges.length}）</summary><div></div></details>`;
    el.scrollTop=0;
    const details=el.querySelector('details');
    details.addEventListener('toggle',()=>{
      if(!details.open || details.dataset.filled)return;
      details.dataset.filled='1';
      details.querySelector('div').innerHTML=edges.map(e=>{
        const other=e.from_id===n.id?e.to_id:e.from_id, r=e.raw;
        const provenance=r.note || (Array.isArray(r.relation_ids)&&r.relation_ids.length?'原图关系 ID：'+r.relation_ids.join(', '):'');
        return `<button type="button" data-map-target="${esc(other)}" data-copy-lock>${esc(e.from_id)} → ${esc(e.to_id)} · ${esc(e.type)}${provenance?' · '+esc(provenance):''}</button>`;
      }).join('')||'无相连关系';
    });
  }
  function select(id,center=false) {
    if(!model.byId.has(id) && !(hasRoot() && id===taskRootId))return;
    view.selected=id;view.framing=null;
    if(sets){
      if(id===taskRootId){view.origin=id;view.bucket='all';view.page=0;}
      else if(sets.groups.has(id)){
        if(view.origin!==id){view.bucket='all';view.page=0;}
        view.origin=id;
      }else if(!sets.owners.get(id)?.has(view.origin)){
        view.origin=sets.owners.get(id)?.values().next().value??'';view.bucket='all';view.page=0;
      }
    }
    if(center){
      const card=cardEls.get(view.origin),p=timeline?.positions.get(id) ??
        (card?{x:card.offsetLeft+card.offsetWidth/2,y:card.offsetTop+card.offsetHeight/2}:null);
      if(p){view.x=p.x;view.y=p.y;view.zoom=Math.max(view.zoom,1);}
    }
    showDetail();repaint();
    // In the two-column workbench the inspector can be just below the fold.
    // Reveal it within the evidence pane without scrolling the annotation pane.
    const scroller=sets && !full() && panel.closest('.panel-body');
    if(scroller && scroller.scrollHeight>scroller.clientHeight){
      const below=$('.atlas-inspector').getBoundingClientRect().top-scroller.getBoundingClientRect().bottom;
      if(below> -64)scroller.scrollTop+=below+Math.min(180,scroller.clientHeight/2);
    }
  }
  function openSet(id,bucket,center=false){
    if(!sets?.groups.has(id))return;
    view.origin=id;view.bucket=bucket;view.page=0;select(id,center);
  }
  function showOverlapGroup(i){const o=sets?.overlaps[i];if(o)openSet(o.tasks[0],'overlap-'+i);}
  function setLayer(next) {
    layer=next==='tsg'?'tsg':'bcg'; model=overviewGraph(graph,layer);
    timeline=layer==='bcg'?timelineDrawing(model):null;
    sets=timeline?null:taskSetIndex(model);
    canvas.hidden=!!sets;setBoard.hidden=!sets;
    nodeResize.disconnect();layoutDirty=true;layoutWidth=-1;diagram=null;
    if(!atlasViews.has(model))atlasViews.set(model,{...freshCamera(),framing:'start',origin:'',bucket:'all',page:0,showShared:false});
    view=atlasViews.get(model); pointers.clear();pinch=null;drag=false;search.value='';matches=[];matchIndex=-1;
    worldEl.innerHTML=hasRoot()?`<svg class="atlas-task-wires" aria-label="从总任务指向子任务的分支"></svg>`+taskRootCard(question)+
      [...sets.groups.values()].map(g=>setCard(g,sets.groups.size<=4)).join('')+sets.overlaps.map((o,i)=>
        `<button type="button" class="atlas-hub" data-hub-open="${i}" title="${esc('仅 '+o.tasks.join('、')+' 共同引用；点击查看成员')}" hidden>共享 ${o.ids.length} 个 Belief</button>`).join(''):'';
    wireSvg=worldEl.querySelector('svg');
    cardEls=new Map([...worldEl.querySelectorAll('[data-set-id]')].map(el=>[el.dataset.setId,el]));
    if(hasRoot())cardEls.set(taskRootId,worldEl.querySelector('[data-task-root]'));
    hubEls=new Map([...worldEl.querySelectorAll('[data-hub-open]')].map(el=>['@shared:'+el.dataset.hubOpen,el]));
    for(const el of cardEls.values())nodeResize.observe(el);
    taskOptions.hidden=!hasRoot();sharedWires.checked=!!sets && view.showShared;sharedWires.disabled=!sets?.overlaps.length;
    for(const b of panel.querySelectorAll('[data-layer]'))b.setAttribute('aria-pressed',String(b.dataset.layer===layer));
    panel.dataset.layer=layer;panel.dataset.nodes=String(model.nodes.length);panel.dataset.relations=String(model.relations.length);
    panel.dataset.layout=timeline?'trajectory':'task-tree';panel.dataset.columns=String(timeline?.columns.length??0);
    timelineNav.hidden=!timeline||timeline.columns.length<=1;scrub.max=Math.max(0,(timeline?.columns.length??0)-1);
    const count=sets?`${model.taskCount} 个任务集合 · ${sets.taskEdges.length} 条任务间关系 · ${sets.owners.size} 个不同的 Belief 成员 · ${sets.sharedCount} 个跨集合共享`:`${model.beliefCount} 个节点`;
    const issues=(model.missingCount?` · ${model.missingCount} 个未收录端点`:'')+(model.invalidEdges?` · ${model.invalidEdges} 条关系缺少端点`:'');
    $('.atlas-count').textContent=model.available?count+` · ${model.relations.length} 条关系`+issues+(timeline?' · 按建图轨迹从左到右排列':''):'本份快照未记录 Task State';
    $('.atlas-legend').textContent=timeline?'● Belief / Decision　红色虚线：矛盾关系':
      !hasRoot()?'':!sets.groups.size?'此快照尚未记录子任务。':
      '从上往下读：总任务 → 一级任务 → 子任务。首层是展示归属，其余实线来自 part_of。'+
      ([...sets.groups.values()].some(g=>g.parents.size>1)?'多父任务只展示一个分支，全部父级见任务详情。':'');
    const overlapNav=$('.atlas-overlaps');overlapNav.hidden=!sets?.overlaps.length;overlapNav.innerHTML=overlapNav.hidden?'':renderOverlaps(sets);
    $('.atlas-extra').innerHTML=sets?.unplaced.length?`<details><summary>未归入集合的原始关系（${sets.unplaced.length}）</summary>${sets.unplaced.map(e=>`<div><button type="button" data-map-target="${esc(e.from_id)}">${esc(e.from_id)}</button> → <button type="button" data-map-target="${esc(e.to_id)}">${esc(e.to_id)}</button> · ${esc(e.type)}</div>`).join('')}</details>`:'';
    const empty=$('.atlas-empty');
    empty.hidden=canNavigate();empty.textContent=!model.available?'本快照未记录 TSG，不从其他轮次推算。':'本快照没有节点。';
    for(const el of panel.querySelectorAll('.atlas-controls button,.atlas-query'))el.disabled=!canNavigate();
    updateSearch();showDetail();repaint();
  }
  function updateSearch(){
    const q=search.value.trim().toLowerCase(),alias=timeline && /^([bd])#?(\d+)$/.exec(q);
    let exact=model.byId.get(q.toUpperCase());
    if(!exact && alias){
      const id='B'+Number(alias[2]),p=timeline.positions.get(id);
      if(p && (alias[1]!=='d'||p.decision))exact=model.byId.get(id);
    }
    matches=q?(exact?[exact]:model.nodes.filter(n=>(n.id+' '+n.text).toLowerCase().includes(q))):[];
    if(q && !exact && hasRoot() && ('user input 总任务 '+userInput).toLowerCase().includes(q))matches.unshift({id:taskRootId});
    matchIndex=-1;const b=$('[data-map="locate"]');b.disabled=!matches.length;b.textContent=q?`定位 (${matches.length})`:'定位';
  }
  function locate(){if(matches.length){if(sets){view.bucket='all';view.page=0;}select(matches[++matchIndex%matches.length].id,true);}}
  function zoom(factor,x=width/2,y=plotHeight()/2){
    if(!width||!height)return;
    const [wx,wy]=world(x,y);view.framing=null;
    view.zoom=Math.min(6,Math.max(Math.min(.2,fitScale()),view.zoom*factor));
    view.x=wx-(x-width/2)/view.zoom;view.y=wy-(y-plotHeight()/2)/view.zoom;repaint();
  }
  function fitAll(){view.framing='fit';repaint();}
  const full=()=>document.fullscreenElement===panel||modal.open;
  function screenLabel(){const b=$('[data-map="screen"]');b.textContent=full()?'退出全屏':'全屏查看';b.setAttribute('aria-pressed',String(full()));repaint();}
  function restore(){slot.insertBefore(panel,modal);screenLabel();$('[data-map="screen"]').focus({preventScroll:true});}
  async function toggleScreen(){
    if(modal.open){modal.close();return;}
    if(document.fullscreenElement===panel){await document.exitFullscreen();return;}
    if(document.fullscreenEnabled&&panel.requestFullscreen){
      try{await panel.requestFullscreen();return;}catch{
        // Some embedding contexts disable native fullscreen. Use a modal filling the viewport.
      }
    }
    if(slot.isConnected){modal.append(panel);modal.showModal();screenLabel();}
  }
  on(modal,'close',restore);on(document,'fullscreenchange',screenLabel);
  on(panel,'click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(stage.contains(b) && e.detail && performance.now()-lastDrag<350)return;
    if(b.dataset.layer){setLayer(b.dataset.layer);saveLayer(layer);return;}
    if(b.hasAttribute('data-task-root')||b.dataset.map==='root'){select(taskRootId,true);return;}
    if(b.hasAttribute('data-root-close')){view.selected='';view.origin='';showDetail();fitAll();$('[data-map="root"]').focus({preventScroll:true});return;}
    if(b.hasAttribute('data-hub-open')){showOverlapGroup(b.dataset.hubOpen);return;}
    if(b.hasAttribute('data-overlap')){showOverlapGroup(b.dataset.overlap);return;}
    if(b.hasAttribute('data-member-back')){select(view.origin);return;}
    if(b.dataset.setOpen){openSet(b.dataset.setOpen,b.dataset.bucket,b.classList.contains('atlas-set-mini')||!worldEl.contains(b));return;}
    if(b.dataset.setMember){select(b.dataset.setMember);return;}
    if(b.dataset.memberPage){view.page+=Number(b.dataset.memberPage);showDetail();return;}
    if(b.dataset.mapTarget){select(b.dataset.mapTarget,true);return;}
    switch(b.dataset.map){case 'screen':void toggleScreen();break;case 'in':zoom(1.45);break;
      case 'out':zoom(1/1.45);break;case 'fit':fitAll();break;case 'locate':locate();break;}
  });
  on(panel,'change',e=>{
    if(e.target===sharedWires && sets){view.showShared=sharedWires.checked;layoutDirty=true;view.framing||='start';repaint();return;}
    if(e.target.matches('[data-member-filter]'))openSet(view.origin,e.target.value);
    if(e.target.matches('[data-overlap-picker]') && e.target.value!==''){showOverlapGroup(e.target.value);e.target.value='';}
  });
  on(worldEl,'focusin',e=>{
    if(!sets || !e.target.matches(':focus-visible'))return;
    const r=e.target.getBoundingClientRect(),v=stage.getBoundingClientRect();
    if(r.left<v.left||r.right>v.right||r.top<v.top||r.bottom>v.bottom){
      view.framing=null;view.x+=(r.left+r.right-v.left-v.right)/(2*view.zoom);
      view.y+=(r.top+r.bottom-v.top-v.bottom)/(2*view.zoom);repaint();
    }
  });
  on(search,'input',updateSearch);on(search,'keydown',e=>{if(e.key==='Enter'){e.preventDefault();locate();}});
  on(scrub,'input',()=>{
    const c=timeline?.columns[Number(scrub.value)];if(!c)return;
    view.framing=null;view.zoom=Math.max(view.zoom,1);view.x=c.x;view.y=Math.min(timeline.height,plotHeight()/view.zoom)/2;repaint();
  });
  on(stage,'keydown',e=>{
    if(e.target!==stage)return;
    const step=60/view.zoom;
    if(e.key.startsWith('Arrow'))view.framing=null;
    switch(e.key){case 'ArrowRight':view.x+=step;break;case 'ArrowLeft':view.x-=step;break;
      case 'ArrowDown':view.y+=step;break;case 'ArrowUp':view.y-=step;break;
      case '+':case '=':zoom(1.45);break;case '-':zoom(1/1.45);break;case 'Home':fitAll();break;default:return;}
    e.preventDefault();repaint();
  });
  const xy=e=>{const b=stage.getBoundingClientRect();return [e.clientX-b.left-stage.clientLeft,e.clientY-b.top-stage.clientTop];};
  on(stage,'wheel',e=>{if(!canNavigate())return;e.preventDefault();zoom(Math.exp(-Math.max(-200,Math.min(200,e.deltaY*(e.deltaMode===1?16:1)))*.003),...xy(e));},{passive:false});
  function gesture(){const [a,b]=pointers.values();return {x:(a.x+b.x)/2,y:(a.y+b.y)/2,d:Math.max(1,Math.hypot(b.x-a.x,b.y-a.y))};}
  on(stage,'pointerdown',e=>{
    if(e.button!==0||!canNavigate())return;
    const [x,y]=xy(e);pointers.set(e.pointerId,{x,y,sx:x,sy:y});
    // Keep a real click on a set button; capture it only once a drag actually starts.
    (e.target.closest('button,summary')||stage).setPointerCapture(e.pointerId);
    if(pointers.size===1){drag=false;lastDrag=-Infinity;}
    if(pointers.size===2){drag=true;pinch=gesture();}
  });
  on(stage,'pointermove',e=>{
    const p=pointers.get(e.pointerId);if(!p)return;
    const [x,y]=xy(e),dx=x-p.x,dy=y-p.y;p.x=x;p.y=y;
    if(pointers.size===2 && pinch){
      const g=gesture();zoom(g.d/pinch.d,pinch.x,pinch.y);
      view.x-=(g.x-pinch.x)/view.zoom;view.y-=(g.y-pinch.y)/view.zoom;pinch=g;
    }else if(pointers.size===1){
      drag ||= Math.hypot(x-p.sx,y-p.sy)>3;
      if(drag){view.framing=null;view.x-=dx/view.zoom;view.y-=dy/view.zoom;}
    }
    if(drag && !stage.hasPointerCapture(e.pointerId))stage.setPointerCapture(e.pointerId);
    stage.classList.toggle('dragging',drag);repaint();
  });
  function end(e){
    if(e.type==='lostpointercapture' && e.target!==stage)return;
    if(!pointers.delete(e.pointerId))return;
    if(drag)lastDrag=performance.now();
    if(timeline && !drag && e.type==='pointerup'){
      const [x,y]=xy(e);let hit=null,min=Infinity;
      for(const n of model.nodes){const [nx,ny]=screen(timeline.positions.get(n.id)),dx=x-nx,dy=y-ny,d=Math.hypot(dx,dy);
        const label=view.zoom>=.64 && dx>0 && dx<=(timelineSize.radius+4+n.id.length*8)*view.zoom && Math.abs(dy)<=8*view.zoom;
        if(y<plotHeight() && d<min && (d<=timelineSize.radius*view.zoom+3||label)){hit=n;min=d;}}
      if(hit)select(hit.id);
      else if(y<plotHeight() && view.selected){view.selected='';showDetail();repaint();}
    }
    pinch=pointers.size===2?gesture():null;
    if(!pointers.size){drag=false;stage.classList.remove('dragging');}
  }
  for(const name of ['pointerup','pointercancel','lostpointercapture'])on(stage,name,end);
  setLayer(initialLayer);
  const observer=new ResizeObserver(repaint);observer.observe(stage);
  return ()=>{events.abort();observer.disconnect();nodeResize.disconnect();cancelAnimationFrame(frame);if(modal.open)modal.close();};
}

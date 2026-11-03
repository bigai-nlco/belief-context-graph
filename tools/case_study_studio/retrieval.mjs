// Read-only retrieval inspection. Scores always come from the recorded invocation.
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
      scoringSkipped: !selection.strategy && !selection.node_scores && !ctx.nodeIds.length && !ctx.fallback,
      relevanceApplicable: selection.strategy !== 'confidence_pcst'};
  }).sort((a, b) => Number(b.selected) - Number(a.selected) || (b.prize ?? -1) - (a.prize ?? -1) || a.node.id - b.node.id);
}

function scoreText(row, field) {
  if (field === 'relevance' && !row.relevanceApplicable) return '不适用';
  if (!finite(row[field]) && row.scoringSkipped) return '未计算';
  return fmt(row[field]) + (row[`${field}Origin`] === 'replay' ? '（补算）' : '');
}

const layouts = new WeakMap();
export function graphLayout(graph, ids) {
  let cache = layouts.get(graph);
  if (!cache) {cache = new Map(); layouts.set(graph, cache);}
  const key = ids.join(',');
  if (cache.has(key)) return cache.get(key);
  const points = ids.map((id, i) => ({id, x: 500 + Math.cos(i * 2.39996) * 24 * Math.sqrt(i), y: 310 + Math.sin(i * 2.39996) * 24 * Math.sqrt(i), vx: 0, vy: 0}));
  const byId = new Map(points.map(p => [p.id, p]));
  const edges = graph.relations.filter(e => e.from_id !== e.to_id && byId.has(e.from_id) && byId.has(e.to_id));
  // Full graph is never truncated. Large graphs use a spatial grid for repulsion.
  for (let step = 0; step < 150; step++) {
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
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), minY = Math.min(...ys), maxX = Math.max(...xs), maxY = Math.max(...ys);
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
  const embedding=selection.embedding || {}, cache=embedding.persistent_cache || {};
  const file=c.files.find(f=>f.id===ctx.fileId)?.path || '';
  const filter=(state.retrievalQuery || '').toLowerCase();
  const filtered=rows.filter(r=>(state.retrievalFilter!=='selected'||r.selected)&&(state.retrievalFilter!=='unselected'||!r.selected)&&`B${r.node.id} ${text(r.node)}`.toLowerCase().includes(filter));
  const metric=(label,value)=>`<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`;
  const notices={missing:'缺少对应时刻的已确认快照，不使用最终图代替。', ambiguous:'同一调用对应多份不同快照，无法确定当时图。', mismatch:'快照缺少审计中选中的节点或关系，请核对文件是否来自同次运行。'};
  return `<section class="retrieval-view" data-copy-lock>
    <div class="retrieval-heading"><div><span class="eyebrow">RETRIEVAL INSPECTOR</span><h3>沿轨迹查看每次选图</h3><p>完整图、实际注入子图与当轮评分。</p></div><span class="retrieval-badge">${esc(selection.strategy || ctx.raw.retrieval_strategy || '旧版检索')}</span></div>
    <nav class="retrieval-navigation" aria-label="逐轮选图切换"><button class="button" data-retrieval-step="-1" ${index===0?'disabled':''}>← 上一轮</button><label>模型调用<select id="retrieval-round">${rounds.map((r,i)=>`<option value="${esc(r.id)}" ${r.id===ctx.id?'selected':''}>${i+1}. call ${esc(r.call)} · ${esc(r.graphId || c.files.find(f=>f.id===r.fileId)?.path || '')}</option>`).join('')}</select></label><button class="button" data-retrieval-step="1" ${index===rounds.length-1?'disabled':''}>下一轮 →</button></nav>
    <p class="retrieval-provenance">${index+1} / ${rounds.length} 次调用 · ${esc(file)} · Graph 已确认 turns：${esc(graph?.nTurns ?? ctx.raw.graph_n_turns_ingested ?? '未记录')}</p>
    ${ctx.fallback?`<p class="notice amber">本轮回退到原始上下文：${esc(ctx.fallback)}。下方高亮以实际注入 ID 为准。</p>`:!ctx.nodeIds.length?'<p class="notice">本轮未注入子图，可能尚未发生历史裁剪。</p>':''}
    ${selection.score_replay?`<p class="retrieval-legacy">已补算 ${Object.keys(selection.score_replay.node_scores || {}).length} 个候选节点的评分，标记为“补算”。选点、边、子图收益和注入文本仍为原始记录。${selection.score_replay.comparison?.matches_within_tolerance===false?' 补算 prize 与部分原记录有差异，不能当作历史原值。':''}</p>`:selection.strategy&&!selection.node_scores?'<p class="retrieval-legacy">旧记录未保存全部候选评分。缺失值显示“未记录”，不会重新计算后冒充当时结果。</p>':!selection.strategy?'<p class="retrieval-legacy">本轮没有评分记录；未发生检索的调用显示“未计算”。</p>':''}
    <dl class="retrieval-metrics">${metric('完整图',graph?`${graph.nodes.length} 点 / ${graph.relations.length} 边`:'快照未匹配')}${metric('实际选中',`${ctx.nodeIds.length} 点 / ${ctx.relationIds.length} 边`)}${metric('子图收益',fmt(selection.objective))}${metric('图字符 / 上限',`${fmt(ctx.raw.graph_chars)} / ${fmt(budget)}`)}${metric('每条边成本',fmt(selection.parameters?.edge_cost))}${metric('上下文准备（秒）',fmt(finite(ctx.raw.duration_ms) ? ctx.raw.duration_ms / 1000 : null))}</dl>
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

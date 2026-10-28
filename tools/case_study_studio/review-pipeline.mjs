import {STAGES,LIMITS,ARTIFACTS} from './review-pipeline-data.mjs';

const $=selector=>document.querySelector(selector);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const json=value=>JSON.stringify(value,null,2);
const KIND={code:'CODE · 确定性处理',llm:'LLM · 模型判断',gate:'GATE · 程序校验'};
let contract,selected=0,expanded=false,turn=0,promptStage='findings',promptView='template',noticeTimer,printExpanded;
const codeCache=new Map();
function codePanel(title,text,key){codeCache.set(key,text);return `<div class="code-panel"><div class="code-heading"><strong>${esc(title)}</strong><button class="copy-button" data-copy="${esc(key)}">复制</button></div><pre tabindex="0">${esc(text)}</pre></div>`;}
function table(rows,headers=['字段 / 对象','来源 / 去向','定义与作用']){return `<div class="table-wrap"><table class="method-table"><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function sourceButtons(keys){return `<div class="sources"><span>实现依据</span>${keys.map(key=>{const s=contract.sources[key];return `<button class="source-link" data-source="${key}">${esc(s.path.split('/').at(-1))} · ${esc(s.symbol)}():${s.line}</button>`;}).join('')}</div>`;}
function examplePair(stage){
  const e=contract.example;
  return {
    intake:[{run_id:'示例运行',benchmark:'illustration',task_id:'METHOD-DEMO'},e.native],
    align:[e.native,{question:e.case.question,final_report:e.case.final_report,steps:e.case.steps,coverage:e.case.coverage,evidence_sha256:e.case.evidence_sha256}],
    batch:[e.case,e.packets],find:[e.requests.findings,e.findings],
    'find-gate':[e.findings,{validation:'passed by current validate_findings',findings:e.findings}],
    pool:[e.findings,e.summary_input],synthesize:[e.summary_input,e.summary],
    'summary-gate':[e.summary,{validation:'passed by current validate_summary',summary:e.summary}],
    publish:[{selected:e.summary.selected_finding_ids,groups:e.summary.process_groups},e.annotation],
  }[stage.id];
}
function stageHTML(stage){
  const pair=examplePair(stage);
  return `<article class="stage-sheet" id="detail-${stage.id}">
    <header class="stage-header"><div><span class="eyebrow">STAGE ${stage.no} / ${esc(stage.phase)}</span><h3 tabindex="-1">${esc(stage.title)}</h3></div><span class="stage-badge ${stage.kind}">${KIND[stage.kind]}</span></header>
    <p class="stage-lead">${esc(stage.lead)}</p><div class="stage-equation">${esc(stage.formula)}</div>
    <h4 class="block-heading"><span>INPUT</span> 输入对象</h4>${table(stage.inputs)}
    <h4 class="block-heading"><span>TRANSFORM</span> 处理步骤与约束</h4><ol class="operation-list">${stage.operations.map(([title,body])=>`<li><strong>${esc(title)}</strong><p>${esc(body)}</p></li>`).join('')}</ol>
    <h4 class="block-heading"><span>OUTPUT</span> 输出对象</h4>${table(stage.outputs)}
    <div class="handoff"><strong>HANDOFF · 与下一阶段的衔接</strong>${esc(stage.handoff)}</div>
    <details class="guard-box"><summary>失败分支与实现边界</summary><ul>${stage.guards.map(g=>`<li>${esc(g)}</li>`).join('')}</ul></details>
    <details class="example-io"><summary>展开这一步的完整示例输入与输出 <small> · 结构示例</small></summary><div class="io-grid">${codePanel('INPUT / 实际结构',json(pair[0]),stage.id+'-input')}${codePanel('OUTPUT / 实际结构',json(pair[1]),stage.id+'-output')}</div></details>
    ${sourceButtons(stage.sources)}
  </article>`;
}
function renderStages(){
  $('#stage-detail').innerHTML=expanded?STAGES.map(stageHTML).join(''):stageHTML(STAGES[selected]);
  $('[data-expand]').textContent=expanded?'收起为逐步浏览':'展开全部阶段';
  $('[data-expand]').setAttribute('aria-expanded',String(expanded));
  document.querySelectorAll('[data-stage]').forEach(button=>{
    const active=button.dataset.stage===STAGES[selected].id;button.classList.toggle('active',active);
    if(active)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');
  });
  $('#stage-pagination').innerHTML=expanded?'':`<button class="button" data-step="-1" ${selected===0?'disabled':''}>← ${selected?STAGES[selected-1].short:'起点'}</button><button class="button" data-step="1" ${selected===STAGES.length-1?'disabled':''}>${selected<STAGES.length-1?STAGES[selected+1].short:'终点'} →</button>`;
}
function chooseStage(id,scroll=false){
  const index=STAGES.findIndex(s=>s.id===id);if(index<0)return;
  selected=index;expanded=false;renderStages();
  if(scroll){history.replaceState(null,'','#stage-'+id);$('#pipeline').scrollIntoView({block:'start'});}
}
function renderTurn(){
  const e=contract.example,s=e.case.steps[turn],f=e.candidates[turn],a=f.search_assessment;
  $('#turn-example').innerHTML=`<div class="turn-grid"><section class="evidence-box"><h4><span>BEFORE CALL ${s.call}</span> 实际注入的 Graph</h4><blockquote>${esc(s.graph.text)}</blockquote><pre>S-ID: ${esc(s.id)}\npath: ${esc(s.source.path)}\nJSONL line: ${s.source.index} · call: ${s.call}</pre></section>
    <section class="evidence-box"><h4><span>ACTION</span> 该轮 Assistant 输出</h4><blockquote>${esc(e.native.records.find(r=>r.t==='llm'&&r.turn===s.call).content)}</blockquote><hr><h4><span>OBSERVATION</span> 同轮工具返回</h4><blockquote>${esc(e.native.records.find(r=>r.t==='result'&&r.turn===s.call).result)}</blockquote></section></div>
    <div class="verdict-panel"><span class="verdict-pill ${f.kind==='benefit'?'':'negative'}">${esc(a.verdict)}</span><div><p><strong>Graph 充分性：</strong>${esc(a.graph_sufficiency)}　<strong>核验原因：</strong>${esc(a.verification_reason)}　<strong>实际增量：</strong>${esc(a.observed_gain)}</p><p>${esc(a.basis)}</p><p>候选 <code>${esc(f.id)}</code> → anchor <code>${esc(f.anchor_id)}</code>。此判断是用于展示数据结构的预设示例，已通过当前程序校验。</p></div></div>`;
  document.querySelectorAll('[data-turn]').forEach(button=>{const active=Number(button.dataset.turn)===turn;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
}
function renderContract(){
  document.querySelectorAll('[data-prompt-stage]').forEach(button=>{const active=button.dataset.promptStage===promptStage;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  document.querySelectorAll('[data-prompt-view]').forEach(button=>{const active=button.dataset.promptView===promptView;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  let text,note,title;
  if(promptView==='template'){
    title='system + user template · 从运行时代码导出';
    text=`[SYSTEM MESSAGE]\n${contract.prompts.system}\n\n[USER MESSAGE TEMPLATE]\n${contract.prompts[promptStage]}`;
    note='下方是当前完整模板，不是摘要。{dimensions}、{schema}、{evidence} 在 messages() 中填入 JSON。逐轮材料放在规则与输出结构之后。';
  }else if(promptView==='schema'){
    title=(promptStage==='findings'?'Findings':'Summary')+'.model_json_schema()';
    text=json(contract.schemas[promptStage]);note='直接由 Pydantic 生成。可查看必填字段、枚举、长度限制、嵌套结构与 additionalProperties=false；没有手工维护另一份 Schema。';
  }else{
    title='POST /chat/completions · 完整请求体结构示例';
    text=json({model:contract.model.model,reasoning_effort:contract.model.reasoning_effort,messages:contract.example.requests[promptStage],response_format:{type:'json_object'},max_completion_tokens:contract.model.max_output_tokens,stream:false});
    note='以贯穿示例填入模板后的完整请求体。没有读取任何 Key，也没有实际发起请求。请求不包含 tools、temperature 或 seed。';
  }
  codeCache.set('contract',text);$('#contract-note').textContent=note;
  $('#contract-display').innerHTML=`<div class="code-heading"><strong>${esc(title)}</strong><button class="copy-button" data-copy="contract">复制全文</button></div><pre tabindex="0">${esc(text)}</pre>`;
}
function render(){
  const defaults=contract.defaults;
  $('#main').innerHTML=`<section class="hero"><div class="hero-top"><span class="eyebrow">METHOD / EVIDENCE-GROUNDED AGENT REVIEW</span><div class="hero-actions"><button class="button" data-download>下载方法快照</button><button class="button solid" data-print>打印 / 保存 PDF</button></div></div>
    <h1>从逐轮证据到可追溯的<br>Agent Review</h1><p class="hero-subtitle">一份与当前实现对应的交互式方法说明。展开每一步的输入、转换、输出与传递关系，区分模型的语义判断和代码的确定性约束。</p>
    <div class="meta-row"><span><i></i><b>${esc(contract.model.model)}</b> / ${esc(contract.model.reasoning_effort)}</span><span>LiteLLM · Chat Completions</span><span>2 类模型请求 · 9 个处理阶段</span><span class="version">${esc(contract.prompt_version)}</span></div></section>

    <section class="paper-section" id="overview"><div class="section-heading"><div><span class="section-number">01 / PROBLEM FORMULATION</span><h2>评审对象与方法定义</h2><p>输入是一份已经结束的 Agent 运行，输出是由逐轮证据支撑的过程评价，而不是新的 Agent 执行轨迹。</p></div></div>
    <div class="definition"><div class="definition-copy"><p>方法回答两个独立问题：Agent 如何使用实际注入的 Belief Graph，以及最终报告对用户任务的完成程度。前者分析信息复用、搜索、缺口定位、验证、冲突处理与证据保留；后者比较任务要求与最终交付物。</p><p>评审模型不能运行工具。它先在有界批次中生成局部候选，再在全局候选池中归并、排序和总结。两个阶段都要通过结构与证据校验，才会产生可展示的标注。</p>
    <dl class="symbol-list"><dt>q / y</dt><dd>原始任务 / 最终报告</dd><dt>Gₜ / Aₜ</dt><dd>第 t 次调用前注入的 Graph / 该次 Assistant 输出</dd><dt>Oₜ / Hₜ</dt><dd>同轮工具结果 / 相邻两次调用参考</dd><dt>Pᵢ / C / S</dt><dd>批次输入 / 候选池 / 全局汇总</dd></dl></div>
    <aside class="formalism"><div class="equation">E = (q, y, {sₜ}<sub>t=1…T</sub>)<br>sₜ = (Gₜ, Aₜ, Oₜ, Hₜ, sourceₜ)<br>R = Serialize(E, C, S)</div><p><code>R</code> 包含关键过程、维度评价、归因总结和任务结果。候选与代表始终通过 <code>anchor_id → sourceLocator</code> 回到原始记录。</p><p>这是事后评审。y 用于检查结果，不可作为 Agent 在早期轮次已经看到的内容。</p></aside></div>
    <div class="legend"><span><i class="dot"></i>代码处理</span><span><i class="dot llm"></i>LLM 语义判断</span><span><i class="dot gate"></i>程序校验</span><span>点击节点查看精确 I/O</span></div>
    <div class="flow-grid" aria-label="AI Review 计算流程">${STAGES.map(s=>`<button class="flow-node ${s.kind}" data-stage="${s.id}"><span>${s.no}</span><span><strong>${s.short}</strong><small>${s.kind==='llm'?(s.id==='find'?'每批一次 · FIND':'全局一次 · SYNTHESIZE'):s.kind==='gate'?'失败 → 最多修正一次':'确定性转换 / 状态管理'}</small></span></button>`).join('')}</div>
    <div class="repair-strip"><span>校验失败分支</span><code>原请求 + 上次模型回答 + 校验错误</code><span>→ 同一模型重发完整 JSON → 通过 / 终止</span></div>
    <p class="figure-caption"><strong>图 1.</strong> 按 00–08 顺序执行。03 与 04 对每个批次重复，随后汇集为 05；06 与 07 在整个候选池上执行一次。节点展开中的 HANDOFF 标明真实的数据传递，修正请求不是另一个 Judge。</p></section>

    <section class="paper-section" id="pipeline"><div class="section-heading"><div><span class="section-number">02 / STAGE-BY-STAGE SPECIFICATION</span><h2>逐阶段输入、处理与输出</h2><p>每个阶段都给出字段来源、转换规则、输出对象、下游用途、失败分支与可查看的实现代码。</p></div><button class="button" data-expand aria-expanded="false">展开全部阶段</button></div>
    <div class="inspector"><nav class="stage-nav" aria-label="处理阶段">${STAGES.map(s=>`<button class="${s.kind}" data-stage="${s.id}"><span>${s.no}</span>${s.short}</button>`).join('')}<p>点击实现依据可查看当前工作树中的函数快照。</p></nav><div class="stage-content"><div id="stage-detail"></div><div class="stage-pagination" id="stage-pagination"></div></div></div></section>

    <section class="paper-section" id="example"><div class="section-heading"><div><span class="section-number">03 / WORKED EXAMPLE</span><h2>跟随一条证据走完整个 Pipeline</h2><p>三个调用、三个局部候选、两个代表过程。展示“补原文”与“重复查已知条件”怎样得到不同判断，以及重复问题如何归并而不丢失轮次。</p></div><span class="example-label">结构示例 · 非真实任务或模型评测</span></div>
    <div class="example-question"><strong>任务 q</strong>${esc(contract.example.native.question)}</div>
    <div class="turn-tabs" aria-label="示例调用">${contract.example.case.steps.map((s,i)=>`<button data-turn="${i}" aria-pressed="${i===0}">Call ${s.call} · ${i===0?'补充原文':'条件已知后的搜索'}</button>`).join('')}</div><div id="turn-example"></div>
    <div class="lineage-grid"><div class="lineage-cell"><h4>局部候选池 C</h4><ul><li><code>F001 → Call 1</code><br>定向原文核验 · benefit</li><li><code>F002 → Call 2</code><br>无效重复 · missed_opportunity</li><li><code>F003 → Call 3</code><br>无效重复 · missed_opportunity</li></ul></div><span class="lineage-arrow" aria-hidden="true">→</span><div class="lineage-cell"><h4>Summary.process_groups</h4><ul><li><code>representative: F002</code><br>members: [F002, F003]</li><li><code>representative: F001</code><br>members: [F001]</li></ul><p>先放优先问题，后放独立优势。不是按原始调用顺序机械排列。</p></div><span class="lineage-arrow" aria-hidden="true">→</span><div class="lineage-cell"><h4>工作台展示 annotation</h4><ul><li><code>E01 / 内部 F002</code><br>主要问题 · A2<br>Call 3 放入关联轮次</li><li><code>E02 / 内部 F001</code><br>有效做法 · A4</li></ul><p>同一个问题只展开一次；原始记录、引用和归并成员仍可追溯。</p></div></div>
    <div class="callout"><strong>归并不等于丢弃证据。</strong> representative 决定展示的主分析，member_ids 保留候选关系；代码从原始 steps 取回其他轮次的 Graph 与输出，生成 relatedOccurrences。两种相反的判断不会合成一个“模糊的总体评价”。</div>
    <details class="example-io"><summary>查看最终 Annotation 与归并关系的完整 JSON</summary><div class="io-grid">${codePanel('Summary / 归并与维度',json(contract.example.summary),'example-summary')}${codePanel('Annotation / 页面读取对象',json(contract.example.annotation),'example-annotation')}</div></details></section>

    <section class="paper-section" id="contracts"><div class="section-heading"><div><span class="section-number">04 / MODEL CONTRACTS</span><h2>模型究竟收到什么、必须返回什么</h2><p>以下模板、JSON Schema 和示例请求由当前 Python 实现导出。可查看全文或复制，不使用另写的简化版代替真实模板。</p></div></div>
    <div class="contract-controls"><div class="segmented" aria-label="选择模型阶段"><button data-prompt-stage="findings" aria-pressed="true">03 局部评审</button><button data-prompt-stage="summary" aria-pressed="false">06 全局汇总</button></div><div class="segmented" aria-label="选择合同视图"><button data-prompt-view="template" aria-pressed="true">完整 Prompt</button><button data-prompt-view="schema" aria-pressed="false">输出 Schema</button><button data-prompt-view="request" aria-pressed="false">完整示例请求</button></div></div>
    <p class="contract-note" id="contract-note"></p><div class="contract-code" id="contract-display"></div>
    <p class="figure-caption">评审调用使用独立的 LITELLM_OPENAI_API_KEY。此页面不读取或传送凭据，不会启动 Review。两个阶段使用同一模型，没有额外的独立审稿模型。</p></section>

    <section class="paper-section" id="reproduce"><div class="section-heading"><div><span class="section-number">05 / REPRODUCIBILITY & FAILURE MODES</span><h2>预算、状态与可复核边界</h2><p>模型判断、代码保证和尚未覆盖的内容分别说明，避免把可追溯性当成语义正确性的证明。</p></div></div>
    <div class="config-grid"><section><h3>证据裁剪预算</h3>${table(LIMITS,['字段','字符上限','含义'])}<p class="figure-caption">长度按 Python 字符计数，不是 Token。每个包装字段带 truncated 与 original_chars。工具返回先序列化为字符串再裁剪，可能只保留序列化文本的前缀。</p></section><section><h3>当前执行配置</h3>${table([
      ['model / effort',contract.model.model+' / '+contract.model.reasoning_effort,'两个模型阶段共享配置'],
      ['max_completion_tokens',String(contract.model.max_output_tokens),'每个生成请求的输出预算'],
      ['chunk_turns / chars',defaults.chunk_turns+' / '+defaults.chunk_chars,'批次内 step JSON 的软预算'],
      ['max_batches',String(defaults.max_batches),'超过上限，在模型请求前终止'],
      ['结构修正','最多 1 次 / 请求','追加完整错误回答与校验反馈'],
      ['网络重试','最多 2 次 / 生成','仅瞬时网络错误、429、502、503、504；退避 1s、2s'],
      ['temperature / seed','未显式设置','不声称多次评审完全确定'],
    ],['配置','当前值','作用'])}</section></div>
    <div class="formula-strip"><strong>N<sub>generation</sub> = B + 1</strong><p>正常无缓存、无修正时：B 次局部分析，加 1 次全局汇总。每次生成最多允许一次结构修正，每次 HTTP 调用还可有限重试，因此 B + 1 不是实际请求数的严格上限。缓存命中会减少新请求。</p></div>
    <h3 class="block-heading">检查点与产物之间的衔接</h3>${table(ARTIFACTS,['产物','产生 / 消费阶段','保存的内容'])}
    <div class="limitations-grid"><article><h4>续跑与版本一致性</h4><p>fingerprint 由 evidence_sha256、公开模型配置、Prompt 版本、chunk_chars 与 chunk_turns 计算。完成结果直接返回。同版本失败显式重试，验证并复用已完成批次。旧版本失败任务显式重试时先归档旧 JSON，再按新规则分析；已完成旧标注不会自动覆盖。</p></article>
    <article><h4>冻结证据与展示身份</h4><p>服务端的冻结 bundle 哈希用于确认浏览器加载的是同一份包；文件级证据指纹用于 Review 缓存；sourceLocator 用文件路径、digest、行号和 call 定位具体记录。这三个标识作用不同，不能互相替代。</p></article>
    <article><h4>代码验证什么</h4><p>Schema、原文连续引用、同轮来源、分组引用闭合、已知分类前提和部分重复长段落。代码不会判断任意两段文字是否语义相同，也不能证明“Graph 导致了这个行为”。</p></article>
    <article><h4>模型判断什么</h4><p>信息需求与缺口是否对应、搜索是否具有合理目的、现象是否属于同一机制、哪些过程值得优先保留。单次轨迹无法验证整体 Token 节省，因此当前流程禁止 A7；它也不会读取隐藏 Thinking、还原完整 Agent context 或核验所有外部事实。</p></article></div>
    <details class="guard-box"><summary>异常状态与发布条件</summary><ul><li>queued → running → completed：只有整个流程写出有效 Review 后才更新成功状态。</li><li>HTTP、解析或校验失败 → failed；后台进程意外消失 → interrupted。需要显式重试，不自动产生新的付费 Review。</li><li>两次结构校验都失败时保留检查点与请求日志，不发布部分标注。</li><li>reviewer 为模型名，annotation.status=draft；这是 AI 草稿状态，不等于后台任务未完成。</li><li>Review usage 与被评审 Agent / Graph 用量分开。input 包含 Cache，output 包含 reasoning，total=input+output。</li></ul></details>
    <div class="repro-command">生成 / 校验方法快照（无模型调用）<br><code>python -m tools.case_study_studio.build_review_pipeline</code><br><code>python -m tools.case_study_studio.build_review_pipeline --check</code></div>
    </section><footer class="page-footer"><span>BCG · Agent Review Method Notes<br>运行时代码快照 <code>${esc(contract.source_fingerprint.slice(0,16))}</code> · ${Object.keys(contract.source_files).length} 个源文件</span><span>结构示例已通过当前解析与校验代码，未请求模型。<br><a href="./">返回运行与标注工作台 ↗</a></span></footer>`;
  renderStages();renderTurn();renderContract();
}
function showNotice(message){$('#notice').textContent=message;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>{$('#notice').textContent='';},2500);}
async function copy(text){
  if(navigator.clipboard?.writeText){try{await navigator.clipboard.writeText(text);return;}catch{}}
  const input=document.createElement('textarea');input.value=text;input.setAttribute('aria-label','复制内容');input.style.position='fixed';input.style.left='-10000px';document.body.append(input);input.select();
  const done=document.execCommand('copy');input.remove();if(!done)throw new Error('浏览器未允许复制，请选中文本复制');
}
function showSource(key){
  const source=contract.sources[key];if(!source)return;
  $('#source-title').textContent=source.symbol+'()';$('#source-path').textContent=source.path+':'+source.line;
  $('#source-code').innerHTML=source.text.split('\n').map((line,i)=>`<span class="source-line"><b>${source.line+i}</b><span>${esc(line)||' '}</span></span>`).join('');
  $('#source-dialog').showModal();
}
document.addEventListener('click',async event=>{
  const button=event.target.closest('button');if(!button||!contract)return;
  if(button.dataset.stage)chooseStage(button.dataset.stage,true);
  if(button.dataset.step)chooseStage(STAGES[Math.max(0,Math.min(STAGES.length-1,selected+Number(button.dataset.step)))].id,true);
  if(button.hasAttribute('data-expand')){expanded=!expanded;renderStages();}
  if(button.hasAttribute('data-turn')){turn=Number(button.dataset.turn);renderTurn();}
  if(button.dataset.promptStage){promptStage=button.dataset.promptStage;renderContract();}
  if(button.dataset.promptView){promptView=button.dataset.promptView;renderContract();}
  if(button.dataset.source)showSource(button.dataset.source);
  if(button.dataset.copy){try{await copy(codeCache.get(button.dataset.copy)||'');showNotice('已复制完整内容');}catch(error){showNotice(error.message);}}
  if(button.hasAttribute('data-print'))window.print();
  if(button.hasAttribute('data-download')){
    const url=URL.createObjectURL(new Blob([json(contract)],{type:'application/json;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='bcg-agent-review-pipeline-'+contract.prompt_version+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
});
$('#close-source').addEventListener('click',()=>$('#source-dialog').close());
$('#source-dialog').addEventListener('click',event=>{if(event.target===$('#source-dialog'))$('#source-dialog').close();});
window.addEventListener('beforeprint',()=>{if(!contract)return;printExpanded=expanded;expanded=true;renderStages();document.querySelectorAll('.stage-sheet .guard-box').forEach(d=>{d.open=true;});});
window.addEventListener('afterprint',()=>{if(!contract)return;expanded=printExpanded;renderStages();});
window.addEventListener('hashchange',()=>{if(contract&&location.hash.startsWith('#stage-'))chooseStage(location.hash.slice(7),true);});
try{
  const response=await fetch('./review-pipeline-contract.json',{cache:'no-store'});
  if(!response.ok)throw new Error('方法快照加载失败（HTTP '+response.status+'）');
  contract=await response.json();
  if(contract.format!=='bcg-review-method-contract')throw new Error('方法快照格式不匹配');
  render();if(location.hash.startsWith('#stage-'))chooseStage(location.hash.slice(7),true);
}catch(error){$('#main').innerHTML=`<div class="loading" role="alert"><h1>方法页面暂时不可用</h1><p>${esc(error.message)}</p><a class="button" href="./">返回工作台</a></div>`;}

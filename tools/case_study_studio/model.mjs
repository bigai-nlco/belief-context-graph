// Pure data adapters and exports. No network calls; imported evidence is untrusted text.
import {randomId} from './browser_crypto.mjs';
export const DEMO_TITLE = '项目 X 商业运营年份核查';

export function updateDemoTitle(c) {
  if (c.demo === true && c.taskId === 'DEMO-01' && c.title === '试运行 ≠ 正式运营：一次由 Graph 引导的验证') {
    c.title = DEMO_TITLE;
    return true;
  }
  return false;
}

export const DIMENSIONS = [
  {id: 'A1', title: '长程信息复用', short: '记忆', question: '评估关键事实、约束和排除结论在离开近期上下文后的复用情况。'},
  {id: 'A2', title: '减少无效重复搜索', short: '搜索', question: '记录已有搜索结果的复用情况，区分无效重复与时效更新、冲突核验和独立来源验证。'},
  {id: 'A3', title: '定位关键证据缺口', short: '推理', question: '评估 Agent 利用 belief 关系定位缺失前提、开展定向搜索的情况。'},
  {id: 'A4', title: '合理分配验证精力', short: '置信度', question: '评估 Agent 结合 confidence、证据来源与问题重要性分配验证精力的情况。'},
  {id: 'A5', title: '发现冲突并修正判断', short: '修正', question: '记录冲突发现与假设修正过程，核对相关陈述的时间和适用范围。'},
  {id: 'A6', title: '结论与证据可追溯', short: '溯源', question: '检查最终结论的证据链、限定条件和原始来源，记录溯源的完整程度。'},
  {id: 'A7', title: '降低整体上下文开销', short: '效率', question: '在同题可比配置下，对比 Agent + 在线 Graph 的总 Token 和关键证据保留情况。'},
  {id: 'A8', title: '改善多 Agent 协作', short: '协作', question: '评估跨图信息复用、任务分工和重复调查控制，核对证据来源的独立性。'},
];

export const OPTIONS = {
  opportunity: ['有', '无', '无法判断'],
  utilization: ['充分利用', '部分利用', '未利用', '误用', '无法判断', '不适用'],
  effect: ['正向', '无明显变化', '负向', '待验证', '不适用'],
  evidence: ['明确可见', '有较强迹象', '证据不足'],
  fault: ['无明显失效', '信息获取', '建图', 'Graph 选择', '注入表达', 'Agent 使用', '团队协作', '系统执行', '无法判断'],
  result: ['正确 / 达标', '部分正确 / 部分达标', '错误 / 未达标', '未完成', '尚未评判'],
};
export const IMPROVEMENT_STAGES = OPTIONS.fault.filter(stage => !['无明显失效', '无法判断'].includes(stage));

export const uid = randomId;
export const pretty = (x) => JSON.stringify(x, null, 2);
const obj = (x) => x && typeof x === 'object' && !Array.isArray(x);
const arr = (x) => Array.isArray(x) ? x : [];
const string = (x) => typeof x === 'string' ? x : '';
const emptyDimensionRating = () => ({opportunity: '', utilization: '', effect: '', eventIds: [], evidence: '', note: ''});
const dimensionName = value => value.trim().toLocaleLowerCase();
const present = (...xs) => xs.find(x => x !== undefined && x !== null && x !== '');
export const textOf = (x) => {
  if (typeof x === 'string') return x;
  if (Array.isArray(x)) return x.map(textOf).filter(Boolean).join('\n\n');
  if (!obj(x)) return '';
  if (typeof x.text === 'string') return x.text;
  if (typeof x.thinking === 'string') return x.thinking;
  if (['toolCall', 'tool_use', 'function_call'].includes(x.type)) return `${x.name || 'tool'}\n${typeof x.arguments === 'string' ? x.arguments : pretty(x.arguments ?? x.input ?? {})}`;
  if (x.type === 'function_call_output') return textOf(x.output);
  return textOf(x.content ?? x.summary ?? '');
};

export function newAnnotation() {
  return {reviewer: '', status: 'draft', goal: '', criteria: '', settings: '', team: '单 Agent', result: '', noApplicableDimensions: false,
    customDimensions: [], dimensions: Object.fromEntries(DIMENSIONS.map(d => [d.id, emptyDimensionRating()])),
    events: [], conclusion: {benefit: '', problem: '', evidence: '', improvement: '', comparison: '', focus: [], verification: ''}};
}

export function allDimensions(c) {
  return [...DIMENSIONS, ...arr(c.annotation.customDimensions).map(d => ({...d, short: '自定义', custom: true}))];
}

function customDimensionText(c, title, question, exceptId = '') {
  if (typeof title !== 'string' || !title.trim() || title.trim().length > 60) throw new Error('维度名称需填写 1–60 个字符。');
  if (typeof question !== 'string' || question.trim().length > 500) throw new Error('维度说明最多 500 个字符。');
  if (allDimensions(c).some(d => d.id !== exceptId && dimensionName(d.title) === dimensionName(title))) throw new Error('已有同名维度，请使用不同的名称。');
  return {title: title.trim(), question: question.trim()};
}

export function addCustomDimension(c, title, question = '') {
  const text = customDimensionText(c, title, question);
  const custom = arr(c.annotation.customDimensions);
  if (custom.length >= 100) throw new Error('每个 Case 最多添加 100 个自定义维度。');
  let number = 1;
  while (Object.hasOwn(c.annotation.dimensions, `C${number}`)) number++;
  const dimension = {id: `C${number}`, ...text};
  c.annotation.customDimensions = [...custom, dimension];
  c.annotation.dimensions[dimension.id] = {...emptyDimensionRating(), opportunity: '有'};
  c.annotation.noApplicableDimensions = false;
  return dimension;
}

export function updateCustomDimension(c, id, title, question = '') {
  const dimension = arr(c.annotation.customDimensions).find(d => d.id === id);
  if (!dimension) throw new Error('未找到此自定义维度。');
  Object.assign(dimension, customDimensionText(c, title, question, id));
  return dimension;
}

export function removeCustomDimension(c, id) {
  if (!arr(c.annotation.customDimensions).some(d => d.id === id)) return false;
  c.annotation.customDimensions = c.annotation.customDimensions.filter(d => d.id !== id);
  delete c.annotation.dimensions[id];
  for (const event of c.annotation.events) if (Array.isArray(event.dimensions)) event.dimensions = event.dimensions.filter(value => value !== id);
  c.annotation.noApplicableDimensions = false;
  return true;
}

export function newEvent(ref = {}) {
  return {id: uid(), label: '', agent: '', reference: '', quote: '', objective: '', need: '', original: '', ingestion: '', visible: '', alternatives: '', behavior: '', impact: '', fault: '', strength: '', dimensions: [], ...ref};
}

export function selectedDimensions(c) {
  // Keep older uncertain ratings visible until the reviewer confirms them from the trajectory.
  return allDimensions(c).filter(d => ['有', '无法判断'].includes(c.annotation.dimensions[d.id].opportunity));
}

export function hasNoApplicableDimensions(c) {
  if (selectedDimensions(c).length) return false;
  const a = c.annotation;
  return a.noApplicableDimensions === true ||
    (a.noApplicableDimensions === undefined && allDimensions(c).every(d => a.dimensions[d.id].opportunity === '无'));
}

export function setDimensionSelected(c, dimensionId, selected) {
  if (!allDimensions(c).some(d => d.id === dimensionId) || typeof selected !== 'boolean') return false;
  const row = c.annotation.dimensions[dimensionId];
  row.opportunity = selected ? '有' : '无';
  c.annotation.noApplicableDimensions = false;
  // Deselecting retains the draft and its source links. A newly selected dimension needs a real rating.
  if (selected) for (const key of ['utilization', 'effect']) if (row[key] === '不适用') row[key] = '';
  return true;
}

export function confirmNoApplicableDimensions(c, confirmed) {
  if (selectedDimensions(c).length || typeof confirmed !== 'boolean') return false;
  c.annotation.noApplicableDimensions = confirmed;
  if (confirmed) for (const row of Object.values(c.annotation.dimensions)) row.opportunity = '无';
  return true;
}

export function linkedEvents(c, dimensionId) {
  const ids = new Set(arr(c.annotation.dimensions[dimensionId]?.eventIds));
  return c.annotation.events.filter(e => ids.has(e.id));
}

export function setDimensionEvent(c, dimensionId, eventId, selected) {
  if (!allDimensions(c).some(d => d.id === dimensionId) || !c.annotation.events.some(e => e.id === eventId)) return false;
  const row = c.annotation.dimensions[dimensionId];
  const ids = new Set(linkedEvents(c, dimensionId).map(e => e.id));
  if (selected) ids.add(eventId); else ids.delete(eventId);
  row.eventIds = [...ids];
  return true;
}

export function removeEvent(c, eventId) {
  if (!c.annotation.events.some(e => e.id === eventId)) return false;
  c.annotation.events = c.annotation.events.filter(e => e.id !== eventId);
  for (const row of Object.values(c.annotation.dimensions)) {
    if (Array.isArray(row.eventIds)) row.eventIds = row.eventIds.filter(id => id !== eventId);
  }
  return true;
}

export function eventLabel(c, e) {
  return `E${String(c.annotation.events.findIndex(v => v.id === e.id) + 1).padStart(2, '0')} · ${e.label || '关键过程'}`;
}

export function conclusionObservations(c) {
  // Derived from the reviewer's annotations, never from the task type or model-generated judgments.
  return selectedDimensions(c).map(d => {
    const row = c.annotation.dimensions[d.id];
    return {...d, utilization: row.utilization, effect: row.effect, note: row.note,
      pending: row.opportunity === '无法判断', events: linkedEvents(c, d.id)};
  });
}

export function setConclusionFocus(c, stage, selected) {
  if (!IMPROVEMENT_STAGES.includes(stage) || typeof selected !== 'boolean') return false;
  const focus = new Set(arr(c.annotation.conclusion.focus));
  if (selected) focus.add(stage); else focus.delete(stage);
  c.annotation.conclusion.focus = IMPROVEMENT_STAGES.filter(value => focus.has(value));
  return true;
}

function dimensionEvidence(c, dimensionId) {
  const row = c.annotation.dimensions[dimensionId];
  return [
    ...linkedEvents(c, dimensionId).map(e => `${eventLabel(c, e)}${e.reference ? ' · ' + e.reference : ''}`),
    ...(row.evidence?.trim() ? [`原有证据备注：${row.evidence}`] : []),
  ].join('\n');
}

export function parseFile(name, text) {
  const clean = text.replace(/^\uFEFF/, '');
  if (/\.(md|txt)$/i.test(name)) return {type: 'text', data: clean, warnings: []};
  if (/\.(jsonl|ndjson)$/i.test(name)) {
    const records = [], warnings = [];
    // Legacy streaming files can be >80 MB for only a few complete turns.
    // Skip intermediate copies BEFORE JSON parsing, preserving original line numbers.
    const hasCompleted = /^\s*\{\s*"type"\s*:\s*"message_end"/m.test(clean);
    const intermediate = /^\s*\{\s*"type"\s*:\s*"(?:message_update|message_start|turn_end|agent_end)"/;
    clean.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return;
      if (hasCompleted && intermediate.test(line)) return;
      try { records.push({line: i + 1, value: JSON.parse(line)}); }
      catch { warnings.push(`${name} 第 ${i + 1} 行不是有效 JSON，未作为证据导入。`); }
    });
    if (!records.length) throw new Error(`${name} 没有可读取的 JSONL 记录。`);
    return {type: 'jsonl', data: records, warnings};
  }
  try { return {type: 'json', data: JSON.parse(clean), warnings: []}; }
  catch { throw new Error(`${name} 不是有效 JSON。支持 JSON、JSONL、Markdown 和 TXT。`); }
}

const toolBlockTypes = new Set(['toolCall', 'tool_use', 'function_call']);
const thinkingBlockTypes = new Set(['thinking', 'reasoning', 'reasoning_text', 'summary_text']);

function toolCallView(call) {
  const fn = obj(call.function) ? call.function : call;
  let args = fn.arguments ?? fn.args ?? fn.input;
  if (typeof args === 'string') {
    try { args = JSON.parse(args); } catch { /* Keep malformed/incomplete arguments visible. */ }
  }
  return {name: string(fn.name) || '未命名工具', arguments: args};
}

function recordBody(value, role) {
  const m = obj(value.message) ? value.message : value;
  const request = value.payload ?? value.request ?? (value.input_messages ? {messages: value.input_messages} : null);
  const body = m.content ?? m.text ?? value.result ?? value.answer ?? value.output ??
    (value.t === 'start' ? value.user_message : undefined);
  const blocks = role === 'assistant' ? arr(body) : [];
  const explicitCalls = arr(m.tool_calls ?? m.toolCalls ?? value.tool_calls).filter(obj);
  const calls = explicitCalls.length ? explicitCalls : blocks.filter(b => obj(b) && toolBlockTypes.has(b.type));
  const content = textOf(role === 'assistant' && Array.isArray(body)
    ? blocks.filter(b => !toolBlockTypes.has(b?.type) && !thinkingBlockTypes.has(b?.type))
    : body);
  const thinking = textOf(m.reasoning_content ?? m.thinking ?? m.reasoning) ||
    textOf(blocks.filter(b => obj(b) && thinkingBlockTypes.has(b.type)));
  return {
    // An empty Assistant content is not a JSON response. Keep log metadata in raw.
    text: role === 'assistant' ? content : content || (request ? pretty(request) : pretty(value)),
    thinking, tools: calls.length ? pretty(calls) : '', toolCalls: calls.map(toolCallView),
  };
}

function recordOf(value, index, sourceId) {
  if (!obj(value)) value = {content: String(value ?? '')};
  const m = obj(value.message) ? value.message : value;
  const role = m.role ?? (value.t === 'llm' ? 'assistant' : ['tool', 'result'].includes(value.t) ? 'tool' : value.type ?? value.t ?? 'record');
  const call = present(value.call_id, value.call, value.turn, value.turn_index);
  return {id: `${sourceId}:${index}`, index, role, ...(call === undefined ? {} : {call}),
    title: value.t === 'start' ? '任务开始' : `${role}${value.toolName || value.name ? ' · ' + (value.toolName || value.name) : ''}`,
    ...recordBody(value, role), timestamp: value.timestamp ?? '', raw: value};
}

/** Refresh the display of older saved cases without rewriting evidence or citations. */
export function recordPresentation(record) {
  if (obj(record.raw)) return {...record, ...recordBody(record.raw, record.role)};
  let calls = [];
  try { calls = arr(JSON.parse(record.tools || '[]')).filter(obj); } catch {}
  return {...record, toolCalls: record.toolCalls ?? calls.map(toolCallView)};
}

function agentMetadata(value) {
  if (!obj(value)) return null;
  const id = string(value.agent_id), name = string(value.agent_name), role = string(value.role_id);
  if (!id && !name && !role) return null;
  return {id, name, role, task: String(value.task_id ?? value.taskId ?? '')};
}

function isAgentRecord(record) {
  const raw = record.raw;
  if (obj(raw) && (raw.system_memory !== undefined || raw.system_context !== undefined ||
      raw.selected_node_ids !== undefined || raw.payload || raw.request || raw.input_messages)) return false;
  return ['user', 'assistant', 'tool', 'function', 'system', 'developer'].includes(record.role) ||
    ['start', 'llm', 'tool', 'result', 'end', 'error'].includes(raw?.t);
}

export function agentTrajectories(c) {
  const variants = new Map();
  for (const source of c.sources) {
    if (source.kind !== 'trajectory') continue;
    const records = source.records.filter(isAgentRecord);
    if (!records.some(r => !['system', 'developer'].includes(r.role))) continue;
    const start = source.records.find(r => r.raw?.t === 'start')?.raw;
    const metadata = source.agent ?? agentMetadata(start);
    const owner = metadata ? JSON.stringify([metadata.id || metadata.role || metadata.name, metadata.task]) : '';
    const path = c.files.find(f => f.id === source.fileId)?.path ?? source.label;
    const stem = path.replace(/\.(?:jsonl?|ndjson)$/i, '');
    if (!variants.has(stem)) variants.set(stem, []);
    variants.get(stem).push({source, records, metadata, owner, jsonl: /\.(jsonl|ndjson)$/i.test(path)});
  }
  const groups = new Map();
  for (const [stem, candidates] of variants) {
    // JSON/JSONL exports of the same trajectory are alternate representations,
    // not extra Agents. Prefer the most complete record set; ties favor JSONL.
    const knownOwners = [...new Set(candidates.map(c => c.owner).filter(Boolean))];
    const byOwner = new Map();
    for (const candidate of candidates) {
      const owner = candidate.owner || (knownOwners.length === 1 ? knownOwners[0] : '');
      if (!byOwner.has(owner)) byOwner.set(owner, []);
      byOwner.get(owner).push(candidate);
    }
    for (const [owner, copies] of byOwner) {
      const chosen = [...copies].sort((a, b) => b.records.length - a.records.length || Number(b.jsonl) - Number(a.jsonl))[0];
      const metadata = chosen.metadata ?? copies.find(c => c.metadata)?.metadata;
      const key = owner ? `agent:${owner}` : `trajectory:${stem}`;
      if (!groups.has(key)) groups.set(key, {id:key, name:metadata?.name || metadata?.role || metadata?.id || 'Agent', records:[]});
      // Keep original record objects/IDs so citations resolve to their original files.
      groups.get(key).records.push(...chosen.records);
    }
  }
  const agents = [...groups.values()];
  const counts = new Map(), seen = new Map();
  for (const agent of agents) counts.set(agent.name, (counts.get(agent.name) || 0) + 1);
  for (const agent of agents) {
    const name = agent.name, n = (seen.get(name) || 0) + 1; seen.set(name, n);
    if (counts.get(name) > 1) agent.name = `${name} ${n}`;
  }
  return agents;
}

/** Display only the recorded graph block; preserve the full input in source data. */
export function contextBeliefText(context) {
  const blocks = [...(context?.text || '').matchAll(/<belief_graph>([\s\S]*?)<\/belief_graph>/g)];
  // Multiple graph blocks may use different ID namespaces. Do not merge them.
  return blocks.length === 1 ? blocks[0][1].trim() : null;
}

/** Join an injected context to its subsequent Agent output, never by row number. */
export function contextTurn(c, context) {
  const empty = (status, extra = {}) => ({status, agent: null, call: null, assistant: null, results: [], ...extra});
  if (!context) return empty('missing-context');
  const callKey = value => typeof value === 'number' && Number.isFinite(value) ? String(value)
    : typeof value === 'string' && value.trim() ? value.trim() : null;
  // A server Graph turn is not an Agent call. Only explicit local-call metadata
  // can associate legacy render logs with an Agent trajectory.
  const rawCall = context.kind === 'graph_context'
    ? present(context.raw?.local_call, context.raw?.agent_call)
    : present(context.call, context.raw?.call, context.raw?.local_call, context.raw?.call_id);
  const call = callKey(rawCall);
  if (call === null) return empty('missing-call');
  const agents = agentTrajectories(c);
  const filePath = fileId => (c.files.find(f => f.id === fileId)?.path || '').replaceAll('\\', '/').replace(/^\.\//, '');
  const ownerPath = path => {
    const parts = path.split('/');
    const boundary = parts.findIndex(p => p === 'bcg' || p === 'trajectories');
    return boundary >= 0 ? parts.slice(0, boundary).join('/') : parts.slice(0, -1).join('/');
  };
  const requestedOwner = agentMetadata(context.raw);
  const requestedIdentity = requestedOwner?.id || requestedOwner?.role || requestedOwner?.name;
  const contextPath = filePath(context.fileId);
  const contextOwner = ownerPath(contextPath);
  const candidates = agents.map(agent => {
    const ids = new Set(agent.records.map(r => r.id));
    const sources = c.sources.filter(source => source.kind === 'trajectory' && source.records.some(r => ids.has(r.id)));
    const metadata = sources.map(source => source.agent ?? agentMetadata(source.records.find(r => r.raw?.t === 'start')?.raw)).filter(Boolean);
    return {agent, sources, metadata};
  });
  let owners = candidates;
  if (requestedIdentity) {
    owners = owners.filter(({metadata}) => metadata.some(m =>
      [m.id, m.role, m.name].includes(requestedIdentity) && (!requestedOwner.task || !m.task || m.task === requestedOwner.task)));
    if (!owners.length) return empty('unmatched-agent', {call});
  }
  // Prefer explicit source ownership even when only one other Agent was imported.
  // This prevents joining researcher A's audit to researcher B's matching call.
  if (contextPath) {
    const sameSource = owners.filter(x => x.sources.some(s => s.fileId === context.fileId));
    const sameOwner = owners.filter(x => x.sources.some(s => ownerPath(filePath(s.fileId)) === contextOwner));
    if (sameSource.length) owners = sameSource;
    else if (sameOwner.length) owners = sameOwner;
    else if (!requestedIdentity && contextOwner && owners.some(x => x.sources.some(s => ownerPath(filePath(s.fileId))))) {
      return empty('unmatched-agent', {call});
    }
  }
  if (owners.length !== 1) return empty(owners.length ? 'ambiguous-agent' : 'missing-trajectory', {call});
  const agent = owners[0].agent;
  const assistants = agent.records.filter(r => r.role === 'assistant' && callKey(r.call) === call);
  if (assistants.length !== 1) return empty(assistants.length ? 'ambiguous-call' : 'missing-output', {agent, call});
  const assistant = assistants[0], start = agent.records.indexOf(assistant);
  const next = agent.records.findIndex((r, i) => i > start && r.role === 'assistant');
  const results = agent.records.slice(start + 1, next === -1 ? undefined : next)
    .filter(r => ['tool', 'function'].includes(r.role) && callKey(r.call) === call);
  return {status: 'matched', agent, call, assistant, results};
}

function trajectoryQuote(view) {
  return [view.thinking ? 'Thinking:\n' + view.thinking : '', view.text,
    view.tools ? 'Tool calls:\n' + view.tools : ''].filter(Boolean).join('\n\n') || pretty(view.raw) || '';
}

/** Render the saved citation without expanding an explicitly selected excerpt. */
export function eventEvidencePresentation(c, event) {
  const quote = string(event?.quote);
  const ref = string(event?.sourceRef);
  if (quote && ref.startsWith('record:')) {
    const record = arr(c?.sources).flatMap(s => arr(s.records)).find(r => r.id === ref.slice(7));
    if (record) {
      const view = recordPresentation(record);
      if (quote === trajectoryQuote(view) || (obj(record.raw) && quote === pretty(record.raw))) {
        return {kind: 'record', record: view};
      }
    }
  }
  // Selected, truncated, legacy or detached quotations stay exactly as saved.
  return {kind: 'excerpt', text: quote};
}

/** Resolve the recorded injection for a cited step, without changing the citation. */
export function eventBeliefContext(c, event) {
  const empty = (status, extra = {}) => ({status, context: null, text: null, call: null, agent: null, ...extra});
  const ref = string(event?.sourceRef);
  const record = ref.startsWith('record:')
    ? arr(c?.sources).flatMap(s => arr(s.records)).find(r => r.id === ref.slice(7)) : null;
  if (!record) return empty('missing-record');
  if (!['assistant', 'tool', 'function'].includes(record.role)) return empty('not-agent-call');
  // Reuse the same Agent/call mapping as the per-turn view. Tool-result citations
  // get the injection before their producing call, never the next call's graph.
  const matches = arr(c.contexts).map(context => ({context, turn: contextTurn(c, context)}))
    .filter(({turn}) => turn.status === 'matched' &&
      (turn.assistant.id === record.id || turn.results.some(r => r.id === record.id)))
    .map(match => ({...match, text: contextBeliefText(match.context)}));
  if (!matches.length) return empty('missing-context', {call: record.call ?? null});
  // Repeated copies of one injection are fine; conflicting versions are not.
  if (new Set(matches.map(m => m.text ?? '')).size > 1) {
    return empty('ambiguous-context', {call: record.call ?? null});
  }
  const match = matches.find(m => m.text !== null) ?? matches[0];
  const {context, text, turn} = match;
  const knownEmpty = text === '' || (context.kind === 'injection_audit' && context.raw?.graph_chars === 0 && !context.fallback);
  return {status: text ? 'matched' : knownEmpty ? 'empty' : 'missing-graph',
    context, text, call: turn.call, agent: turn.agent};
}

/** Only visible Agent execution records may seed new critical events. */
export function eventFromTrajectory(c, ref, selected = '') {
  if (!c || typeof ref !== 'string' || !ref.startsWith('record:')) return null;
  const agent = agentTrajectories(c).find(a => a.records.some(r => r.id === ref.slice(7)));
  const record = agent?.records.find(r => r.id === ref.slice(7));
  if (!record) return null;
  const source = c.sources.find(s => s.records.includes(record));
  const file = c.files.find(f => f.id === source.fileId);
  const view = recordPresentation(record);
  let quote = trajectoryQuote(view);
  if (selected && quote.includes(selected)) quote = selected;
  if (quote.length > 12000) quote = quote.slice(0, 12000) + '\n[摘录截断；完整内容见来源记录]';
  return newEvent({
    sourceRef: ref,
    reference: `${source.label} [SHA-256 ${file?.digest.slice(0, 16)}] · 记录 ${record.index}${record.call !== undefined ? ' · call ' + record.call : ''}`,
    quote, label: `记录 ${record.index} · ${record.role}`, agent: agent.name,
  });
}

function graphOf(value, sourceId, label) {
  const v = obj(value.snapshot) ? value.snapshot : obj(value.graph) ? value.graph : value;
  const nodes = v.nodes ?? v.all_nodes ?? (Array.isArray(v.beliefs) ? [...v.beliefs, ...arr(v.decisions)] : null);
  if (!Array.isArray(nodes)) return null;
  return {id: `${sourceId}:graph`, fileId: sourceId, label, graphId: String(value.problem_id ?? value.graph_id ?? v.problem_id ?? value.item_id ?? ''),
    // No temporal reconstruction from final confidence/content: retain the uploaded snapshot as-is.
    kind: 'uploaded_snapshot', scope: 'full', nodes: nodes.filter(obj), relations: arr(v.relations).filter(obj)};
}

export function graphViews(c) {
  const views = {full: [], turns: []};
  for (const graph of c.graphs) {
    // Earlier saved workspaces identified JSONL snapshots only by their line index.
    const isTurn = graph.scope === 'turns' || (graph.scope === undefined && Number.isInteger(graph.index));
    views[isTurn ? 'turns' : 'full'].push(graph);
  }
  return views;
}

function contextOf(value, sourceId, index) {
  if (!obj(value)) return null;
  const request = value.payload ?? value.request;
  const messages = request?.messages ?? request?.input ?? value.input_messages;
  if (Array.isArray(messages)) {
    const systems = messages.filter(m => ['system', 'developer'].includes(m.role));
    const text = systems.map(m => textOf(m.content)).join('\n\n');
    if (text) return {id: `${sourceId}:request:${index}`, fileId: sourceId, index, kind: 'model_request',
      label: `请求 ${value.call_id ?? value.call ?? index} · System / Developer`, text,
      call: present(value.call_id, value.call) ?? null, graphId: '', nodeIds: [], relationIds: [], raw: value};
  }
  const text = value.system_memory ?? value.system_context ?? (value.view !== undefined ? value.text : null);
  const isAudit = Object.hasOwn(value, 'selected_node_ids') || Object.hasOwn(value, 'fallback');
  if (typeof text !== 'string' && !isAudit) return null;
  const kind = value.view !== undefined ? 'graph_context' : 'injection_audit';
  return {id: `${sourceId}:context:${index}`, fileId: sourceId, index, kind,
    label: value.call !== undefined ? `本地调用 ${value.call}` : value.local_call !== undefined ? `本地调用 ${value.local_call}` : `Graph turn ${value.streamTurnIndex ?? index}`,
    text: text ?? '', call: present(value.call, value.local_call) ?? null, graphId: value.graph_id ?? value.problem_id ?? '',
    nodeIds: arr(value.selected_node_ids ?? value.selectedNodeIds), relationIds: arr(value.selected_relation_ids ?? value.selectedRelationIds),
    fallback: value.fallback ?? null, raw: value};
}

export function ingestFiles(files, existing = null) {
  const c = existing ? structuredClone(existing) : {id: uid(), createdAt: new Date().toISOString(), updatedAt: '', taskId: '', taskIdVerified: false, title: '', question: '',
    benchmark: '', model: '', mode: '', report: '', metrics: {}, files: [], sources: [], contexts: [], graphs: [], warnings: [], annotation: newAnnotation()};
  const parsed = files.filter(f => !c.files.some(old => old.digest === f.digest && old.path === f.path)).map(f => ({...f, parsed: parseFile(f.path, f.text)}));
  const taskDocs = parsed.filter(f => obj(f.parsed.data) && f.parsed.data.task_id !== undefined);
  const results = taskDocs.filter(f => Object.hasOwn(f.parsed.data, 'benchmark'));
  const primaryDocs = results.length ? results : taskDocs;
  const taskIds = new Set(primaryDocs.map(f => String(f.parsed.data.task_id)));
  const modes = new Set(results.map(f => f.parsed.data.mode).filter(Boolean));
  if (modes.size > 1 || (existing?.mode && modes.size && !modes.has(existing.mode))) throw new Error('这些文件包含不同模式的实验，请分别导入为独立 Case。');
  if (existing?.taskIdVerified && taskIds.size && !taskIds.has(existing.taskId)) throw new Error('附件的 Task ID 与当前 Case 不一致。请作为新的 Case 导入。');
  if (taskIds.size > 1) throw new Error('这些文件包含多道题。每次请选择同一道题的文件或单题目录，避免混合轨迹。');
  for (const f of parsed) {
    const id = uid(), data = f.parsed.data;
    c.files.push({id, path: f.path, digest: f.digest, size: f.size ?? f.text.length, kind: f.parsed.type});
    c.warnings.push(...f.parsed.warnings);
    if (f.parsed.type === 'text') {
      c.sources.push({id, fileId: id, label: f.path, kind: 'document', records: [recordOf({role: 'document', content: data}, 1, id)]});
      if (/report|answer|final/i.test(f.path)) c.report = data;
      continue;
    }
    let records = [];
    let kind = 'records', agent = null;
    if (f.parsed.type === 'jsonl') {
      // Prefer completed messages; streaming deltas are not distinct model turns.
      const hasCompleted = data.some(r => r.value?.type === 'message_end');
      records = data.filter(r => !['message_update', 'message_start', 'turn_end', 'agent_end'].includes(r.value?.type) || !hasCompleted)
        .map(r => recordOf(r.value, r.line, id));
      const start = data.find(r => r.value?.t === 'start')?.value;
      agent = agentMetadata(start);
      if (start) {
        c.question ||= string(start.user_message);
        c.model ||= string(start.model_name);
      }
      if (!c.question) c.question = textOf(data.find(r => r.value?.type === 'message_end' && r.value?.message?.role === 'user')?.value.message.content);
      data.forEach(({value, line}) => {
        const ctx = contextOf(value, id, line); if (ctx) c.contexts.push(ctx);
        const graph = graphOf(value, id, `${f.path} · 行 ${line} · Graph turn ${value.stream_turn_index ?? '—'}`);
        if (graph) c.graphs.push({...graph, id: `${id}:graph:${line}`, index: line, scope: 'turns'});
      });
      kind = /model.?io/i.test(f.path) ? 'model_io' : /requests|context/i.test(f.path) ? 'context_audit' : 'trajectory';
      if (data.some(({value}) => Array.isArray(value.nodes) || Array.isArray(value.snapshot?.nodes))) kind = 'graph_snapshots';
    } else if (Array.isArray(data)) {
      records = data.map((r, i) => recordOf(r, i + 1, id));
      data.forEach((r, i) => { const ctx = contextOf(r, id, i + 1); if (ctx) c.contexts.push(ctx); });
      kind = 'trajectory';
    } else if (obj(data)) {
      agent = agentMetadata(data);
      const explicitTask = data.task_id ?? data.taskId;
      if (explicitTask !== undefined && primaryDocs.includes(f)) { c.taskId = String(explicitTask); c.taskIdVerified = true; }
      c.question = string(data.question ?? data.task_description ?? data.prompt) || c.question;
      c.title = string(data.title ?? data.name) || c.title;
      c.benchmark = string(data.benchmark) || c.benchmark;
      if (explicitTask !== undefined || data.history) c.mode = string(data.mode) || c.mode;
      c.report = string(data.final_response ?? data.final_answer ?? data.answer) || c.report;
      if (explicitTask !== undefined) {
        const metrics = {usage: data.usage, graph_usage: data.graph_usage, duration_seconds: data.wall_time_seconds ?? data.duration_seconds,
          search_calls: data.search_calls, status: data.status, correct: data.correct, official_score: data.official_score};
        for (const [k, v] of Object.entries(metrics)) if (v !== undefined) c.metrics[k] = v;
      }
      const graph = graphOf(data, id, f.path);
      if (graph) c.graphs.push(graph);
      const messages = data.history ?? data.messages ?? (Array.isArray(data.trajectory) ? data.trajectory : null) ?? data.acknowledged ?? data.events ?? data.turns;
      if (Array.isArray(messages)) {
        records = messages.map((r, i) => recordOf(r, i + 1, id));
        kind = data.acknowledged || data.n_turns !== undefined || (graph && !data.messages && !data.history) ? 'graph_trajectory' : 'trajectory';
        if (!c.question) c.question = textOf(messages.find(m => m.role === 'user')?.content);
      } else records = [recordOf(data, 1, id)];
      const ctx = contextOf(data, id, 1); if (ctx) c.contexts.push(ctx);
      if (!graph) c.model = string(data.model ?? data.model_name) || c.model;
    } else throw new Error(`${f.path} 的顶层必须是 JSON 对象或数组。`);
    c.sources.push({id, fileId: id, label: f.path, kind, records, ...(agent ? {agent} : {})});
  }
  c.taskId ||= c.files[0]?.path.split('/').at(-1)?.replace(/\.[^.]+$/, '') || 'untitled';
  c.title ||= c.question ? c.question.slice(0, 72) + (c.question.length > 72 ? '…' : '') : `Case ${c.taskId}`;
  c.updatedAt = new Date().toISOString();
  return c;
}

export function completion(c) {
  const a = c.annotation, dimensions = selectedDimensions(c);
  const completed = dimensions.filter(d => {
    const row = a.dimensions[d.id];
    return row.opportunity === '有' && row.utilization && row.utilization !== '不适用' &&
      row.effect && row.effect !== '不适用' && row.note.trim() && linkedEvents(c, d.id).length;
  }).length;
  return {completed, total: dimensions.length, percent: dimensions.length ? Math.round(completed / dimensions.length * 100) : hasNoApplicableDimensions(c) ? 100 : 0};
}

export function validateReview(c) {
  const a = c.annotation, missing = [];
  if (!a.reviewer.trim()) missing.push('标注人');
  if (!a.result) missing.push('任务结果');
  const progress = completion(c);
  if (!progress.total && !hasNoApplicableDimensions(c)) missing.push('根据关键过程选择维度，或确认未发现相关维度');
  else if (progress.completed !== progress.total) missing.push('已选维度的评价');
  if (!a.conclusion.benefit.trim() && !a.conclusion.problem.trim()) missing.push('Case 结论');
  if (!a.conclusion.evidence) missing.push('结论的证据强度');
  for (const d of selectedDimensions(c)) {
    const row = a.dimensions[d.id];
    if (row.opportunity === '无法判断') missing.push(`${d.id} 根据关键过程确认影响可能`);
    const events = linkedEvents(c, d.id);
    if (!events.length) missing.push(`${d.id} 关联的关键过程`);
    else if (['正向', '负向'].includes(row.effect) && !events.some(e => e.reference && e.behavior.trim() && e.impact.trim() && e.strength)) {
      missing.push(`${d.id} 关联的关键过程需包含来源引用、实际行为、后续影响和证据强度`);
    }
  }
  return missing;
}

export function tokenMetrics(c) {
  const u = c.metrics.usage ?? {};
  const total = v => {
    if (!obj(v)) return null;
    const t = present(v.total_tokens, v.total, v.totals?.total_tokens);
    return typeof t === 'number' && Number.isFinite(t) ? t : null;
  };
  const agent = total(u.agent ?? u), graph = total(u.graph ?? c.metrics.graph_usage);
  return {agent, graph, combined: agent !== null && graph !== null ? agent + graph : null,
    graphScope: u.graph ? '含最终建图（源文件统计口径）' : c.metrics.graph_usage ? '源文件 graph_usage（请核对是否仅在线）' : '未知'};
}

const mdCell = v => String(v ?? '').replace(/\|/g, '\\|').replace(/\r?\n/g, '<br>');
export function markdownReport(c) {
  const a = c.annotation;
  const lines = [`# Case Study · ${c.taskId}`, '', `标注人：${a.reviewer || '未填写'}  `, `状态：${a.status === 'reviewed' ? '已完成' : '草稿'}  `,
    `导出时间：${new Date().toISOString()}`, '', '## 任务', '', c.question || c.title, '',
    ...(a.criteria?.trim() ? [`成功标准：${a.criteria}`, ''] : []), `结果：${a.result}`, '',
    ...(a.settings?.trim() ? ['## 运行配置', '', a.settings, ''] : []),
    ];
  if(a.resultAssessment){
    const assessment=a.resultAssessment;
    lines.push('## 任务结果 · AI 评估', '', '非 Benchmark 官方评分。', '', `AI 结果：${assessment.result}`, '', assessment.rationale, '');
    for(const [key,label] of [['task_quotes','任务要求原文'],['report_quotes','最终报告原文']]){
      lines.push(`### ${label}`, '');
      for(const quote of assessment[key]||[])lines.push('> '+quote.replace(/\n/g,'\n> '), '');
    }
    if(assessment.limitations?.length)lines.push('### 判断范围与待核实内容', '', ...assessment.limitations.map(item=>'- '+item), '');
  }
  lines.push('## BCG 利用情况', '');
  const dimensions = selectedDimensions(c);
  if (dimensions.length) lines.push('以下维度由标注人根据轨迹中的关键过程选择。', '', '| 维度 | 是否利用 | 是否产生效果 | 关联关键过程 | 判断依据 |', '|---|---|---|---|---|');
  else lines.push(hasNoApplicableDimensions(c) ? '已查看轨迹，未发现相关维度。' : '尚未选择维度。');
  for (const d of dimensions) {
    const r = a.dimensions[d.id];
    const links = linkedEvents(c, d.id).map(e => {
      const label = eventLabel(c, e).replace(/[[\]\\<>]/g, '\\$&');
      return `[${label}](#process-${a.events.indexOf(e) + 1})`;
    });
    if (r.evidence?.trim()) links.push(`原有证据备注：${r.evidence}`);
    lines.push(`| ${mdCell(`${d.id} ${d.title}${r.opportunity === '无法判断' ? '（待确认）' : ''}`)} | ${mdCell(r.utilization)} | ${mdCell(r.effect)} | ${mdCell(links.join('\n'))} | ${mdCell(r.note)} |`);
  }
  for (const d of dimensions.filter(d => d.custom && d.question)) lines.push('', `${d.id} 维度说明：${d.question}`);
  lines.push('', '## 关键过程', '');
  for (const [i, e] of a.events.entries()) {
    lines.push(`<a id="process-${i + 1}"></a>`, '', `### ${eventLabel(c, e)}`, '');
    for (const [key, label] of [['need', 'BCG 应发挥的作用'], ['behavior', '实际发生的情况'], ['impact', '影响与改进线索'], ['fault', '归因环节'], ['strength', '证据强度'], ['reference', '证据定位'], ['quote', '证据摘录'], ['agent', 'Agent']]) {
      lines.push(`**${label}**`, '', e[key] || '未填写', '');
    }
    const supplemental = [['objective', '当前子问题'], ['original', '原始来源'], ['ingestion', '入图情况'], ['visible', '实际可见内容'], ['alternatives', '其他来源与影响因素']].filter(([key]) => e[key]?.trim());
    if (supplemental.length) lines.push('#### 补充证据检查', '');
    for (const [key, label] of supplemental) lines.push(`**${label}**`, '', e[key], '');
    if(e.searchAssessment){
      const s=e.searchAssessment;
      lines.push('<details>', `<summary>搜索判断 · ${s.verdict}</summary>`, '');
      for(const [key,label] of [['information_need','信息需求'],['graph_sufficiency','Graph 是否已满足'],['verification_reason','核验目的'],['expected_gain','预期新增内容'],['basis','判断依据'],['observed_gain','实际新增证据'],['result_quote','工具结果原文'],['unavailable_quote','事前已知的工具状态']]){
        if(s[key])lines.push(`${label}：${s[key]}`, '');
      }
      lines.push('</details>', '');
    }
    if(e.relatedOccurrences?.length){
      lines.push('<details>', '<summary>同类过程与关联轮次</summary>', '', e.groupReason||'', '');
      for(const item of e.relatedOccurrences)lines.push(`#### ${item.agent} · call ${item.call}`, '', 'Belief 注入原文：', '', '> '+item.graphEvidence.text.replace(/\n/g,'\n> '), '', 'Agent 输出：', '', '> '+item.quote.replace(/\n/g,'\n> '), '');
      lines.push('</details>', '');
    }
  }
  lines.push('## 归因总结', '', `主要作用：${a.conclusion.benefit}`, '', `主要问题：${a.conclusion.problem}`, '', `证据强度：${a.conclusion.evidence}`, '',
    ...(a.conclusion.focus?.length ? [`优先改进环节：${a.conclusion.focus.join('、')}`, ''] : []),
    `具体改进方案：${a.conclusion.improvement}`, '',
    ...(a.conclusion.verification?.trim() ? [`验证方式与预期变化：${a.conclusion.verification}`, ''] : []),
    `对照条件与其他影响因素：${a.conclusion.comparison}`, '', '## 来源文件', '');
  c.files.forEach(f => lines.push(`- ${f.path} · SHA-256 ${f.digest}`));
  return lines.join('\n');
}

export function csvSummary(cases) {
  // Neutralize spreadsheet formula injection from user-provided text.
  const cell = v => { let s = String(v ?? ''); if (/^\s*[=+@-]/.test(s)) s = "'" + s; return '"' + s.replace(/"/g, '""') + '"'; };
  const rows = [['task_id', 'reviewer', 'status', 'result', 'dimension', 'opportunity', 'utilization', 'effect', 'evidence', 'note', 'agent_tokens', 'graph_tokens', 'total_tokens', 'event_ids', 'dimension_description']];
  for (const c of cases) {
    const dimensions = selectedDimensions(c), t = tokenMetrics(c);
    if (!dimensions.length) {
      rows.push([c.taskId, c.annotation.reviewer, c.annotation.status, c.annotation.result, '', hasNoApplicableDimensions(c) ? '无' : '', '', '', '', '', t.agent, t.graph, t.combined, '', '']);
    }
    for (const d of dimensions) {
      const r = c.annotation.dimensions[d.id];
      rows.push([c.taskId, c.annotation.reviewer, c.annotation.status, c.annotation.result, `${d.id} ${d.title}`, r.opportunity, r.utilization, r.effect, dimensionEvidence(c, d.id), r.note, t.agent, t.graph, t.combined, linkedEvents(c, d.id).map(e => e.id).join(';'), d.custom ? d.question : '']);
    }
  }
  return '\uFEFF' + rows.map(r => r.map(cell).join(',')).join('\r\n');
}

export function workspaceExport(cases) {
  return {format: 'bcg-case-study-workspace', version: 1, exportedAt: new Date().toISOString(), cases};
}

export function restoreWorkspace(data) {
  if (data?.format !== 'bcg-case-study-workspace' || data.version !== 1 || !Array.isArray(data.cases)) throw new Error('不支持的标注工作包版本。');
  for (const c of data.cases) {
    if (!obj(c) || typeof c.id !== 'string' || !obj(c.annotation) || !['draft', 'reviewed'].includes(c.annotation.status)) throw new Error('工作包中的 Case 数据不完整。');
    if (!['taskId', 'title', 'question', 'report', 'mode', 'benchmark'].every(k => typeof c[k] === 'string') || !obj(c.metrics)) throw new Error('工作包中的任务信息无效。');
    if (!['files', 'sources', 'contexts', 'graphs', 'warnings'].every(k => Array.isArray(c[k]))) throw new Error('工作包中的来源数据不完整。');
    if (!Array.isArray(c.annotation.events) || !obj(c.annotation.conclusion) || !obj(c.annotation.dimensions)) throw new Error('工作包中的标注数据不完整。');
    if (c.annotation.events.some(e => !obj(e) || typeof e.id !== 'string' || !e.id)) throw new Error('工作包中的关键过程 ID 无效。');
    const eventIds = new Set(c.annotation.events.map(e => e.id));
    if (eventIds.size !== c.annotation.events.length) throw new Error('工作包中的关键过程 ID 重复。');
    const custom = c.annotation.customDimensions;
    if (custom !== undefined) {
      if (!Array.isArray(custom) || custom.length > 100 || custom.some(d => !obj(d) || typeof d.id !== 'string' || !/^C[1-9]\d{0,5}$/.test(d.id) || typeof d.title !== 'string' || !d.title.trim() || d.title.length > 60 || typeof d.question !== 'string' || d.question.length > 500)) throw new Error('工作包中的自定义维度无效。');
      const ids = new Set(custom.map(d => d.id));
      const names = allDimensions(c).map(d => dimensionName(d.title));
      if (ids.size !== custom.length || new Set(names).size !== names.length) throw new Error('工作包中的自定义维度 ID 或名称重复。');
    }
    const fields = ['opportunity', 'utilization', 'effect'];
    for (const d of allDimensions(c)) {
      const r = c.annotation.dimensions[d.id];
      if (!obj(r) || !['note', 'evidence', ...fields].every(k => typeof r[k] === 'string')) throw new Error('工作包中的维度标注不完整。');
      if (fields.some(k => r[k] && !OPTIONS[k].includes(r[k]))) throw new Error('工作包中包含未知的标注选项。');
      if (r.eventIds !== undefined && (!Array.isArray(r.eventIds) || r.eventIds.some(id => typeof id !== 'string' || !eventIds.has(id)) || new Set(r.eventIds).size !== r.eventIds.length)) {
        throw new Error('工作包中的维度与关键过程关联无效。');
      }
    }
    if (c.annotation.noApplicableDimensions !== undefined && (typeof c.annotation.noApplicableDimensions !== 'boolean' || (c.annotation.noApplicableDimensions && selectedDimensions(c).length))) throw new Error('工作包中的维度选择状态无效。');
    if (!['reviewer', 'criteria', 'result'].every(k => typeof c.annotation[k] === 'string')) throw new Error('工作包中的基本信息不完整。');
    if (!['benefit', 'problem', 'evidence', 'improvement', 'comparison'].every(k => typeof c.annotation.conclusion[k] === 'string')) throw new Error('工作包中的结论无效。');
    const conclusion = c.annotation.conclusion;
    if (conclusion.focus !== undefined && (!Array.isArray(conclusion.focus) || conclusion.focus.some(stage => !IMPROVEMENT_STAGES.includes(stage)) || new Set(conclusion.focus).size !== conclusion.focus.length)) throw new Error('工作包中的改进环节无效。');
    if (conclusion.verification !== undefined && typeof conclusion.verification !== 'string') throw new Error('工作包中的验证方式无效。');
    if (c.sources.some(s => !Array.isArray(s.records)) || c.graphs.some(g => !Array.isArray(g.nodes) || !Array.isArray(g.relations))) throw new Error('工作包中的轨迹或 Graph 无效。');
    if (c.files.some(f => !obj(f) || !['id', 'path', 'digest'].every(k => typeof f[k] === 'string'))) throw new Error('工作包中的文件来源无效。');
    if (c.sources.some(s => !['id', 'fileId', 'label', 'kind'].every(k => typeof s[k] === 'string') || s.records.some(r => !obj(r) || !['id', 'role', 'text', 'title'].every(k => typeof r[k] === 'string')))) throw new Error('工作包中的消息记录无效。');
    if (c.contexts.some(x => !obj(x) || !['id', 'fileId', 'label', 'kind', 'text'].every(k => typeof x[k] === 'string') || !Array.isArray(x.nodeIds))) throw new Error('工作包中的 Context 无效。');
    if (c.graphs.some(g => !['id', 'fileId', 'label', 'graphId'].every(k => typeof g[k] === 'string') || g.nodes.some(n => !obj(n)) || g.relations.some(r => !obj(r)))) throw new Error('工作包中的 Graph 记录无效。');
    if (c.annotation.events.some(e => !obj(e) || typeof e.id !== 'string' || !['behavior', 'impact', 'strength'].every(k => typeof e[k] === 'string'))) throw new Error('工作包中的关键过程无效。');
  }
  const cases = structuredClone(data.cases);
  cases.forEach(updateDemoTitle);
  return cases;
}

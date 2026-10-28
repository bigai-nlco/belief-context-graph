import {ingestFiles} from './model.mjs';
import {sha256Hex} from './browser_crypto.mjs';

export const REAL_DEMO_ID = 'europa-clipper-juice-20261020';

export function demoLabel(c) {
  return c.demoKind === 'real' ? '真实运行 Demo' : c.demo ? '虚构教学示例' : c.mode || '待分析';
}

export async function buildDemoCase(bundle) {
  if (bundle?.format !== 'bcg-real-case-demo' || bundle.version !== 1 ||
      bundle.taskId !== REAL_DEMO_ID || !Array.isArray(bundle.files) ||
      !bundle.files.length || bundle.files.length > 100 ||
      bundle.provenance?.status !== 'completed') {
    throw new Error('真实 Demo 文件格式不完整。');
  }
  const paths = new Set();
  for (const f of bundle.files) {
    if (typeof f.path !== 'string' || /(^\/|\\|(?:^|\/)\.\.(?:\/|$))/.test(f.path) ||
        paths.has(f.path) || typeof f.text !== 'string') {
      throw new Error('真实 Demo 的来源文件无效。');
    }
    paths.add(f.path);
    const bytes = new TextEncoder().encode(f.text);
    if (f.size !== bytes.byteLength || await sha256Hex(bytes) !== f.digest) {
      throw new Error(`真实 Demo 文件校验失败：${f.path}`);
    }
  }
  const c = ingestFiles(bundle.files);
  if (c.taskId !== REAL_DEMO_ID || !c.report || !c.graphs.length ||
      !c.contexts.some(ctx => ctx.text && ctx.nodeIds.length && !ctx.fallback)) {
    throw new Error('真实 Demo 缺少报告、Graph 或成功的注入记录。');
  }
  c.demo = true;
  c.demoKind = 'real';
  c.demoProvenance = bundle.provenance;
  return c;
}

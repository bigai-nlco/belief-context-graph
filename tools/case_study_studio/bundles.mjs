// Hash-checked, browser-local imports. Archive paths are labels, never fetched.
import {ingestFiles} from './model.mjs';
import {sha256Hex} from './browser_crypto.mjs';

export const BUNDLE_FORMAT = 'bcg-case-bundle';
export const MAX_IMPORT_BYTES = 250 * 1024 * 1024;
const encoder = new TextEncoder();
const validPath = path => typeof path === 'string' && path.length <= 1024 &&
  !/^(?:\/|[A-Za-z]:)|\\|[\u0000-\u001f]|(?:^|\/)\.\.?(?:\/|$)/.test(path) &&
  /\.(json|jsonl|ndjson|md|txt)$/i.test(path);

export async function readImportFile(file, remaining = MAX_IMPORT_BYTES, {hash=true} = {}) {
  const path = file.webkitRelativePath || file.name;
  if (file.size > MAX_IMPORT_BYTES) throw new Error('单次导入最多 250 MiB。');
  let stream = file.stream();
  if (/\.gz$/i.test(path)) {
    if (typeof DecompressionStream === 'undefined') throw new Error('当前浏览器不支持压缩包，请升级浏览器或先解压为 JSON。');
    stream = stream.pipeThrough(new DecompressionStream('gzip'));
  }
  const reader = stream.getReader(), decoder = new TextDecoder('utf-8', {fatal:true});
  const parts = []; let size = 0;
  try {
    for (;;) {
      const {done,value} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > remaining) throw new Error('解压后的文件合计超过 250 MiB，请分批导入。');
      parts.push(decoder.decode(value,{stream:true}));
    }
    parts.push(decoder.decode());
  } catch(error) {
    await reader.cancel().catch(()=>{});
    throw error;
  } finally {reader.releaseLock();}
  const text = parts.join('');
  return {path:path.replace(/\.gz$/i,''),text,size,...(hash?{digest:await sha256Hex(encoder.encode(text))}:{})};
}

export async function importBundles(bundles, existing = []) {
  const additions = [], seen = new Set(existing.map(c=>c.id));
  let bytes = 0, count = 0, skipped = 0, revisions = 0;
  const sourceIds = new Set(existing.map(c=>c.bundleSourceId).filter(Boolean));
  for (const bundle of bundles) {
    if (bundle?.format !== BUNDLE_FORMAT || bundle.version !== 1 ||
        !Array.isArray(bundle.cases) || !bundle.cases.length || bundle.cases.length > 1000) {
      throw new Error('不支持的 Case 运行包格式或版本。');
    }
    for (const entry of bundle.cases) {
      if (++count > 1000) throw new Error('一次最多导入 1000 个 Case，请分批导入。');
      if (!entry || typeof entry.id !== 'string' || !/^[a-f0-9]{64}$/.test(entry.id) ||
          !['task_id','benchmark','mode'].every(k=>typeof entry[k]==='string'&&entry[k].length<=2048) ||
          !Array.isArray(entry.files) || !entry.files.length || entry.files.length > 300) {
        throw new Error('Case 运行包缺少任务标识或来源文件。');
      }
      const paths = new Set();
      for (const file of entry.files) {
        if (!file || !validPath(file.path) || paths.has(file.path) || typeof file.text !== 'string') {
          throw new Error('Case 运行包含无效或重复的来源路径。');
        }
        paths.add(file.path);
        const raw = encoder.encode(file.text); bytes += raw.byteLength;
        if (raw.byteLength > 200*1024*1024 || bytes > MAX_IMPORT_BYTES) throw new Error('Case 来源文件超过导入大小限制。');
        if (file.size !== raw.byteLength || file.digest !== await sha256Hex(raw)) {
          throw new Error(`来源文件校验失败：${file.path}`);
        }
      }
      if (!paths.has('case.json')) throw new Error('Case 运行包缺少 case.json。');
      const metadata = JSON.parse(entry.files.find(f=>f.path==='case.json').text);
      if (!['task_id','benchmark','mode'].every(k=>metadata[k]===entry[k])) {
        throw new Error('Case 运行包的任务或模式标识不一致。');
      }
      const fingerprint = await sha256Hex(encoder.encode(JSON.stringify([
        entry.id, entry.files.map(f=>[f.path,f.digest]).sort((a,b)=>a[0].localeCompare(b[0])),
      ])));
      const id = 'bundle-' + fingerprint;
      // A new evidence revision is a separate case, never a rewrite of cited records.
      if (seen.has(id)) {skipped++;continue;}
      const c = ingestFiles(entry.files);
      if (c.taskId !== entry.task_id || c.mode !== entry.mode) throw new Error('来源文件包含其他任务或运行模式。');
      c.id=id; c.bundleSourceId=entry.id;
      if (sourceIds.has(entry.id)) {
        revisions++;
        c.warnings.push('同一次运行的新证据版本，原 Case 和标注保持不变。');
      }
      if (entry.warnings !== undefined && (!Array.isArray(entry.warnings) ||
          entry.warnings.length > 10000 || !entry.warnings.every(w=>typeof w==='string'&&w.length<4096))) {
        throw new Error('Case 运行包的记录提示格式无效。');
      }
      c.warnings.push(...(entry.warnings || []));
      // No annotations or automatic ratings are accepted from an evidence bundle.
      additions.push(c); seen.add(id); sourceIds.add(entry.id);
    }
  }
  return {cases:additions,skipped,revisions};
}

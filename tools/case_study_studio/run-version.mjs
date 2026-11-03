// Render only the version saved with this run, never the server's current HEAD.
const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const validSha=x=>typeof x==='string'&&/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(x);
const fingerprint=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
const pending=run=>run.managed&&['queued','starting'].includes(run.status);

export function versionLabel(run){
  const v=run.code_version;
  if(!v)return pending(run)?'启动时记录版本':'版本未记录';
  const revision=validSha(v.commit)?v.commit.slice(0,8):'无 Git 版本';
  return revision+(v.dirty===true?' · 未提交修改':v.dirty===false?'':' · 状态未知');
}

export function renderVersion(run){
  const v=run.code_version;
  if(!v)return `<p class="run-version-missing" data-live-key="version">${pending(run)?'代码版本将在测试启动时记录。':'代码版本：未记录'}</p>`;
  const sha=validSha(v.commit)?v.commit:null;
  const repo=typeof v.repository_url==='string'&&/^https:\/\/github\.com\/[\w.-]+\/[\w.-]+$/.test(v.repository_url)?v.repository_url:null;
  const item=(label,value)=>`<div><dt>${label}</dt><dd>${value}</dd></div>`;
  const hashRow=(label,value)=>item(label,fingerprint(value)?`<code>${value}</code>`:'未记录');
  const dirty=v.dirty===true?'未提交修改':v.dirty===false?'已提交 · 工作区干净':'工作区状态未知';
  const branch=v.branch||(sha?'Detached HEAD':'未记录');
  return `<details class="run-version" data-live-key="version" data-preserve-open>
    <summary aria-label="查看代码版本详情"><span class="version-summary-label">代码版本</span><code>${sha?sha.slice(0,8):'无 Git 版本'}</code><span class="version-branch">${escape(branch)}</span><span class="version-state${v.dirty===true?' dirty':''}">${dirty}</span></summary>
    <div class="version-body"><dl>
      ${item('Commit',sha?`<code>${sha}</code>${repo?` <a href="${repo}/commit/${sha}" target="_blank" rel="noopener noreferrer">GitHub ↗</a>`:''}`:'未记录（非 Git 工作区或 Git 不可用）')}
      ${item('分支',escape(branch))}
      ${item('记录时间',escape(v.captured_at||'未记录'))}
      ${hashRow('运行源码 / 配置指纹',v.source_sha256)}
      ${hashRow('Agent 源码指纹',v.agent_sha256)}
      ${run.mode==='default'?'':hashRow('Graph 源码指纹',v.graph_sha256)}
      ${hashRow('依赖锁文件 uv.lock',v.lock_sha256)}
    </dl><p class="version-note">记录启动时的本地代码，未校验是否已推送到 GitHub。指纹用于区分代码内容，不代替源码备份；运行中修改代码不会更新此记录。</p>
    ${v.fingerprint_complete===false?'<p class="version-warning">部分源码无法读取，指纹不完整。</p>':''}</div>
  </details>`;
}

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {versionLabel,renderVersion} from '../run-version.mjs';

const version={schema_version:1,commit:'a'.repeat(40),branch:'feat/example',dirty:false,repository_url:'https://github.com/example/project',source_sha256:'b'.repeat(64),agent_sha256:'c'.repeat(64),graph_sha256:'d'.repeat(64),lock_sha256:'e'.repeat(64),fingerprint_complete:true};

test('old runs stay unknown; only queued managed runs promise a later stamp',()=>{
  assert.equal(versionLabel({status:'finished'}),'版本未记录');
  assert.equal(versionLabel({managed:true,status:'queued'}),'启动时记录版本');
  assert.match(renderVersion({status:'finished'}),/代码版本：未记录/);
  assert.match(renderVersion({managed:true,status:'starting'}),/启动时记录/);
});

test('dirty worktrees, full commit, and independent component hashes are visible',()=>{
  assert.equal(versionLabel({code_version:version}),'aaaaaaaa');
  const html=renderVersion({mode:'bcg',code_version:{...version,dirty:true}});
  assert.match(html,/未提交修改/);
  for(const key of ['commit','source_sha256','agent_sha256','graph_sha256','lock_sha256'])assert(html.includes(version[key]));
  assert(html.includes(version.repository_url+'/commit/'+version.commit));
  assert(!renderVersion({mode:'default',code_version:version}).includes('Graph 源码指纹'));
});

test('HTML is escaped; invalid revisions or URLs cannot create links',()=>{
  const html=renderVersion({code_version:{...version,branch:'<img src=x onerror=alert(1)>',repository_url:'javascript:alert(1)'}});
  assert(!html.includes('<img'));
  assert(!html.includes('href='));
  assert(html.includes('&lt;img'));
  const unknown=renderVersion({code_version:{commit:'<script>',dirty:null,fingerprint_complete:false}});
  assert(!unknown.includes('<script>'));
  assert.match(unknown,/状态未知/);
  assert.match(unknown,/指纹不完整/);
  assert.match(renderVersion({code_version:{...version,branch:null}}),/Detached HEAD/);
});

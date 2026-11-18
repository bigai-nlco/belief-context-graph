import test from 'node:test';
import assert from 'node:assert/strict';
import {diffHTML, messageHTML, messagesHTML, turnLabel, duration} from '../clm.mjs';

test('context edits and submissions have distinct labels',()=>{
  assert.equal(turnLabel({edited:true,command:'cat report.md'}),'Context 编辑');
  assert.equal(turnLabel({submitted:true}),'提交报告');
  assert.equal(turnLabel({command:'web_search query'}),'搜索');
  assert.equal(duration(923.85),'15 分 24 秒');
});
test('untrusted raw diff is escaped, and changes are distinguished',()=>{
  const html=diffHTML('--- before\n+++ after\n@@ lines @@\n-old\n+<script>alert(1)</script>');
  assert(!html.includes('<script>'));
  assert(html.includes('clm-diff-add'));
  assert(html.includes('clm-diff-del'));
  assert(html.includes('clm-diff-heading'));
});
test('reasoning, assistant text and command are shown separately without rewriting',()=>{
  const html=messageHTML({role:'assistant',reasoning:'VERIFIED facts',content:'## Next step',tool_calls:[
    {function:{name:'bash',arguments:JSON.stringify({command:'echo "<script>"'})}}
  ]});
  assert(html.includes('VERIFIED facts'));
  assert(html.includes('Next step'));
  assert(html.includes('Tool call · bash'));
  assert(html.includes('echo &quot;&lt;script&gt;&quot;'));
  assert(!html.includes('<script>'));
});
test('messages retain roles and empty input is handled',()=>{
  const html=messagesHTML([{role:'system',content:'Pinned rules'}, {role:'user',content:'Pinned task'},
    {role:'assistant',reasoning_content:'Thinking'}, {role:'tool',content:'Observed'}]);
  for(const text of ['system','user','assistant','tool','Pinned rules','Pinned task','Thinking','Observed'])assert(html.includes(text));
  assert(messagesHTML(null).includes('没有 messages'));
});

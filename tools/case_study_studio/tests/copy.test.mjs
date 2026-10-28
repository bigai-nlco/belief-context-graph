import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, readFile, copyFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {copyKey, describeText, emptyDraft, emptyPublished, mergedCopy, publishDraft, renderCopy, validateCopy, validateEntry} from '../copy-model.mjs';

const entry={kind:'text',original:'标注教程',value:'新手指南'};
test('draft changes only presentation and published version advances on apply',()=>{
  const original=emptyPublished(), draft={...emptyDraft(0),entries:[entry]};
  assert.equal(mergedCopy(original,draft).get(copyKey(entry)).value,'新手指南');
  assert.deepEqual(original,emptyPublished());
  const next=publishDraft(original,draft);
  assert.equal(next.revision,1);assert.deepEqual(next.entries,[entry]);
  const revert=publishDraft(next,{...emptyDraft(1),entries:[{...entry,value:entry.original}]});
  assert.deepEqual(revert.entries,[]);
});
test('dynamic counters remain exact when their surrounding copy changes',()=>{
  const d=describeText('  1,234 / 2,000 条记录  ',true);
  assert.equal(d.original,'{{n1}} / {{n2}} 条记录');
  const e={kind:'text',original:d.original,value:'共 {{n2}} 条，当前 {{n1}} 条'};
  validateEntry(e);
  assert.equal(renderCopy(e,d),'  共 2,000 条，当前 1,234 条  ');
  assert.throws(()=>validateEntry({...e,value:'100 条'}),/占位符/);
  assert.throws(()=>validateEntry({...e,value:'{{n1}} {{n1}}'}),/占位符/);
  assert.equal(describeText('call 4 · B10',false).original,'call 4 · B10');
});
test('rejects duplicate entries, unsupported kinds, oversized files and stale revisions',()=>{
  assert.throws(()=>validateCopy({...emptyDraft(0),entries:[entry,entry]},true),/重复/);
  assert.throws(()=>validateEntry({...entry,kind:'innerHTML'}),/无效/);
  assert.throws(()=>validateEntry({...entry,value:'a'.repeat(12001)}),/限制/);
  assert.throws(()=>publishDraft(emptyPublished(),emptyDraft(1)),/正式文案已更新/);
  assert.throws(()=>validateCopy({format:'bad',version:1,revision:0,entries:[]}),/格式/);
});
test('HTML-looking copy remains plain text and whitespace or empty replacements are supported',()=>{
  const d=describeText(' 标题 ',false);
  assert.equal(renderCopy({value:'<img src=x onerror=alert(1)>'},d),' <img src=x onerror=alert(1)> ');
  assert.equal(renderCopy({value:''},d),'  ');
  assert.equal(copyKey({kind:'text',original:'Name'}),copyKey({kind:'text',original:'Name',value:'X'}));
  assert.notEqual(copyKey({kind:'text',original:'Name'}),copyKey({kind:'placeholder',original:'Name'}));
});
test('apply command checks without writing, explicitly publishes, and rejects stale reapplication',async()=>{
  const root=await mkdtemp(join(tmpdir(),'bcg-copy-apply-'));
  try{
    for(const f of ['apply-copy.mjs','copy-model.mjs'])await copyFile(new URL('../'+f,import.meta.url),join(root,f));
    const target=join(root,'copy-overrides.json'), draft=join(root,'draft.json');
    await writeFile(target,JSON.stringify(emptyPublished()));
    await writeFile(draft,JSON.stringify({...emptyDraft(0),entries:[entry]}));
    const run=(...args)=>spawnSync(process.execPath,[join(root,'apply-copy.mjs'),draft,...args],{encoding:'utf8'});
    assert.equal(run().status,0);assert.equal(JSON.parse(await readFile(target)).revision,0);
    assert.equal(run('--write').status,0);
    assert.deepEqual(JSON.parse(await readFile(target)),publishDraft(emptyPublished(),{...emptyDraft(0),entries:[entry]}));
    assert.notEqual(run('--write').status,0);assert.equal(JSON.parse(await readFile(target)).revision,1);
  }finally{await rm(root,{recursive:true,force:true});}
});

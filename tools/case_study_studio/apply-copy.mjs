#!/usr/bin/env node
// Review by default. --write applies only a validated, revision-matched draft.
import {readFile, writeFile, rename, unlink} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {publishDraft, validateCopy} from './copy-model.mjs';

const args=process.argv.slice(2), file=args.find(x=>!x.startsWith('--'));
if(!file||args.some(x=>x.startsWith('--')&&x!=='--write')){
  console.error('Usage: node tools/case_study_studio/apply-copy.mjs <bcg-ui-copy.json> [--write]');
  process.exit(1);
}
const target=fileURLToPath(new URL('./copy-overrides.json',import.meta.url));
if(resolve(file)===target)throw new Error('修改稿和正式文案文件不能是同一个文件。');
const published=validateCopy(JSON.parse(await readFile(target,'utf8')));
const draft=validateCopy(JSON.parse(await readFile(file,'utf8')),true);
const next=publishDraft(published,draft);
console.log(`正式版本 ${published.revision} → ${next.revision}，修改 ${draft.entries.length} 条文案。`);
for(const e of draft.entries)console.log(JSON.stringify({kind:e.kind,original:e.original,value:e.value}));
if(args.includes('--write')){
  const temporary=target+`.${process.pid}.tmp`;
  try{
    await writeFile(temporary,JSON.stringify(next,null,2)+'\n',{flag:'wx'});
    await rename(temporary,target);
  }catch(error){await unlink(temporary).catch(()=>{});throw error;}
  console.log('已应用正式文案，刷新页面即可看到。');
}else console.log('仅检查，尚未修改正式页面。确认后添加 --write。');

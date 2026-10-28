import {readImportFile,importBundles} from './bundles.mjs';

export async function parseEvidence(blob) {
  // The wrapper hash is unused for shared evidence. Every retained source file
  // is still independently verified by importBundles before it is ingested.
  const {text}=await readImportFile(new File([blob],'case-study.json.gz'),undefined,{hash:false});
  const bundle=JSON.parse(text),{cases}=await importBundles([bundle]);
  if(cases.length!==1)throw new Error('该任务的证据包应只包含一个 Case');
  return {c:cases[0],projection:bundle.projection||null};
}

export function loadEvidence(blob) {
  if(typeof Worker==='undefined')return parseEvidence(blob);
  return new Promise((resolve,reject)=>{
    let worker,settled=false;
    try{worker=new Worker(new URL('./evidence-worker.mjs',import.meta.url),{type:'module'});}
    catch{resolve(parseEvidence(blob));return;}
    const finish=(error,value)=>{if(settled)return;settled=true;clearTimeout(timeout);worker.terminate();error?reject(error):resolve(value);};
    const timeout=setTimeout(()=>finish(new Error('证据处理超时，请重试。')),180000);
    worker.onmessage=({data})=>finish(data.error?new Error(data.error):null,data.value);
    worker.onerror=()=>finish(new Error('证据处理进程异常，请刷新后重试。'));
    try{worker.postMessage(blob);}catch(error){finish(error);}
  });
}

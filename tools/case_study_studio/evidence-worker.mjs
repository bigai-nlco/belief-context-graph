import {parseEvidence} from './evidence-loader.mjs';

self.onmessage=async({data})=>{
  try{self.postMessage({value:await parseEvidence(data)});}
  catch(error){self.postMessage({error:error.message||'证据解析失败'});}
};

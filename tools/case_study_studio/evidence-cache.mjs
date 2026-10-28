// Only parsed, immutable preview evidence is cached here. No identity, AI review
// or human annotations. Full Graph bundles deliberately do not enter this store.
import {newAnnotation} from './model.mjs';
const VERSION=1,MAX_ENTRIES=8,MAX_SOURCE_BYTES=4*1024*1024;
let connection;

export const previewCacheKey=(remote,sha)=>`v${VERSION}:${remote.run}:${remote.benchmark}:${remote.task}:${sha}`;

function database(){
  if(!globalThis.indexedDB)return Promise.resolve(null);
  if(!connection)connection=new Promise(resolve=>{
    const request=indexedDB.open('bcg-studio-evidence-cache',VERSION);
    request.onupgradeneeded=()=>{request.result.createObjectStore('previews',{keyPath:'key'}).createIndex('used','used');};
    request.onsuccess=()=>{const db=request.result;db.onversionchange=()=>{db.close();connection=null;};resolve(db);};
    request.onerror=()=>resolve(null);
    request.onblocked=()=>resolve(null);
  });
  return connection;
}

export async function readPreviewCache(key){
  try{
    const db=await database();if(!db)return null;
    return await new Promise(resolve=>{
      const tx=db.transaction('previews','readwrite'),store=tx.objectStore('previews'),request=store.get(key);
      let value=null;
      request.onsuccess=()=>{const row=request.result;if(row?.value?.c?.files&&row.value.projection?.kind==='review-preview'){
        value=row.value;store.put({...row,used:Date.now()});
      }};
      tx.oncomplete=()=>resolve(value);tx.onerror=tx.onabort=()=>resolve(null);
    });
  }catch{return null;}
}

export async function writePreviewCache(key,value){
  // Snapshot before awaiting IndexedDB: callers subsequently attach personal or
  // AI annotation state to their copy, which must never leak into this cache.
  if(value.c.files.reduce((n,f)=>n+f.size,0)>MAX_SOURCE_BYTES)return;
  const clean=structuredClone(value);delete clean.c.remote;clean.c.annotation=newAnnotation();
  try{
    const db=await database();if(!db)return;
    await new Promise(resolve=>{
      const tx=db.transaction('previews','readwrite'),store=tx.objectStore('previews');
      store.put({key,value:clean,used:Date.now()});
      const keys=store.index('used').getAllKeys();
      keys.onsuccess=()=>{for(const old of keys.result.slice(0,Math.max(0,keys.result.length-MAX_ENTRIES)))store.delete(old);};
      tx.oncomplete=tx.onerror=tx.onabort=()=>resolve();
    });
  }catch{/* Quota or private-browsing restrictions must not block reading. */}
}

export async function deletePreviewCache(key){
  try{const db=await database();if(db)db.transaction('previews','readwrite').objectStore('previews').delete(key);}catch{}
}

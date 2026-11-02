import assert from 'node:assert/strict';
import {test} from 'node:test';
import {scoreDetailsMarkup,scoreItemsMarkup} from '../score-details.mjs';

const data={status:'scored',score:1/3,model:'offline-judge',reasoning_effort:'high',dimensions:[
  {id:'info_recall',label:'信息召回',score:0.5,items:[
    {criterion:'A',score:1,reason:'Reason A',evidence:'Quote A\nLine 2'},
    {criterion:'B',score:0,reason:'Reason B',evidence:''},
  ]},
  {id:'analysis',label:'分析',score:0,items:[{criterion:'C',score:-1,reason:'Blocked source',evidence:'Quote C'}]},
  {id:'presentation',label:'表达呈现',score:null,items:[]},
]};

test('saved rubric explanations retain raw text and original item numbers',()=>{
  const all=scoreItemsMarkup(data);
  assert.equal(all.count,3);
  for(const text of ['Reason A','Quote A\nLine 2','报告原文证据','0 分项按评分规则不提供原文证据','命中受限引用'])assert(all.html.includes(text));
  const filtered=scoreItemsMarkup(data,'info_recall','0');
  assert.equal(filtered.count,1);
  assert(filtered.html.includes('>2</span>'));
  assert(!filtered.html.includes('Reason A'));
  assert.equal(scoreItemsMarkup(data,'analysis','1').count,0);
  assert(scoreItemsMarkup(data,'analysis','1').html.includes('没有符合'));
});

test('score overview uses stored aggregates, never counts blocked citations as minus points',()=>{
  const html=scoreDetailsMarkup(data);
  assert(html.includes('33.3 / 100'));
  assert(html.includes('0.0 / 100'));
  assert(html.includes('得 1 分的项目占全部评分项的比例'));
  assert(html.includes('全部维度（3）'));
  assert(html.includes('从 Agent 最终报告中摘录'));
  assert(html.includes('按每条评分项记录'));
});

test('untrusted rubric text, reasoning, evidence and model names are escaped',()=>{
  const unsafe='<img src=x onerror=alert(1)>';
  const html=scoreDetailsMarkup({...data,model:unsafe,dimensions:[{id:'" onclick="attack()',label:unsafe,score:0,items:[
    {criterion:unsafe,score:0,reason:unsafe,evidence:unsafe},
  ]}]});
  assert(!html.includes('<img'));
  assert(!html.includes('value="" onclick='));
  assert(html.includes('&lt;img'));
});

test('a failed judge has an explicit incomplete notice and no fabricated numeric score',()=>{
  const html=scoreDetailsMarkup({...data,status:'judge_error',score:null,dimensions:[]});
  assert(html.includes('本次评分未完成'));
  assert(!html.includes('0.0 / 100'));
});

test('rule-based empty evidence is distinct from missing required evidence',()=>{
  const render=(score,evidence)=>scoreItemsMarkup({...data,dimensions:[
    {id:'analysis',label:'分析',score:0,items:[{criterion:'A',score,reason:'Original reason',evidence}]},
  ]}).html;
  for(const empty of ['',null,undefined,'  \n  ']){
    const zero=render(0,empty);
    assert(zero.includes('data-evidence-status="not-required"'));
    assert(!zero.includes('证据缺失'));
    for(const score of [1,-1]){
      const missing=render(score,empty);
      assert(missing.includes('data-evidence-status="missing"'));
      assert(missing.includes('证据缺失'));
      assert(!missing.includes('0 分项按评分规则'));
      assert(missing.includes('Original reason'));
    }
    const unknown=render(null,empty);
    assert(unknown.includes('data-evidence-status="unrecorded"'));
    assert(!unknown.includes('按评分规则'));
  }
  for(const score of [0,1,-1]){
    const provided=render(score,'  Exact quote\nSecond line  ');
    assert(provided.includes('data-evidence-status="provided"'));
    assert(provided.includes('  Exact quote\nSecond line  '));
    assert(!provided.includes('data-evidence-status="not-required"'));
    assert(!provided.includes('证据缺失'));
  }
});

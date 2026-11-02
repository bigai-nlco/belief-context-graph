import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseTaskIds,mergeTaskIds,filterTasks,taskIdsInRect} from '../task-picker.mjs';

test('task input accepts paste formats, ranges and stable deduplication',()=>{
  assert.deepEqual(parseTaskIds('2，5  30\n5; 2'),['2','5','30']);
  assert.deepEqual(parseTaskIds('drb2:2,5,20-23'),['2','5','20','21','22','23']);
  assert.deepEqual(parseTaskIds('--task-ids "drb2:2,5,30"'),['2','5','30']);
  assert.deepEqual(parseTaskIds("--task-ids='drb2:2,5'"),['2','5']);
  assert.deepEqual(parseTaskIds('2 - 5, 4–6, 002'),['2','3','4','5','6']);
  assert.deepEqual(parseTaskIds(''),[]);
  assert.deepEqual(parseTaskIds('1-32'),Array.from({length:32},(_,i)=>String(i+1)));
});

test('malformed and excessive inputs fail atomically',()=>{
  for(const spec of ['3-1','abc','0','-1','2.5','1-999999999','9007199254740993', '2,<script>','drbench:2'])
    assert.throws(()=>parseTaskIds(spec));
  assert.throws(()=>parseTaskIds('1-33'),/最多选择 32/);
  assert.throws(()=>parseTaskIds('1-32,100'),/最多选择 32/);
  assert.deepEqual(parseTaskIds('1-32,2,3').length,32);
});

test('selection preserves existing order, enforces capacity and validates real IDs',()=>{
  const previous=['5','2'];
  assert.deepEqual(mergeTaskIds(previous,['2','30'],3,new Set(['2','5','30'])),['5','2','30']);
  assert.throws(()=>mergeTaskIds(previous,['999'],32,new Set(['2','5'])),/999/);
  assert.throws(()=>mergeTaskIds(previous,['3'],2),/最多选择 2/);
  assert.deepEqual(previous,['5','2']);
  assert.deepEqual(mergeTaskIds([],['999'],32),['999']); // Missing catalog: runner validates later.
});

test('task catalog search matches IDs, questions, themes and languages',()=>{
  const tasks=[{id:'2',preview:'Climate change projections',theme:'environment',language:'en'},
    {id:'30',preview:'教育政策与体育活动',theme:'education',language:'zh'}];
  assert.deepEqual(filterTasks(tasks,'30'),[tasks[1]]);
  assert.deepEqual(filterTasks([...tasks,{id:'130',preview:'30 candidates'}],'30'),[tasks[1]]);
  assert.deepEqual(filterTasks(tasks,'0030'),[tasks[1]]);
  assert.deepEqual(filterTasks(tasks,'climate EN'),[tasks[0]]);
  assert.deepEqual(filterTasks(tasks,'体育'),[tasks[1]]);
  assert.deepEqual(filterTasks(tasks,'missing'),[]);
  assert.deepEqual(filterTasks(tasks,'  '),tasks);
});

test('marquee picks intersecting chips across rows, in either drag direction',()=>{
  const items=[{id:'2',left:10,right:60,top:10,bottom:40},{id:'5',left:70,right:120,top:10,bottom:40},
    {id:'30',left:10,right:60,top:50,bottom:80},{id:'49',left:70,right:120,top:50,bottom:80}];
  assert.deepEqual(taskIdsInRect(items,{x:5,y:5},{x:65,y:85}),['2','30']);
  assert.deepEqual(taskIdsInRect(items,{x:65,y:85},{x:5,y:5}),['2','30']);
  assert.deepEqual(taskIdsInRect(items,{x:55,y:35},{x:75,y:55}),['2','5','30','49']);
  assert.deepEqual(taskIdsInRect(items,{x:60,y:0},{x:70,y:90}),[],'touching a border is not overlap');
  assert.deepEqual(taskIdsInRect(items,{x:200,y:200},{x:220,y:220}),[]);
});

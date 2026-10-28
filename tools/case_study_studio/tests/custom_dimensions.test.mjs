import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DIMENSIONS, allDimensions, addCustomDimension, updateCustomDimension, removeCustomDimension,
  ingestFiles, newEvent, setDimensionEvent, linkedEvents, setDimensionSelected,
  selectedDimensions, completion, validateReview, hasNoApplicableDimensions,
  confirmNoApplicableDimensions, conclusionObservations, removeEvent,
  workspaceExport, restoreWorkspace, markdownReport, csvSummary,
} from '../model.mjs';

const createCase = () => ingestFiles([{path:'task.json',digest:'test',text:JSON.stringify({task_id:'custom-test',question:'Review the evidence.'})}]);

test('plus adds multiple case-local dimensions without modifying the eight presets', () => {
  const c = createCase(), other = createCase();
  confirmNoApplicableDimensions(c, true);
  const first = addCustomDimension(c, '  适用范围  ', ' 区分结论的适用条件。 ');
  const second = addCustomDimension(c, '保留限定条件');
  assert.deepEqual(first, {id:'C1',title:'适用范围',question:'区分结论的适用条件。'});
  assert.equal(second.id, 'C2');
  assert.equal(DIMENSIONS.length, 8);
  assert.equal(allDimensions(c).length, 10);
  assert.equal(allDimensions(other).length, 8);
  assert.deepEqual(other.annotation.customDimensions, []);
  assert.deepEqual(selectedDimensions(c).map(d => d.id), ['C1','C2']);
  assert.equal(hasNoApplicableDimensions(c), false);
  assert.equal(c.annotation.dimensions.C1.utilization, '');
  assert.deepEqual(completion(c), {completed:0,total:2,percent:0});
});

test('custom dimensions use the same evidence, review, rename and deselection rules', () => {
  const c = createCase(), d = addCustomDimension(c, '适用范围', '辨认信息范围');
  const event = newEvent({reference:'trace:2',behavior:'Checked evidence',impact:'Preserved scope',strength:'明确可见'});
  c.annotation.events.push(event);
  Object.assign(c.annotation, {reviewer:'Reviewer',result:'尚未评判'});
  Object.assign(c.annotation.conclusion, {benefit:'Retains conditions',evidence:'明确可见'});
  Object.assign(c.annotation.dimensions[d.id], {utilization:'部分利用',effect:'正向',note:'Used the stated scope.'});
  assert(validateReview(c).includes('C1 关联的关键过程'));
  assert.equal(setDimensionEvent(c,d.id,event.id,true), true);
  assert.deepEqual(validateReview(c), []);
  const rating = structuredClone(c.annotation.dimensions.C1);
  updateCustomDimension(c,d.id,'限定条件','检查限定条件');
  assert.deepEqual(c.annotation.dimensions.C1, rating);
  assert.equal(conclusionObservations(c)[0].title,'限定条件');
  assert.deepEqual(conclusionObservations(c)[0].events,[event]);
  setDimensionSelected(c,d.id,false);
  assert.equal(selectedDimensions(c).length,0);
  assert.doesNotMatch(markdownReport(c),/检查限定条件/);
  setDimensionSelected(c,d.id,true);
  assert.deepEqual(c.annotation.dimensions.C1,rating);
  assert.deepEqual(completion(c),{completed:1,total:1,percent:100});
  removeEvent(c,event.id);
  assert.deepEqual(linkedEvents(c,d.id),[]);
  assert.equal(completion(c).completed,0);
});

test('removing a custom dimension preserves source processes and other reviews', () => {
  const c=createCase(), first=addCustomDimension(c,'First'), second=addCustomDimension(c,'Second');
  const event=newEvent({dimensions:[first.id,'A1',second.id]});c.annotation.events.push(event);
  setDimensionEvent(c,first.id,event.id,true);setDimensionEvent(c,second.id,event.id,true);
  c.annotation.dimensions.A1.note='Existing review';
  assert.equal(removeCustomDimension(c,'A1'),false);
  assert.equal(removeCustomDimension(c,'missing'),false);
  assert.equal(removeCustomDimension(c,first.id),true);
  assert.equal(Object.hasOwn(c.annotation.dimensions,first.id),false);
  assert.deepEqual(event.dimensions,['A1',second.id]);
  assert.deepEqual(linkedEvents(c,second.id),[event]);
  assert.equal(c.annotation.dimensions.A1.note,'Existing review');
  assert.equal(c.annotation.events.length,1);
});

test('invalid names and duplicate names do not mutate annotations', () => {
  const c=createCase();addCustomDimension(c,'Scope');
  for(const title of ['', '  ', null, 'x'.repeat(61), 'scope', '长程信息复用']) {
    const before=structuredClone(c);
    assert.throws(()=>addCustomDimension(c,title));
    assert.deepEqual(c,before);
  }
  for(const description of [null, 'x'.repeat(501)]) assert.throws(()=>addCustomDimension(c,'Other',description));
  assert.throws(()=>updateCustomDimension(c,'A1','Changed'),/未找到/);
  assert.throws(()=>updateCustomDimension(c,'C1','长程信息复用'),/同名/);
  updateCustomDimension(c,'C1','scope');
  assert.equal(c.annotation.customDimensions[0].title,'scope');
});

test('custom definitions, ratings and source links survive JSON, Markdown and CSV exports', () => {
  const c=createCase(), d=addCustomDimension(c,'Scope | boundary','Specific conditions');
  const e=newEvent({label:'Source review',reference:'trace:2'});c.annotation.events.push(e);
  setDimensionEvent(c,d.id,e.id,true);
  c.annotation.dimensions.C1.note='Keep this assessment';
  const restored=restoreWorkspace(JSON.parse(JSON.stringify(workspaceExport([c]))))[0];
  assert.deepEqual(restored,c);
  assert.match(markdownReport(restored),/C1 Scope \\\| boundary/);
  assert.match(markdownReport(restored),/C1 维度说明：Specific conditions/);
  assert.match(markdownReport(restored),/\[E01 · Source review\]\(#process-1\)/);
  const csv=csvSummary([restored]);
  assert.match(csv,/dimension_description/);
  assert.match(csv,/C1 Scope \| boundary/);
  assert.match(csv,/Specific conditions/);
  assert(csv.includes(e.id));
  setDimensionSelected(restored,d.id,false);
  assert.doesNotMatch(csvSummary([restored]),/Specific conditions/);
  assert.deepEqual(restoreWorkspace(workspaceExport([restored]))[0],restored);
});

test('old workspaces without custom dimensions remain unchanged', () => {
  const c=createCase();delete c.annotation.customDimensions;
  assert.deepEqual(restoreWorkspace(workspaceExport([c]))[0],c);
  assert.equal(allDimensions(c).length,8);
  assert.equal(addCustomDimension(c,'New dimension').id,'C1');
});

test('restore rejects unsafe, malformed or duplicate custom definitions and bad evidence links', () => {
  const c=createCase();addCustomDimension(c,'Scope');
  const mutations=[
    a=>{a.customDimensions=null;},
    a=>{a.customDimensions='bad';},
    a=>{a.customDimensions[0].id='__proto__';},
    a=>{a.customDimensions[0].id='C1.bad';},
    a=>{a.customDimensions[0].id='A1';},
    a=>{a.customDimensions.push({...a.customDimensions[0]});},
    a=>{a.customDimensions[0].title='长程信息复用';},
    a=>{a.customDimensions[0].title=' ';},
    a=>{a.customDimensions[0].question={text:'bad'};},
    a=>{a.customDimensions[0].title='x'.repeat(61);},
    a=>{delete a.dimensions.C1;},
    a=>{a.dimensions.C1.eventIds=['foreign-case-event'];},
    a=>{a.dimensions.C1.effect='unexpected';},
  ];
  for(const mutate of mutations){
    const invalid=structuredClone(c);mutate(invalid.annotation);
    assert.throws(()=>restoreWorkspace(workspaceExport([invalid])));
  }
});

test('custom dimension limit is enforced without changing an existing case', () => {
  const c=createCase();
  for(let i=0;i<100;i++)addCustomDimension(c,'Dimension '+i);
  const before=structuredClone(c);
  assert.throws(()=>addCustomDimension(c,'Overflow'),/最多/);
  assert.deepEqual(c,before);
  assert.deepEqual(restoreWorkspace(workspaceExport([c]))[0],c);
});

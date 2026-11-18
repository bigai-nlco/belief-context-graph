import assert from 'node:assert/strict';
import {test} from 'node:test';
import {remoteLocation,recordsLink,canViewRecords,canAnnotateTask,saveSharedCase} from '../shared.mjs';

test('record routes are distinct from personal and AI annotations',async()=>{
  const link=recordsLink(26,'drb2','2');
  assert.equal(link,'./index.html?run=26&benchmark=drb2&task=2&view=records');
  assert.deepEqual(remoteLocation(link.slice(link.indexOf('?'))),{run:26,benchmark:'drb2',task:'2',view:'records'});
  assert.throws(()=>remoteLocation('?run=26&benchmark=drb2&task=2&view=other'),/无效/);
  assert.throws(()=>remoteLocation('?run=26&benchmark=drb2&task=2&view=records&review=ai'),/无效/);
  const task={status:'interrupted',viewable:true,annotatable:false};
  assert(canViewRecords(task));
  assert(!canAnnotateTask(task));
  assert(!canViewRecords({status:'interrupted'}));
  await assert.rejects(saveSharedCase({remote:{view:'records',readOnly:true}}),/只读/);
});

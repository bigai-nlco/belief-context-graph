import test from 'node:test';
import assert from 'node:assert/strict';
import {graphViews, ingestFiles, overviewGraph, trajectoryLayout, timelineSize, taskSetIndex, taskRootId, taskDiagramLayout, workflowDetails} from '../model.mjs';
import {normalizeView} from '../view-state.mjs';

const file=(path,body)=>({path,text:typeof body==='string'?body:JSON.stringify(body),digest:path});
const belief=id=>({id,belief:`source ${id}`});
const edge=(from_id,to_id,type)=>({from_id,to_id,type});
const task=(id)=>({id:`T${id}`,belief_id:id,status:'in_progress',text:`task ${id}`});
test('root inspector shows only source-backed work, never infers it for old runs',()=>{
  const root={plan_ids:[134],open_gap_ids:[41],status_source_id:104};
  const graph={nodes:[{id:41,belief:'BRAC',gap_quote:'failures are still missing'},
    {id:134,belief:'Investigation',next_action_quote:'look for self-exclusion mechanisms'},
    {id:104,belief:'Agent reports reviewing four sources'}],relations:[],taskState:{version:4,nodes:[],relations:[],root}};
  assert.deepEqual(workflowDetails(graph,root).map(r=>[r.label,r.id]),
    [['待核实','B41'],['明确的下一步','B134'],['最近报告的进展','B104']]);
  assert.deepEqual(workflowDetails({...graph,taskState:{...graph.taskState,version:3}},root).map(r=>[r.label,r.id]),
    [['待核实','B41'],['明确的下一步','B134'],['最近报告的进展','B104']]);
  const withUnassigned={...graph,nodes:[...graph.nodes,{id:55,belief:'AAY',gap_quote:'need more on failures',
    supporting_excerpts:['- AAY (Antyodaya Anna Yojana): Information on food grain entitlements (need more on failures)']}],
    taskState:{...graph.taskState,unassigned_gap_ids:[55]}};
  const details=workflowDetails(withUnassigned,root);
  assert.deepEqual(details.map(r=>[r.label,r.id]),
    [['待核实','B41'],['待归属缺口','B55'],['明确的下一步','B134'],['最近报告的进展','B104']]);
  assert.match(details[1].text,/AAY.*need more on failures/);
  assert.deepEqual(workflowDetails({...graph,taskState:{version:2}},root),[]);
  assert.equal(overviewGraph(graph,'tsg').taskCount,0);
});
const snapshot={
  nodes:[0,2,4,114,116,999].map(belief),
  relations:[edge(114,116,'supplements'),edge(2,4,'depends_on')],
  task_state:{version:2,nodes:[task(2),task(4),task(7)],relations:[
    edge('T2','B2','described_by'),edge('B114','T2','addresses'),
    edge('B114','T4','addresses'),edge('T4','T2','part_of'),
    {...edge('T2','B116','related_via_graph'),relation_ids:[13,14]},
  ]},
};

test('whole BCG includes disconnected components and isolates; TSG references, not factual edges',()=>{
  const c=ingestFiles([file('agent/bcg/state.json',{snapshot})]), g=graphViews(c).full[0];
  const b=overviewGraph(g), t=overviewGraph(g,'tsg');
  assert.equal(overviewGraph(g,'bcg'),b);
  assert.equal(b.beliefCount,6);assert.equal(b.relations.length,2);assert.equal(b.incident.get('B0').length,0);
  assert.equal(t.taskCount,3);assert.equal(t.beliefCount,3);assert.equal(t.relations.length,5);
  assert.equal(t.incident.get('T7').length,0);
  assert.deepEqual(t.incident.get('B114').map(e=>e.to_id),['T2','T4']);
  assert.equal(t.byId.get('B114').raw,b.byId.get('B114').raw);
  assert(!t.byId.has('B999'));assert(!t.relations.some(e=>e.from_id.startsWith('B')&&e.to_id.startsWith('B')));
  assert.deepEqual(t.relations.at(-1).raw.relation_ids,[13,14]);
  assert.equal(taskSetIndex(t).groups.size,3);
});

test('task sets keep exact membership, duplicate records, real hierarchy, role provenance, and shared identity',()=>{
  const c=structuredClone(snapshot);
  c.nodes.find(n=>n.id===114).task_role='progress';
  Object.assign(c.task_state.nodes[0],{evidence_ids:[116,999],related_action_ids:[999]});
  c.task_state.relations.push(edge('T2','B114','updated_by'),edge('T2','B114','updated_by'),edge('T2','B91','related_via_graph'),edge('B2','B4','legacy_record'));
  const t=overviewGraph({nodes:c.nodes,relations:c.relations,taskState:c.task_state},'tsg');
  const s=taskSetIndex(t),a=s.groups.get('T2'),b=s.groups.get('T4');
  assert.equal(taskSetIndex(t),s);
  assert.deepEqual([...s.groups.keys()],['T2','T4','T7']);
  assert.deepEqual([...s.owners.get('B114')],['T2','T4']);assert.equal(s.sharedCount,1);
  assert.equal(a.shared[0].node,b.shared[0].node);assert.equal(a.shared[0].edges.length,3);
  assert.deepEqual(a.rows.map(m=>[m.id,m.bucket]),[['B2','anchor'],['B91','other'],['B114','progress'],['B116','evidence']]);
  assert.equal(s.owners.has('B999'),false);assert.equal(a.members.get('B91').node.kind,'missing');
  assert.deepEqual([...a.children],['T4']);assert.deepEqual([...b.parents],['T2']);
  assert.deepEqual([...a.parents],[]);assert.deepEqual(s.groups.get('T7').rows,[]);
  assert.deepEqual(s.unplaced.map(e=>e.type),['legacy_record']);
  assert.equal(a.members.get('B116').edges[0].raw.relation_ids[1],14);
  const noHierarchy=taskSetIndex(overviewGraph({nodes:c.nodes,relations:[],taskState:{...c.task_state,relations:c.task_state.relations.filter(e=>e.type!=='part_of')}},'tsg'));
  assert.equal(noHierarchy.groups.get('T4').parents.size,0);
  assert.equal(noHierarchy.sharedCount,1);
});

test('shared sections are disjoint exact owner sets, not intersections counted twice',()=>{
  const taskState={nodes:[task(1),task(2),task(3)],relations:[
    edge('B4','T2','addresses'),edge('B4','T1','addresses'),edge('T1','B4','updated_by'),
    edge('B7','T2','addresses'),edge('B7','T3','addresses'),edge('B7','T1','addresses'),edge('B8','T1','addresses')]};
  const s=taskSetIndex(overviewGraph({nodes:[4,7,8].map(belief),relations:[],taskState},'tsg'));
  assert.deepEqual(s.overlaps,[{tasks:['T1','T2'],ids:['B4']},{tasks:['T1','T2','T3'],ids:['B7']}]);
  assert.equal(s.sharedCount,2);assert.equal(s.owners.size,3);
});

test('one user-input root, single parent per task, deepest recorded parent then lowest id, no membership branches',()=>{
  const state={nodes:[9,4,7,1,6].map(task),relations:[
    edge('T1','T9','part_of'),edge('T1','T4','part_of'),edge('T6','T1','part_of'),
    edge('T6','T9','part_of'),edge('B88','T1','addresses'),edge('B88','T4','addresses'),
    edge('T7','T99','part_of'), // Missing task record: preserve in unplaced, not as a made-up task.
  ]};
  const index=taskSetIndex(overviewGraph({nodes:[belief(88)],relations:[],taskState:state},'tsg'));
  const sizes=new Map([...index.groups.keys()].map((id,i)=>[id,{width:172+i*3,height:90+i*14}]));
  const diagram=taskDiagramLayout(index,sizes),box=id=>diagram.boxes.get(id);
  assert.equal(index.unplaced[0].to_id,'T99');assert.equal(index.taskEdges.length,4);
  assert.equal(diagram.boxes.size,6);assert(!box('T99'));assert(!box('B88'));
  assert.equal(diagram.links.length,5);assert.equal(box(taskRootId).rank,0);
  const byChild=new Map(diagram.links.map(e=>[e.to,e.from]));
  assert.deepEqual([...byChild].sort(),[['T1','T4'],['T4',taskRootId],['T6','T1'],['T7',taskRootId],['T9',taskRootId]]);
  assert.equal(diagram.focus.x,box(taskRootId).x+box(taskRootId).width/2);
  for(const e of diagram.links){
    if(e.kind==='branch'){assert.equal(e.record.raw.from_id,e.to);assert.equal(e.record.raw.to_id,e.from);}
    else {assert.equal(e.kind,'root');assert.equal(e.from,taskRootId);assert.equal(e.record,null);}
    assert.equal(box(e.from).rank+1,box(e.to).rank);assert(box(e.from).y+box(e.from).height<box(e.to).y);
    assert.equal(e.points[0][1],box(e.from).y+box(e.from).height+2);assert.equal(e.points.at(-1)[1],box(e.to).y-3);
    for(const [px,py] of e.points){assert(px>=0 && px<=diagram.width && py>=0 && py<=diagram.height);}
    // No segment cuts through an unrelated task card, even on a skipped rank.
    for(const [id,b] of diagram.boxes){if(id===e.from||id===e.to)continue;
      e.points.slice(1).forEach(([x,y],i)=>{const [u,v]=e.points[i];
        assert(!(Math.max(x,u)>b.x && Math.min(x,u)<b.x+b.width && Math.max(y,v)>b.y && Math.min(y,v)<b.y+b.height));
      });
    }
  }
  const withShared=taskDiagramLayout(index,sizes,true);
  assert(withShared.boxes.has('@shared:0'));
  assert.deepEqual(withShared.links.filter(e=>e.kind==='shared').map(e=>[e.from,e.to]),[['T1','@shared:0'],['T4','@shared:0']]);
  assert.deepEqual(withShared.links.filter(e=>e.kind!=='shared'),diagram.links);
  for(const [id,b] of diagram.boxes)assert.deepEqual(withShared.boxes.get(id),b);
  for(const [a,ba] of withShared.boxes)for(const [b,bb] of withShared.boxes)if(a!==b)
    assert(ba.x+ba.width<=bb.x||bb.x+bb.width<=ba.x||ba.y+ba.height<=bb.y||bb.y+bb.height<=ba.y);
});

test('cycles open at a cycle member, not a leaf; original duplicate, self and non-hierarchy records remain inspectable',()=>{
  const relations=[edge('T3','T4','part_of'),edge('T4','T3','part_of'),edge('T4','T4','part_of'),edge('T3','T4','part_of'),edge('T3','T3','legacy_type'),edge('T1','T4','part_of')];
  const idx=taskSetIndex(overviewGraph({nodes:[],relations:[],taskState:{nodes:[1,3,4].map(task),relations}},'tsg'));
  const drawing=taskDiagramLayout(idx,new Map());
  assert.deepEqual(idx.taskEdges.map(e=>e.raw),relations);
  assert.deepEqual(drawing.links.map(({from,to})=>[from,to]),[[taskRootId,'T3'],['T3','T4'],['T4','T1']]);
  assert(drawing.links.every(e=>e.points.flat().every(Number.isFinite)));
});

test('empty snapshot has a root only; isolated tasks or shared beliefs alone never invent subtask levels',()=>{
  for(const nodes of [[],[2,4,5].map(task)]){
    const relations=nodes.map(t=>edge('B88',t.id,'addresses'));
    const idx=taskSetIndex(overviewGraph({nodes:[belief(88)],relations:[],taskState:{nodes,relations}},'tsg'));
    const tree=taskDiagramLayout(idx,new Map());
    assert.equal(tree.boxes.size,1+nodes.length);
    assert.deepEqual(tree.links.map(l=>[l.from,l.to]),nodes.map(t=>[taskRootId,t.id]));
    assert.equal(idx.taskEdges.length,0);
  }
});

test('TSG belongs to its own saved snapshot, missing, empty and populated records stay distinct',()=>{
  const history=[{snapshot:{nodes:[belief(1)],relations:[]}},
    {snapshot:{nodes:[belief(2)],relations:[],task_state:{nodes:[],relations:[]}}},
    {snapshot}].map(v=>JSON.stringify(v)).join('\n');
  const c=ingestFiles([file('state.json',{snapshot}),file('graph_snapshots.jsonl',history)]);
  const [missing,empty,recorded]=graphViews(c).turns.map(g=>overviewGraph(g,'tsg'));
  assert.equal(missing.available,false);assert.equal(empty.available,true);assert.equal(empty.nodes.length,0);
  assert.equal(recorded.taskCount,3);
  assert.equal(normalizeView({graphLayer:'tsg'}).graphLayer,'tsg');
  assert.equal(normalizeView({graphLayer:'invalid'}).graphLayer,'bcg');
});

test('recorded dangling and duplicate links stay visible without invented facts or links',()=>{
  const g={nodes:[belief(2)],relations:[],taskState:{nodes:[task(2)],relations:[
    edge('T2','B90','related_via_graph'),edge('T2','B90','related_via_graph'),
    edge('T2','T2','part_of'),{type:'incomplete'},
  ]}};
  const t=overviewGraph(g,'tsg');
  assert.equal(t.byId.get('B90').kind,'missing');assert.equal(t.missingCount,1);
  assert.equal(t.beliefCount,0);assert.equal(t.incident.get('T2').length,3);assert.equal(t.invalidEdges,1);
});

test('BCG chronology uses recorded Graph row indices, numeric IDs and snapshot confidence, not connectivity',()=>{
  const rows=[
    {id:21,role:'assistant',node_type:'decision',source:{turn_id:8},confidence:.974},
    {id:12,role:'assistant',source:{turn_id:1},confidence:0},
    {id:2,role:'assistant',source:{turn_id:1}},
    {id:3,source:{turn_index:'4',type:'function'}},
    {id:40,source:{turn_id:-1}},{id:0,source:{trajectory_index:0,role:'user'}},
  ];
  const snapshot={nodes:rows,relations:[edge(21,2,'depends_on'),edge(2,12,'supplements'),edge(12,99,'contradicts')]};
  const graph=overviewGraph(snapshot), map=trajectoryLayout(graph), {positions:p,columns}=map;
  assert.equal(trajectoryLayout(graph),map);
  assert.deepEqual(columns.map(c=>c.turn),[0,1,4,8,null]);
  assert.deepEqual(columns.map(c=>c.count),[1,2,1,1,2]);
  assert.equal(columns.at(-1).label,'轮次未记录');
  assert.equal(p.size,graph.nodes.length);
  assert.equal(p.get('B2').x,p.get('B12').x);
  assert.equal(p.get('B2').y+timelineSize.row,p.get('B12').y);
  assert(p.get('B12').x<p.get('B3').x && p.get('B3').x<p.get('B21').x);
  assert.equal(p.get('B12').confidence,'0.00');assert.equal(p.get('B2').confidence,'—');
  assert.equal(p.get('B21').caption,'D#21');assert.equal(p.get('B21').confidence,'0.97');
  assert.equal(p.get('B3').role,'tool');assert.equal(p.get('B99').turn,null);
  assert.deepEqual(graph.relations.map(r=>r.raw),snapshot.relations);
  const reordered=trajectoryLayout(overviewGraph({nodes:[...rows].reverse(),relations:[...snapshot.relations].reverse()}));
  assert.deepEqual([...reordered.positions], [...p]);
});

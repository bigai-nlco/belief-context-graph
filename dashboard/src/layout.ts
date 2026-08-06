import type { BeliefMemoryGraph, BeliefNode, LayoutMode } from "./types.ts";

/**
 * Graph layout functions extracted from main.ts (step 12): pure position
 * computation. `applyLayout` mutates node coordinates in place.
 */

const typeOrder: Record<string, number> = {
  Evidence: 0,
  Claim: 1,
  BeliefVariable: 2,
  Factor: 3,
  Decision: 4,
  Other: 2,
};

export function originalLayout(nodes: BeliefNode[]): Map<string, { x: number; y: number }> {
  const hasPositions = nodes.some((node) => node.x > 0 || node.y > 0);
  if (hasPositions) return new Map(nodes.map((node) => [node.id, { x: node.x, y: node.y }]));
  return layeredLayout(nodes);
}

export function layeredLayout(nodes: BeliefNode[]): Map<string, { x: number; y: number }> {
  const groups = new Map<number, BeliefNode[]>();
  for (const node of nodes) {
    const level = typeOrder[node.type] ?? typeOrder.Other;
    const group = groups.get(level) ?? [];
    group.push(node);
    groups.set(level, group);
  }
  const layout = new Map<string, { x: number; y: number }>();
  for (const [level, group] of groups) {
    const x = 82 + level * 184;
    const step = group.length > 1 ? 420 / (group.length - 1) : 0;
    group.forEach((node, index) => {
      layout.set(node.id, {
        x,
        y: group.length > 1 ? 70 + index * step : 280,
      });
    });
  }
  return layout;
}

export function starLayout(nodes: BeliefNode[]): Map<string, { x: number; y: number }> {
  const center =
    nodes.find((node) => node.type === "Decision") ??
    nodes.find((node) => node.type === "BeliefVariable") ??
    nodes[0];
  const layout = new Map<string, { x: number; y: number }>();
  if (!center) return layout;
  layout.set(center.id, { x: 450, y: 280 });
  const outer = nodes.filter((node) => node.id !== center.id);
  outer.forEach((node, index) => {
    const angle = -Math.PI / 2 + (index / Math.max(1, outer.length)) * Math.PI * 2;
    layout.set(node.id, {
      x: 450 + Math.cos(angle) * 225,
      y: 280 + Math.sin(angle) * 205,
    });
  });
  return layout;
}

export function applyLayout(memory: BeliefMemoryGraph, mode: LayoutMode = "original"): void {
  const layout =
    mode === "star"
      ? starLayout(memory.nodes)
      : mode === "layers"
        ? layeredLayout(memory.nodes)
        : originalLayout(memory.nodes);
  for (const node of memory.nodes) {
    const point = layout.get(node.id);
    if (point) {
      node.x = point.x;
      node.y = point.y;
    }
  }
}

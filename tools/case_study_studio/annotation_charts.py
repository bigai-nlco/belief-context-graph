"""Dependency-free, accessible SVG charts and a portable HTML statistics report."""
from __future__ import annotations

from html import escape

from tools.case_study_studio.annotation_stats import MISSING, UNKNOWN

COLORS = {
    "有": "#263c59", "无": "#c6ced8", "无法判断": "#b89d64", MISSING: "#e2e6ed", UNKNOWN: "#555d6a",
    "充分利用": "#263c59", "部分利用": "#8498b2", "未利用": "#ab6a7e", "误用": "#7a2942", "不适用": "#c6ced8",
    "正向": "#426d63", "无明显变化": "#a3aebb", "负向": "#943d56", "待验证": "#b89d64",
    "正确 / 达标": "#426d63", "部分正确 / 部分达标": "#8498b2", "错误 / 未达标": "#943d56",
    "未完成": "#b89d64", "尚未评判": "#a3aebb",
    "仅 AI": "#263c59", "仅人工": "#943d56", "AI 和人工": "#648782", "待标注": "#c6ced8",
}
RULES = [
    "只汇总已保存的标注字段，不调用模型，不重新分析轨迹、Graph 或报告。",
    "人工以“完成标注”为准；AI 以后端标注任务完成为准，生成文件内部的 draft 不影响 AI 完成状态。草稿、失败和进行中的标注不计入分析。",
    "每位人工标注者独立统计；AI 按模型、思考强度和 Prompt 版本分组。同一题的不同标注不投票、不混合。",
    "维度统计单位是题，不是关键过程数。每题每维度只计一次。利用与效果的分母是该维度“存在影响可能 = 有”的题数；无法判断不计入这一分母。",
    "取消选择维度后保留的旧评分不计入利用与效果。空项和未知值分别显示，不视为负面结果。",
    "自定义维度仅在名称和说明完全一致（去除首尾空白）时跨题合并。C1 等局部编号不用于跨题匹配。",
    "任务结果是标注者的判断，不是 Benchmark 正确率或官方 Judge 分数。改进环节允许多选，比例不能相加。",
    "关键过程类别、搜索判断与证据强度仅使用显式标签，不从文字推断；同组归并后的过程不拆成多个事件。统计不生成综合分数或因果结论。",
]
STYLE = """
:root{color-scheme:light;--ink:#26364b;--muted:#617083;--line:#e0e5ec;--accent:#7c2945}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.65 system-ui,-apple-system,'Segoe UI',sans-serif}
main{max-width:1400px;margin:auto;padding:32px}h1,h2,h3,p{margin:0}h1{font-size:30px;line-height:1.3}h2{font-size:22px}h3{font-size:17px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.15em;color:var(--accent)}.subtitle{margin-top:10px;color:var(--muted);overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:24px 0}.metric,.card,.rules,.audit{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}
.metric span{display:block;color:var(--muted);font-size:13px}.metric strong{font-size:30px;line-height:1.5;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:18px 0 28px}.wide{grid-column:1/-1}.card{min-width:0}.card p{font-size:13px;color:var(--muted);margin:6px 0 12px}
.chart{overflow-x:auto}.chart svg{display:block;width:100%;min-width:500px;height:auto}.cohort{margin:34px 0;scroll-margin-top:12px}.cohort-title{overflow-wrap:anywhere;border-left:4px solid var(--accent);padding-left:14px}
.cohort-title p{font-size:13px;color:var(--muted);margin-top:4px}.note{color:var(--muted);font-size:13px}.empty{padding:32px;text-align:center;background:#fff;border:1px dashed #ccd4df;border-radius:12px;margin:20px 0}
.rules{margin:20px 0}summary{cursor:pointer;font-weight:600}ol{padding-left:24px}li{margin:7px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}th,td{border-bottom:1px solid var(--line);padding:9px 12px;text-align:left;white-space:nowrap}th{background:#f4f6f9}td.wrap{white-space:normal;min-width:200px;overflow-wrap:anywhere}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.toc{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:16px}code{font-size:12px;overflow-wrap:anywhere}.warnings{border-left:4px solid #b89d64}.footer{margin-top:28px;color:var(--muted);font-size:12px}
@media(max-width:1000px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
@media(max-width:600px){main{padding:18px 12px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.metric,.card,.rules,.audit{padding:14px}.metric strong{font-size:26px}h1{font-size:25px}h2{font-size:20px}.chart svg{min-width:510px}}
@media print{body{background:white}main{padding:0}.card,.metric,.rules{break-inside:avoid}.cohort{break-before:page}.chart{overflow:visible}.chart svg{min-width:0}.grid{display:block}.card{margin:12px 0}}
"""


def _color(label: str) -> str:
    return COLORS.get(label, "#627c97")


def _short(text: str, limit: int = 14) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def bars(title: str, rows: list[tuple[str, dict[str, int]]], *, denominator: int | None = None) -> str:
    """Count-scaled horizontal stacks; whole counts, zero-based and no smoothing."""
    labels = list(dict.fromkeys(label for _, counts in rows for label, n in counts.items() if n))
    legend = []
    x, y = 14, 28
    for label in labels:
        width = 31 + sum(14 if ord(c) > 127 else 8 for c in label)
        if x + width > 622:
            x, y = 14, y + 26
        legend.append(f'<rect x="{x}" y="{y-11}" width="10" height="10" rx="2" fill="{_color(label)}"/>'
            f'<text x="{x+16}" y="{y-2}" font-size="13">{escape(label)}</text>')
        x += width
    top = y + 22
    row_height, left, bar_width = 43, 195, 365
    height = int(top + max(1, len(rows)) * row_height + 32)
    maximum = max([sum(counts.values()) for _, counts in rows] + [denominator or 0, 1])
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {height}" role="img" aria-label="{escape(title, quote=True)}">',
        f'<title>{escape(title)}</title>',
        '<g font-family="system-ui, sans-serif" fill="#26364b">', *legend]
    for tick in range(5):
        tx = left + tick * bar_width / 4
        chunks.append(f'<line x1="{tx:g}" y1="{top-5}" x2="{tx:g}" y2="{height-30}" stroke="#edf0f4"/>')
    for i, (name, counts) in enumerate(rows):
        by = top + i * row_height
        total = sum(counts.values())
        description = "，".join(f"{k} {v}" for k, v in counts.items() if v) or "0 题"
        chunks.append(f'<g><title>{escape(name + "：" + description)}</title><text x="181" y="{by+18}" text-anchor="end" font-size="15">{escape(_short(name))}</text>')
        bx = float(left)
        for label, count in counts.items():
            if not count:
                continue
            width = count / maximum * bar_width
            chunks.append(f'<rect x="{bx:.3f}" y="{by}" width="{width:.3f}" height="26" fill="{_color(label)}"><title>{escape(label)}：{count}</title></rect>')
            if width >= 27:
                text_color = "#26364b" if _color(label) in {"#c6ced8", "#e2e6ed", "#a3aebb", "#b89d64", "#8498b2"} else "#ffffff"
                chunks.append(f'<text x="{bx+width/2:.3f}" y="{by+18}" text-anchor="middle" font-size="14" fill="{text_color}">{count}</text>')
            bx += width
        chunks.append(f'<text x="{left+bar_width+12}" y="{by+18}" font-size="14">{total}</text></g>')
    chunks.append(f'<text x="{left}" y="{height-8}" font-size="12" fill="#617083">0</text>'
        f'<text x="{left+bar_width}" y="{height-8}" text-anchor="end" font-size="12" fill="#617083">{maximum}</text></g></svg>')
    return "".join(chunks)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{escape(h)}</th>" for h in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f'<td class="wrap">{escape(str(v))}</td>' for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def render_report(stats: dict) -> tuple[str, dict[str, str]]:
    charts = {}

    def card(name: str, title: str, note: str, rows: list, denominator: int | None = None) -> str:
        svg = bars(title, rows, denominator=denominator)
        charts[name] = svg
        return f'<section class="card"><h3>{escape(title)}</h3><p>{escape(note)}</p><div class="chart">{svg}</div></section>'

    coverage = stats["coverage"]
    run = stats["run"]
    title = f"{run['number']} · 标注统计"
    header = f'<div class="eyebrow">BCG · ANNOTATION STATISTICS</div><h1>{escape(title)}</h1><p class="subtitle">{escape(run["title"])}</p>'
    metrics = [("已完成标注的题目", f"{coverage['reviewed_tasks']} / {coverage['eligible_tasks']}"),
        ("Run 总题数", str(coverage["total_tasks"])), ("独立标注组", str(len(stats["cohorts"]))),
        ("未计入的标注记录", str(len(stats["excluded_records"])))]
    header += '<div class="metrics">' + "".join(f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label, value in metrics) + '</div>'
    header += '<div class="grid">' + card("coverage", "标注覆盖", "以正常完成的任务为分母，人工与 AI 同时完成的题只计一次。",
        [("可标注任务", {"仅 AI": coverage["ai_only"], "仅人工": coverage["human_only"],
         "AI 和人工": coverage["both"], "待标注": coverage["unreviewed"]})], coverage["eligible_tasks"])
    header += '<section class="card"><h3>统计口径</h3><p>按已保存标签确定性计数，不新增评价。</p><ul><li>人工与 AI 独立呈现，不合成一个总评。</li><li>利用 / 效果：每题每维度计一次。</li><li>空项、无法判断和不适用单独保留。</li></ul>'
    header += f'<p>{coverage["ineligible_tasks"]} 道未正常完成的任务不计入可标注题数。图表均从 0 开始，以数量为单位。</p></section></div>'
    header += '<nav class="toc" aria-label="标注组">' + "".join(f'<a href="#cohort-{c["id"]}">{escape(c["label"])}</a>' for c in stats["cohorts"]) + '</nav>'
    sections = []
    for cohort in stats["cohorts"]:
        cid = cohort["id"]
        n = cohort["tasks"]
        dims = cohort["dimensions"]
        built_in = [d for d in dims if not d["id"].startswith("custom-")]
        custom = [d for d in dims if d["id"].startswith("custom-")]
        body = f'<section class="cohort" id="cohort-{cid}"><div class="cohort-title"><h2>{escape(cohort["label"])}</h2><p>{n} 道已完成标注 · {cohort["events"]} 个关键过程 · {cohort["no_applicable_dimensions"]} 道明确未发现相关维度</p></div><div class="grid">'
        for slug, title_text, note, metric in [
            ("opportunity", "哪些维度存在影响可能", f"分母：本组已完成标注的 {n} 题。不是仅根据任务类型判断。", "opportunity"),
            ("utilization", "Agent 是否利用了 BCG", "分母：各维度“有影响可能”的题数，显示在条形右侧。", "utilization"),
            ("effect", "利用后产生的效果", "同一维度每题只计一次。未填写不等于没有效果。", "effect"),
        ]:
            body += card(f"{cid}-{slug}", title_text, note, [(d["title"], d[metric]) for d in built_in])
        body += card(f"{cid}-focus", "优先改进环节", f"分母：本组 {n} 题；可多选，每题每环节计一次。未填写：{cohort['focus_unfilled_tasks']} 题。",
            [(label, {"题数": count}) for label, count in cohort["focus"].items()], n)
        body += card(f"{cid}-result", "标注中的任务结果", "这是标注判断，不是官方 Judge 评分或 Benchmark 正确率。",
            [(label, {label: count}) for label, count in cohort["results"].items() if count], n)
        body += card(f"{cid}-strength", "关键过程证据强度", f"统计单位：{cohort['events']} 个已保存关键过程；不与题数相加。",
            [(label, {label: count}) for label, count in cohort["event_strength"].items() if count], cohort["events"])
        body += '</div>'
        if custom:
            body += '<details class="audit" open><summary>自定义维度</summary>' + _table(
                ["名称", "定义", "定义于几题", "有影响可能", "充分 / 部分利用", "未利用 / 误用", "正向 / 负向"],
                [[d["title"], d["description"], d["available_tasks"], d["applicable_tasks"],
                  f"{d['utilization']['充分利用']} / {d['utilization']['部分利用']}",
                  f"{d['utilization']['未利用']} / {d['utilization']['误用']}",
                  f"{d['effect']['正向']} / {d['effect']['负向']}"] for d in custom]) + '</details>'
        body += '<details class="audit"><summary>查看逐题维度与关键过程索引</summary><p class="note">数字来自原始选择。未选维度的残留评分不参与统计。</p>' + _table(
            ["Task", "维度", "影响可能", "利用", "效果", "关键过程 ID"],
            [[r["task_id"], r["title"], r["opportunity"], r["utilization"] or "—", r["effect"] or "—", ", ".join(r["event_ids"]) or "—"]
             for r in stats["dimension_rows"] if r["cohort_id"] == cid and r["opportunity"] in {"有", "无法判断"}]) + '</details>'
        kinds = {"benefit": "发挥优势", "missed_opportunity": "未把握的机会", "misuse": "误用", MISSING: MISSING, UNKNOWN: UNKNOWN}
        body += '<details class="rules"><summary>关键过程标签明细</summary><p class="note">只统计已保存的显式标签。旧版与人工标注缺少的字段显示为未填写，不从文字补判。</p>'
        body += _table(["类别", "关键过程数"], [[kinds[k], v] for k, v in cohort["event_kind"].items()])
        body += _table(["搜索判断", "关键过程数"], [[k, v] for k, v in cohort["search"].items()]) + '</details></section>'
        sections.append(body)
    if not sections:
        sections.append('<section class="empty"><h2>暂无已完成标注</h2><p>完成人工标注或 AI 标注后刷新此页即可统计。不会自动启动 AI 标注。</p></section>')
    tail = '<details class="rules"><summary>完整计算规则与复现方式</summary><ol>' + "".join(f'<li>{escape(rule)}</li>' for rule in RULES) + '</ol>'
    tail += '<p>下载包包含输入快照、统计 JSON、CSV 和 SVG。相同输入快照会生成字节一致的报告与 ZIP。更新标注后请刷新并重新导出。</p><code>python -m tools.case_study_studio.annotation_stats --input input_snapshot.json --output /tmp/annotation-replay</code></details>'
    if stats["excluded_records"]:
        tail += '<details class="audit"><summary>未计入的标注记录</summary>' + _table(["Task", "来源", "标注者", "原因"],
            [[r["task_id"], r["source"], r["reviewer"], r["reason"]] for r in stats["excluded_records"]]) + '</details>'
    if stats["warnings"]:
        tail += '<details class="rules warnings"><summary>数据检查提示</summary><ul>' + "".join(f'<li>{escape(w)}</li>' for w in stats["warnings"]) + '</ul></details>'
    tail += f'<footer class="footer">统计版本 {stats["version"]} · 输入快照 SHA-256 <code>{stats["input_sha256"]}</code></footer>'
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
    html += f'<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{header}{"".join(sections)}{tail}</main></body></html>'
    return html, charts

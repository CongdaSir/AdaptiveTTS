"""Parse and visualize per-task AdaptiveTTS search-tree JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip malformed JSONL line {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                print(f"warning: skip non-object JSONL line {line_no}", file=sys.stderr)
    return records


def _children(record: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    nodes = record.get("nodes", [])
    by_parent: dict[str, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    known = {str(node.get("node_id")) for node in nodes}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is None or str(parent) not in known:
            roots.append(node)
        else:
            by_parent.setdefault(str(parent), []).append(node)
    return by_parent, roots


def _depth(record: dict[str, Any]) -> int:
    by_parent, roots = _children(record)
    def visit(node: dict[str, Any], level: int) -> int:
        children = by_parent.get(str(node.get("node_id")), [])
        return max([level] + [visit(child, level + 1) for child in children])
    return max((visit(root, 1) for root in roots), default=0)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter()
    node_count = 0
    depths = []
    costs = []
    correct = 0
    for record in records:
        actions.update(item.get("action", "UNKNOWN") for item in record.get("actions", []))
        node_count += len(record.get("nodes", []))
        depths.append(_depth(record))
        costs.append(float(record.get("spent_budget", 0) or 0))
        correct += int(bool(record.get("correct", False)))
    total = len(records)
    return {
        "tasks": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_nodes": node_count,
        "average_nodes": node_count / total if total else 0.0,
        "max_tree_depth": max(depths, default=0),
        "average_spent_budget": sum(costs) / total if total else 0.0,
        "action_counts": dict(actions),
    }


def _short(value: Any, limit: int = 180) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _node_line(node: dict[str, Any], show_traces: bool = False) -> list[str]:
    usage = node.get("usage") or {}
    verifications = node.get("verifications") or []
    label = f"{node.get('node_id')} [{node.get('created_by', '?')}]"
    details = [f"method={node.get('method', '')}", f"cost={usage.get('cost', 0)}"]
    if verifications:
        passed = sum(bool(item.get("passed")) for item in verifications)
        details.append(f"verify={passed}/{len(verifications)}")
    lines = [f"{label} ({', '.join(details)})", f"  answer: {_short(node.get('answer'))}"]
    if node.get("summary"):
        lines.append(f"  summary: {_short(node['summary'])}")
    if node.get("failed") or node.get("error"):
        lines.append(f"  error: {_short(node.get('error') or 'failed')}")
    if show_traces:
        for step in node.get("trajectory", []):
            lines.append(f"  trace/{step.get('kind', 'step')}: {_short(step.get('content', step))}")
    return lines


def render_text(record: dict[str, Any], show_traces: bool = False) -> str:
    by_parent, roots = _children(record)
    lines = [
        f"Task {record.get('task_id')} | correct={record.get('correct')} | prediction={record.get('prediction')}",
        f"budget={record.get('spent_budget')} | stopped={record.get('stopped_reason')} | nodes={len(record.get('nodes', []))}",
        f"prompt: {_short(record.get('prompt'), 240)}",
    ]
    actions = record.get("actions", [])
    if actions:
        lines.append("actions: " + " -> ".join(f"{a.get('action')}({a.get('target_node_id') or '-'})" for a in actions))
    def visit(node: dict[str, Any], prefix: str, is_last: bool) -> None:
        branch = "└─ " if is_last else "├─ "
        node_lines = _node_line(node, show_traces)
        lines.append(prefix + branch + node_lines[0])
        lines.extend(prefix + ("   " if is_last else "│  ") + line for line in node_lines[1:])
        children = by_parent.get(str(node.get("node_id")), [])
        for index, child in enumerate(children):
            visit(child, prefix + ("   " if is_last else "│  "), index == len(children) - 1)
    for index, root in enumerate(roots):
        visit(root, "", index == len(roots) - 1)
    return "\n".join(lines)


def _html_page(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdaptiveTTS Search Trees</title>
<style>
body{{font:14px system-ui,sans-serif;background:#f6f7f9;color:#20242a;margin:24px}} h1{{margin-bottom:6px}}
#summary{{color:#555;margin-bottom:18px}} .task{{background:white;border:1px solid #d9dde3;border-radius:8px;margin:12px 0;padding:14px}}
.task h2{{font-size:16px;margin:0 0 8px}} .meta{{color:#666;font-size:12px;margin-bottom:8px}} .tree{{font-family:ui-monospace,monospace;font-size:12px}}
details{{margin:5px 0 5px 18px}} summary{{cursor:pointer}} .answer{{color:#0b6b3a;white-space:pre-wrap;word-break:break-word}} .bad{{color:#a12622}}
.trace{{margin:5px 0 5px 20px;padding:7px;background:#f1f3f5;white-space:pre-wrap;max-height:220px;overflow:auto}}
</style></head><body><h1>AdaptiveTTS Search Trees</h1><div id="summary"></div><main id="app"></main>
<script id="tree-data" type="application/json">{payload}</script>
<script>
const records=JSON.parse(document.getElementById('tree-data').textContent), app=document.getElementById('app');
const total=records.length, correct=records.filter(x=>x.correct).length, nodes=records.reduce((n,x)=>n+(x.nodes||[]).length,0);
document.getElementById('summary').textContent=`tasks=${{total}} | correct=${{correct}} | accuracy=${{total?(correct/total).toFixed(3):'0.000'}} | nodes=${{nodes}}`;
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
for(const r of records){{
 const box=document.createElement('section'); box.className='task';
 box.innerHTML=`<h2>Task ${{esc(r.task_id)}} <span class="${{r.correct?'':'bad'}}">${{r.correct?'✓ correct':'✗ incorrect'}}</span></h2><div class="meta">prediction=${{esc(r.prediction)}} | budget=${{esc(r.spent_budget)}} | stopped=${{esc(r.stopped_reason)}} | nodes=${{(r.nodes||[]).length}}</div><div class="meta">${{esc(r.prompt)}}</div><div class="tree"></div>`;
 const tree=box.querySelector('.tree'), children=new Map();
 for(const n of r.nodes||[]){{const p=n.parent_id==null?'':String(n.parent_id); if(!children.has(p))children.set(p,[]); children.get(p).push(n);}}
 function add(n,parent){{const d=document.createElement('details'); d.open=true; const v=n.verifications||[]; d.innerHTML=`<summary>${{esc(n.node_id)}} [${{esc(n.created_by)}}] method=${{esc(n.method)}} verify=${{v.filter(x=>x.passed).length}}/${{v.length}}</summary><div class="answer">answer: ${{esc(n.answer)}}<br>summary: ${{esc(n.summary)}}</div>`; for(const t of n.trajectory||[]){{const tr=document.createElement('div');tr.className='trace';tr.textContent=`${{t.kind||'step'}}: ${{t.content||JSON.stringify(t)}}`;d.appendChild(tr)}}; parent.appendChild(d); for(const c of children.get(String(n.node_id))||[])add(c,d)}}
 for(const n of children.get('')||[])add(n,tree); app.appendChild(box);
}}
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and visualize AdaptiveTTS search-tree JSONL.")
    parser.add_argument("--input", required=True, help="Per-task search-tree JSONL produced by evaluate.")
    parser.add_argument("--task-id", help="Only display this task id.")
    parser.add_argument("--format", choices=("text", "json", "html"), default="text")
    parser.add_argument("--output", help="Output file; HTML defaults to a sibling .html file.")
    parser.add_argument("--show-traces", action="store_true", help="Include intermediate trajectory steps in text output.")
    args = parser.parse_args()
    records = load_records(args.input)
    if args.task_id is not None:
        records = [record for record in records if str(record.get("task_id")) == args.task_id]
    if args.format == "json":
        output = json.dumps({"summary": summarize(records), "tasks": records}, ensure_ascii=False, indent=2)
    elif args.format == "html":
        output = _html_page(records)
        target = Path(args.output) if args.output else Path(args.input).with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        print(target)
        return
    else:
        output = json.dumps(summarize(records), ensure_ascii=False, indent=2) + "\n\n"
        output += "\n\n".join(render_text(record, args.show_traces) for record in records)
    if args.output:
        Path(args.output).write_text(output + ("\n" if not output.endswith("\n") else ""), encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()

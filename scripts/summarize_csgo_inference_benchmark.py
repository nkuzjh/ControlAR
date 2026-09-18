#!/usr/bin/env python3
"""Read-only CSGO inference-study summarizer; it never imports torch or starts CUDA."""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "inference_speed_study"
TARGETS = {"discrete": (20000, "离散"), "continuous": (12800, "连续")}
KNOWN_EXTERNAL = {432478: "openpi 烟测训练", 433651: "openvla-oft 训练"}

def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def read_json(path):
    if not path.is_file(): return None, f"缺失：{path.name}"
    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
        return (value, None) if isinstance(value, dict) else (None, f"{path.name} 不是 JSON 对象")
    except (OSError, json.JSONDecodeError) as e: return None, f"读取失败 {path.name}：{e}"

def read_progress(path):
    if not path.is_file(): return [], [f"缺失：{path.name}"]
    rows, errors = [], []
    try:
        with path.open(encoding="utf-8") as f:
            for no, line in enumerate(f, 1):
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict): rows.append(row)
                    else: errors.append(f"第{no}行不是对象")
                except json.JSONDecodeError as e: errors.append(f"第{no}行 JSON 失败：{e}")
    except OSError as e: errors.append(f"读取 progress.jsonl 失败：{e}")
    return rows, errors

def sample_ids(value):
    return [str(x.get("sample_id")) if isinstance(x, dict) and "sample_id" in x else str(x) for x in value] if isinstance(value, list) else []

def stats(values):
    values = [x for x in (num(v) for v in values) if x is not None]
    return {"count": len(values), "min": min(values), "max": max(values)} if values else {"count": 0}

def compare(left, right):
    return {"equal_in_order": bool(left and right and left == right), "left_count": len(left), "right_count": len(right), "missing_from_right": sorted(set(left)-set(right)), "extra_in_right": sorted(set(right)-set(left)), "duplicate_left": sorted({x for x in left if left.count(x)>1}), "duplicate_right": sorted({x for x in right if right.count(x)>1})}

def id_check(expected, summary, progress):
    observed = summary or progress; checks = {}
    for name, left, right in (("expected_vs_summary", expected, summary), ("expected_vs_progress", expected, progress), ("summary_vs_progress", summary, progress)):
        if left and right: checks[name] = compare(left, right)
    bad = any(c["missing_from_right"] or c["extra_in_right"] or c["duplicate_left"] or c["duplicate_right"] for c in checks.values())
    state = "inconsistent" if bad else ("consistent" if checks and all(c["equal_in_order"] for c in checks.values()) else "unknown")
    return {"status": state, "expected_count": len(expected), "summary_count": len(summary), "progress_count": len(progress), "observed_count": len(observed), "checks": checks}

def gpu_summary(records, own_pid):
    snapshots, utils, memories, processes = [], [], [], {}
    for row in records:
        snap = row.get("gpu_snapshot")
        if not isinstance(snap, dict): continue
        self_pid = num(snap.get("pid")) or own_pid; others = []
        for p in snap.get("compute_processes", []) if isinstance(snap.get("compute_processes"), list) else []:
            if not isinstance(p, dict): continue
            pid, used = num(p.get("pid")), num(p.get("used_memory_mib"))
            if pid is None or int(pid) == int(self_pid or -1): continue
            item = {"pid": int(pid), "used_memory_mib": used}; others.append(item)
            entry = processes.setdefault(str(int(pid)), {"pid": int(pid), "_memory": []})
            if used is not None: entry["_memory"].append(used)
        gpus = []
        for g in snap.get("gpus", []) if isinstance(snap.get("gpus"), list) else []:
            if not isinstance(g, dict): continue
            util, used, total = (num(g.get(k)) for k in ("utilization_gpu_percent", "memory_used_mib", "memory_total_mib"))
            gpus.append({"index": g.get("index"), "utilization_gpu_percent": util, "memory_used_mib": used, "memory_total_mib": total})
            if util is not None: utils.append(util)
            if used is not None: memories.append(used)
        snapshots.append({"event": row.get("event"), "task": row.get("task"), "batch_index": row.get("batch_index"), "gpus": gpus, "other_processes": others})
    for p in processes.values(): p["memory_mib"] = stats(p.pop("_memory"))
    return {"snapshot_count": len(snapshots), "gpu_utilization_percent": stats(utils), "gpu_memory_used_mib": stats(memories), "other_processes_by_pid": processes, "per_batch": snapshots}

def task_summary(task, result, records, warmup_seconds):
    ts = result.get("task_summaries", {}).get(task, {}) if isinstance(result.get("task_summaries"), dict) else {}
    ts = ts if isinstance(ts, dict) else {}
    batches = [r for r in records if r.get("event") == "batch" and r.get("task") == task]
    stages = ts.get("stage_totals_s", {}) if isinstance(ts.get("stage_totals_s"), dict) else {}
    count, total = num(ts.get("sample_count")), num(stages.get("total_s")); spi = total/count if count and total is not None else None
    observed_batches = [num(r.get("timing_s", {}).get("total_s")) for r in batches if isinstance(r.get("timing_s"), dict)]; observed_batches = [x for x in observed_batches if x is not None]
    selected = result.get("selected_samples", {}) if isinstance(result.get("selected_samples"), dict) else {}; expected = sample_ids(selected.get(task)); summary_ids = sample_ids(ts.get("sample_ids")); progress_ids = [x for r in batches for x in sample_ids(r.get("sample_ids"))]
    batch_count = int(num(ts.get("batch_count")) or len(batches)); full_batch = total/batch_count if total is not None and batch_count else (sum(observed_batches)/len(observed_batches) if observed_batches else None)
    state = "complete" if result.get("status") == "complete" and count and total is not None else "incomplete"
    if result.get("status") == "error" or any(r.get("event") == "error" for r in records): state = "failed"
    alloc, reserved = num(ts.get("max_torch_allocated_peak_bytes")), num(ts.get("max_torch_reserved_peak_bytes"))
    if alloc is None: alloc = max((num(r.get("torch_allocated_peak_bytes")) or 0 for r in batches), default=0) or None
    if reserved is None: reserved = max((num(r.get("torch_reserved_peak_bytes")) or 0 for r in batches), default=0) or None
    target = TARGETS[task][0]
    return {"status": state, "count": int(count) if count is not None else None, "batch_count": batch_count, "total_seconds": total, "seconds_per_image": spi, "observed_batch_seconds": observed_batches, "full_batch_seconds": full_batch, "warmup_seconds": warmup_seconds, "torch_allocated_peak_gib": alloc/2**30 if alloc else None, "torch_reserved_peak_gib": reserved/2**30 if reserved else None, "projected_target": task+str(target), "projected_hours": spi*target/3600 if spi is not None else None, "sample_id_consistency": id_check(expected, summary_ids, progress_ids), "observed_sample_ids": summary_ids or progress_ids, "progress_batch_count": len(batches)}

def mode_summary(directory, result, result_error, records, progress_errors):
    result = result if isinstance(result, dict) else {}; own = num((result.get("environment", {}) or {}).get("pid"))
    warmup = sum(num(r.get("timing_s", {}).get("total_s")) or 0 for r in records if r.get("event") == "warmup" and isinstance(r.get("timing_s"), dict))
    if not warmup: warmup = sum(num(r.get("timing_s", {}).get("total_s")) or 0 for r in result.get("warmup_records", []) if isinstance(r, dict) and isinstance(r.get("timing_s"), dict))
    tasks = {task: task_summary(task, result, records, warmup) for task in TARGETS}
    failures = [f"{r.get('type', 'error')}：{r.get('message', '')}" for r in records if r.get("event") == "error"]
    if result_error: state = "missing" if not result else "failed"
    elif result.get("status") == "error" or failures: state = "failed"
    elif result.get("status") == "complete" and all(t["status"] == "complete" for t in tasks.values()): state = "complete"
    else: state = "incomplete"
    return {"directory": directory.name, "mode": result.get("mode", directory.name), "batch_size": result.get("batch_size"), "benchmark_status": result.get("status", "missing"), "status": state, "error": result_error, "result_error": result.get("error"), "tasks": tasks, "warmup_seconds": warmup, "measured_seconds_excluding_warmup": sum(t["total_seconds"] or 0 for t in tasks.values()), "parse_errors": progress_errors + failures, "gpu_competition": gpu_summary(records, own)}

def fmt(x, digits=3): return "—" if x is None else f"{x:.{digits}f}"

def early_summary(live):
    if not isinstance(live, dict): return {"status": "missing", "tasks": {}, "projection_at_observation": {}}
    return {"status": "available", "observed_at": live.get("observed_at"), "tasks": live.get("tasks", {}), "projection_at_observation": live.get("projection_at_observation", {}), "gpu": live.get("gpu"), "processes": live.get("processes")}

def make_report(summary):
    lines = ["# CSGO 推理速度研究汇总", "", f"生成时间：{summary['generated_at']}", "", "## 前期正式推理", ""]; early = summary["early_inference"]
    if early["status"] != "available": lines.append("live_before.json 缺失或不可解析，前期正式推理数据不可用。")
    else:
        for task, d in early.get("tasks", {}).items():
            lines.append(f"- {TARGETS.get(task, (0, task))[1]}：已完成 {d.get('count', '—')} / {d.get('total', '—')} 张；原始全量跨度 {fmt(d.get('all_span_seconds_per_image'))} 秒/张。")
            for key, label in (("last_100", "recent100"), ("last_20", "最近20张")):
                q = d.get(key, {}) or {}; lines.append(f"  - {label}：{fmt(q.get('sec_per_image'))} 秒/张；全量 {fmt(q.get('full_task_hours'))} 小时；剩余 {fmt(q.get('remaining_hours'))} 小时。")
        p = early.get("projection_at_observation", {})
        if p: lines.append(f"- 观察时点原始外推：离散全量 {fmt(p.get('discrete_full_hours'))} 小时、剩余 {fmt(p.get('discrete_remaining_hours'))} 小时；连续全量 {fmt(p.get('continuous_full_hours'))} 小时；合计剩余 {fmt(p.get('combined_remaining_hours'))} 小时（连续 count=0 时剩余按全量理解）。")
    if summary.get("early_parse_error"): lines.append(f"- live_before 读取问题：{summary['early_parse_error']}")
    lines += ["", "## 成功微基准结果", "", "| mode | batch | 任务 | count | 总秒 | 秒/张 | 平均整批秒 | 首次预热秒 | allocated 峰值 GiB | reserved 峰值 GiB | 外推小时（discrete20000/continuous12800） | 相对 eager b1 提速 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["successful_results"]: lines.append("| {mode} | {batch} | {task} | {count} | {total} | {spi} | {full} | {warm} | {alloc} | {reserved} | {hours} | {speed} |".format(**row))
    if not summary["successful_results"]: lines.append("| （当前没有完整成功结果） |  |  |  |  |  |  |  |  |  |  |  |")
    names = {"complete": "完成", "incomplete": "未完成", "failed": "失败", "missing": "缺失"}; id_names = {"consistent": "一致", "unknown": "未知", "inconsistent": "不一致"}; lines += ["", "## 模式状态与样本 ID 一致性", ""]
    for m in summary["modes"]:
        lines.append(f"- {m['directory']}（{m['mode']}、batch{m.get('batch_size') or '—'}）：{names.get(m['status'], m['status'])}；benchmark 状态 {m['benchmark_status']}。")
        if m.get("error"): lines.append(f"  - {m['error']}")
        if m.get("result_error"): lines.append(f"  - benchmark 错误：{m['result_error']}")
        if m.get("parse_errors"): lines.append(f"  - progress/失败信息：{'; '.join(m['parse_errors'])}")
        for task, t in m["tasks"].items():
            c = t["sample_id_consistency"]; lines.append(f"  - {task}：ID {id_names.get(c['status'], c['status'])}（期望 {c['expected_count']}、摘要 {c['summary_count']}、进度 {c['progress_count']}、观测 {c['observed_count']}）。")
    lines.append(f"- 跨成功模式 ID 比对：{id_names.get(summary['cross_mode_sample_id_consistency']['status'], summary['cross_mode_sample_id_consistency']['status'])}。")
    lines += ["", "## GPU 竞争快照", ""]
    for m in summary["modes"]:
        g = m["gpu_competition"]; lines.append(f"- {m['directory']}：{g['snapshot_count']} 个批次快照；GPU 利用率 {fmt(g['gpu_utilization_percent'].get('min'))}–{fmt(g['gpu_utilization_percent'].get('max'))}%；显存 {fmt(g['gpu_memory_used_mib'].get('min'))}–{fmt(g['gpu_memory_used_mib'].get('max'))} MiB。")
        if g["other_processes_by_pid"]: lines.append("  - 每批排除 benchmark 自身 PID 后的其他进程显存：" + "；".join(f"PID {p['pid']}：{fmt(p['memory_mib'].get('min'))}–{fmt(p['memory_mib'].get('max'))} MiB" for p in g["other_processes_by_pid"].values()) + "。")
        known = [f"PID {pid}（{KNOWN_EXTERNAL[pid]}）" for pid in KNOWN_EXTERNAL if str(pid) in g["other_processes_by_pid"]]
        if known: lines.append("  - 本轮已知外部任务：" + "、".join(known) + "；它们的显存或出现时点变化说明竞争不恒定。")
        for s in g["per_batch"]:
            gpu = ", ".join(f"GPU{x.get('index')} 利用率 {fmt(x.get('utilization_gpu_percent'))}%/显存 {fmt(x.get('memory_used_mib'))} MiB" for x in s["gpus"]) or "无 GPU 数据"; other = ", ".join(f"PID{x['pid']} {fmt(x['used_memory_mib'])}MiB" for x in s["other_processes"]) or "无其他进程数据"; lines.append(f"  - {s['event']} {s['task']} batch{s['batch_index']}：{gpu}；其他：{other}。")
    lines += ["", "## 口径与并行说明", "", "样本规模为 small32samples：共32张固定样本（两个任务各16张）；batch16 时每个任务仅1批。编译模式和批量模式不保证逐像素一致；VQ 按图逐张 FP32 解码。CPU threads 为1（OMP_NUM_THREADS/MKL_NUM_THREADS=1）。正式推理与其它任务并行运行，微基准还会引入额外竞争，因此这里是非独占卡性能。", "GPU snapshot 由 nvidia-smi 在计时窗口外采集，诊断命令可能花数秒，进程 walltime 因而可长于 timed_s 样本总和；外推使用计时窗口内端到端秒/张（含 JPEG、排除诊断开销）。外部进程只记录 PID/显存，快照中的竞争变化应按观测比值解释；相对 eager b1 的 speedup 是并行环境下的观测比值。", ""]
    return "\n".join(lines)

def main():
    live, live_error = read_json(OUT / "live_before.json"); modes = []; directories = sorted(p for p in OUT.iterdir() if p.is_dir()) if OUT.is_dir() else []
    for d in directories:
        bp, pp = d / "benchmark_results.json", d / "progress.jsonl"
        if not bp.is_file() and not pp.is_file(): continue
        result, error = read_json(bp); records, errors = read_progress(pp) if result is not None or pp.is_file() else ([], []); modes.append(mode_summary(d, result, error, records, errors))
    baselines = [m for m in modes if m["benchmark_status"] == "complete" and m["mode"] == "eager" and num(m.get("batch_size")) == 1]; baseline = {task: baselines[0]["tasks"][task]["seconds_per_image"] for task in TARGETS if baselines and baselines[0]["tasks"][task]["seconds_per_image"] is not None}
    successful, eligible = [], {task: [] for task in TARGETS}
    for m in modes:
        for task, t in m["tasks"].items():
            if m["benchmark_status"] != "complete" or t["status"] != "complete": continue
            speed = baseline.get(task) / t["seconds_per_image"] if task in baseline else None; successful.append({"mode": m["mode"], "batch": m.get("batch_size") or "—", "task": task, "count": t["count"], "total": fmt(t["total_seconds"]), "spi": fmt(t["seconds_per_image"]), "full": fmt(t["full_batch_seconds"]), "warm": fmt(t["warmup_seconds"]), "alloc": fmt(t["torch_allocated_peak_gib"]), "reserved": fmt(t["torch_reserved_peak_gib"]), "hours": f"{t['projected_target']}={fmt(t['projected_hours'])}", "speed": fmt(speed) if speed is not None else "—"}); eligible[task].append((m["directory"], t.get("observed_sample_ids", []), t["sample_id_consistency"]["status"]))
    cross = {"status": "insufficient_modes", "comparisons": []}
    cross_states = []
    for task, entries in eligible.items():
        if len(entries) < 2: continue
        same = all(e[1] == entries[0][1] for e in entries[1:]); known = entries[0][2] == "consistent" and all(e[2] == "consistent" for e in entries[1:]); state = "consistent" if same and known else ("inconsistent" if not same else "unknown"); cross_states.append(state); cross["comparisons"].append({"task": task, "reference": entries[0][0], "modes": [e[0] for e in entries], "same_observed_ids": same, "all_id_checks_consistent": known})
    if cross_states: cross["status"] = "inconsistent" if "inconsistent" in cross_states else ("unknown" if "unknown" in cross_states else "consistent")
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": ["live_before.json", "*/benchmark_results.json", "*/progress.jsonl"], "projection_targets": {"discrete20000": 20000, "continuous12800": 12800}, "early_inference": early_summary(live), "early_parse_error": live_error, "modes": modes, "successful_results": successful, "eager_b1_baseline_seconds_per_image": baseline, "cross_mode_sample_id_consistency": cross}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); (OUT / "REPORT.md").write_text(make_report(summary), encoding="utf-8")

if __name__ == "__main__": main()

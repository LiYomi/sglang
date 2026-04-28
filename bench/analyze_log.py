"""Extract structured switch stats from baseline log.

用法:
    python3 analyze_log.py --log /path/to/baseline_out.log
    python3 analyze_log.py --log /path/to/baseline_out.log --from-line 117
    python3 analyze_log.py --log /path/to/baseline_out.log --since '2026-04-19 15:'
    python3 analyze_log.py --log /path/to/baseline_out.log --short    # 紧凑方向名

提取 TP0 视角的:
    1. 每次 switch 的方向 + 路径（D2D / partial D2D / FALLBACK）+ TOTAL 时延
    2. D2D scatter 的 d2d/h2d bytes 和 stopped chunk
    3. 按方向分布的 p50/p99/max

filter 条件:
    - --from-line N: 只看从第 N 行起的日志（排除之前 session 的残留）
    - --since 'YYYY-MM-DD HH': 只看时间戳 >= 这个前缀的日志
"""

import argparse
import re
from collections import defaultdict
from typing import List, Optional, Tuple


RE_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_D2D = re.compile(
    r"D2D scatter-gather: d2d=([\d.]+)MB h2d=([\d.]+)MB time=([\d.]+)ms"
)
RE_STOP = re.compile(r"H2D scatter stopped by caller at chunk (\d+)/(\d+)")
RE_SWITCH = re.compile(
    r"SWITCH (.+?) -> (.+?): save=([\d.]+)ms load=([\d.]+)ms "
    r"kv=([\d.]+)ms rt\+graph=([\d.]+)ms TOTAL=([\d.]+)ms"
)
RE_FALLBACK = re.compile(r"FALLBACK: full H2D load for (\S+?),")


SHORT_NAME = {
    "Qwen2.5-0.5B-Instruct": "0.5B",
    "Q32-1": "Q32-1",
    "Q32-test": "Q32-test",
    "Yi34": "Yi34",
    "Yi-1.5-34B-Chat": "Yi34",
    "SmolLM2-360M-Instruct": "SmolLM",
    "Llama-3.2-1B-Instruct": "Llama1B",
}


def short(name: str, enabled: bool) -> str:
    if not enabled:
        return name
    return SHORT_NAME.get(name, name)


def parse_events(log_path: str, from_line: int, since_prefix: Optional[str]):
    """每条 TP0 日志匹配为一个 event，附带时间戳和行号。"""
    with open(log_path, "rb") as f:
        for idx, raw in enumerate(f, start=1):
            if idx < from_line:
                continue
            line = raw.decode("utf-8", errors="replace")
            if "TP0" not in line:
                continue
            if since_prefix and since_prefix not in line:
                continue
            ts_m = RE_TS.search(line)
            ts = ts_m.group(1) if ts_m else ""

            m = RE_D2D.search(line)
            if m:
                yield idx, ts, "d2d", (
                    float(m.group(1)), float(m.group(2)), float(m.group(3))
                )
                continue
            m = RE_STOP.search(line)
            if m:
                yield idx, ts, "stop", (int(m.group(1)), int(m.group(2)))
                continue
            m = RE_FALLBACK.search(line)
            if m:
                yield idx, ts, "fallback", m.group(1)
                continue
            m = RE_SWITCH.search(line)
            if m:
                yield idx, ts, "switch", (
                    m.group(1).strip(),
                    m.group(2).strip(),
                    float(m.group(3)), float(m.group(4)),
                    float(m.group(5)), float(m.group(6)),
                    float(m.group(7)),
                )


def build_timeline(events):
    """把 d2d/fallback/stop 归属到紧跟的 SWITCH 记录上。

    策略：扫到 SWITCH 时，把它之前未被消费的 stop / d2d / fallback 全部归到这条
    SWITCH，按时间顺序输出每次切换的完整信息。
    """
    timeline = []   # list of dict(ts, prev, tgt, path, d2d, h2d, time_ms, stop, total)
    pending_stop = None
    pending_d2d = None       # (d2d_mb, h2d_mb, scatter_ms)
    pending_fallback = False

    for idx, ts, kind, payload in events:
        if kind == "stop":
            # 只保留多 chunk 的 stop（过滤 0/1 单块 stop）
            if payload[1] > 1:
                pending_stop = payload
        elif kind == "d2d":
            pending_d2d = payload
        elif kind == "fallback":
            pending_fallback = True
        elif kind == "switch":
            prev, tgt, save, load, kv, rtg, total = payload
            if pending_d2d is not None:
                d2d_mb, h2d_mb, scatter_ms = pending_d2d
                if h2d_mb == 0:
                    path = "D2D"
                else:
                    path = "partial"
            elif pending_fallback:
                path = "FALLBACK"
                d2d_mb, h2d_mb, scatter_ms = (0.0, 0.0, 0.0)
            else:
                path = "?"
                d2d_mb, h2d_mb, scatter_ms = (0.0, 0.0, 0.0)

            timeline.append({
                "ts": ts,
                "prev": prev,
                "tgt": tgt,
                "path": path,
                "d2d": d2d_mb,
                "h2d": h2d_mb,
                "scatter_ms": scatter_ms,
                "stop": pending_stop,
                "save": save,
                "load": load,
                "kv": kv,
                "rtg": rtg,
                "total": total,
            })
            pending_stop = None
            pending_d2d = None
            pending_fallback = False

    return timeline


def summarize(log_path: str, from_line: int, since_prefix: Optional[str], use_short: bool):
    events = list(parse_events(log_path, from_line, since_prefix))
    timeline = build_timeline(events)

    # 打印逐次切换的时间线
    print(f"{'#':>3}  {'time':<19}  {'direction':<30}  {'path':<9}"
          f"  {'d2d MB':>8}  {'h2d MB':>8}  {'stop':>7}  {'TOTAL':>8}")
    for i, e in enumerate(timeline, start=1):
        direction = f"{short(e['prev'], use_short)} → {short(e['tgt'], use_short)}"
        stop_str = f"{e['stop'][0]}/{e['stop'][1]}" if e['stop'] else "-"
        # 只显示大 target 的 d2d/h2d 数字，小的省略
        if e["d2d"] + e["h2d"] > 5000:
            d2d_str = f"{e['d2d']:.0f}"
            h2d_str = f"{e['h2d']:.0f}"
        elif e["d2d"] + e["h2d"] > 0:
            d2d_str = f"{e['d2d']:.0f}"
            h2d_str = f"{e['h2d']:.0f}"
        else:
            d2d_str = "-"
            h2d_str = "-"
        print(f"{i:>3}  {e['ts']:<19}  {direction:<30}  {e['path']:<9}"
              f"  {d2d_str:>8}  {h2d_str:>8}  {stop_str:>7}  {e['total']:>7.1f}ms")

    # 统计
    path_count = defaultdict(int)
    dir_path = defaultdict(lambda: defaultdict(list))
    for e in timeline:
        path_count[e["path"]] += 1
        dir_path[(e["prev"], e["tgt"])][e["path"]].append(e["total"])

    print()
    print(f"Total switches: {len(timeline)}")
    for p in ("D2D", "partial", "FALLBACK", "?"):
        if path_count[p]:
            print(f"  {p:<9} {path_count[p]}")

    print()
    print(f"{'direction':<35}  {'n':>3}  {'D2D':>3}  {'par':>3}  {'FB':>3}"
          f"  {'p50':>7}  {'p99':>7}  {'max':>7}")
    for (prev, tgt), paths in sorted(dir_path.items()):
        direction = f"{short(prev, use_short)} → {short(tgt, use_short)}"
        all_totals = [t for lst in paths.values() for t in lst]
        all_sorted = sorted(all_totals)
        n = len(all_sorted)
        p50 = all_sorted[n // 2] if n > 0 else 0
        p99 = all_sorted[min(n - 1, int(n * 0.99))] if n > 0 else 0
        mx = all_sorted[-1] if n > 0 else 0
        n_d2d = len(paths.get("D2D", []))
        n_par = len(paths.get("partial", []))
        n_fb = len(paths.get("FALLBACK", []))
        print(f"  {direction:<33}  {n:>3}  {n_d2d:>3}  {n_par:>3}  {n_fb:>3}"
              f"  {p50:>5.1f}ms  {p99:>5.1f}ms  {mx:>5.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--from-line", type=int, default=1)
    ap.add_argument("--since", type=str, default=None,
                    help="Timestamp prefix filter, e.g. '2026-04-19 15:'")
    ap.add_argument("--short", action="store_true",
                    help="使用紧凑模型名 (Qwen2.5-0.5B-Instruct → 0.5B)")
    args = ap.parse_args()
    summarize(args.log, args.from_line, args.since, args.short)


if __name__ == "__main__":
    main()

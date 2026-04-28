"""Partial staging 扫频 bench.

逐级缩短 active decode 时间，验证 partial D2D + H2D 并行的退化曲线。

运行:
    python3 bench/partial_bench.py

前置:
    - server 在 localhost:30011 已 ready
    - Qwen2.5-0.5B-Instruct 是 launch 模型（默认 active）
    - Q32-1 已 register_model 且 status=ready

每 level 5 round，每 round:
    - 0.5B 请求 mt=mt_active   (active, 持续 decode 撑出 preload 窗口)
    - sleep 0.2s
    - Q32-1 请求 mt=800         (target, 长请求，decode 时间不是限制因素)
    - 两者 join

预期:
    mt_active ≥ 30 (decode ≥ 0.6s): preload 完整, 100% D2D, scatter ~10.6ms
    mt_active = 15 (0.3s): partial ~44-48% D2D, switch ~400ms
    mt_active = 8  (0.16s): partial ~20% D2D, switch ~780ms
    mt_active = 5  (0.1s):  partial ~15% D2D, switch ~900ms
"""

import urllib.request
import json
import time
import threading

URL = "http://localhost:30011/v1/completions"
PROMPT = (
    "Write a very long and detailed essay about the history of artificial "
    "intelligence from 1950 to today, covering all major milestones, "
    "researchers, and breakthroughs. Include many examples and anecdotes."
)


def req(model: str, mt: int) -> None:
    r = urllib.request.Request(
        URL,
        data=json.dumps(
            {"model": model, "prompt": PROMPT, "max_tokens": mt, "temperature": 0}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=300) as resp:
        json.loads(resp.read())


def main():
    # Warmup both models + prime graph caches in both directions
    print("warmup")
    req("Qwen2.5-0.5B-Instruct", 15)
    req("Q32-1", 15)
    req("Qwen2.5-0.5B-Instruct", 15)

    # 0.5B decodes ~20ms/token. mt=150 → ~3s; mt=5 → ~0.1s.
    levels = [150, 100, 60, 30, 15, 8, 5]
    for mt_active in levels:
        print(f"=== level mt_active={mt_active} (~{mt_active*20}ms active decode), 5 rounds ===")
        for rnd in range(1, 6):
            tA = threading.Thread(
                target=lambda mt=mt_active: req("Qwen2.5-0.5B-Instruct", mt)
            )
            tQ = threading.Thread(target=lambda: req("Q32-1", 800))
            tA.start()
            time.sleep(0.2)  # let active enter decode before target arrives
            tQ.start()
            tA.join()
            tQ.join()
            print(f"  R{rnd}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

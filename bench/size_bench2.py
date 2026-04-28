"""Paired switch bench across model sizes.

测两对切换，验证 D2D 时延与权重大小的关系:
    - Qwen2.5-0.5B-Instruct (0.3 GB) ↔ Q32-1 (30 GB): 100× 权重差
    - Q32-1 (30 GB) ↔ Yi34 (33 GB): 同量级

每 round `tA -> sleep(0.8) -> tB -> join`, 长请求 mt=800 保证 preload 时间窗。

前置:
    - Qwen2.5-0.5B-Instruct 是 launch 模型
    - Q32-1, Yi34 已 register_model 且 status=ready

预期 (稳态 D2D):
    0.5B↔Q32:  ~21ms TOTAL (load ~16ms)
    Q32↔Yi34:  ~23ms TOTAL (load ~18ms)
    pair 过渡首次: 偶尔 FALLBACK ~610ms (pair join 后 active idle)
"""

import urllib.request
import json
import time
import threading

URL = "http://localhost:30011/v1/completions"
PROMPT = (
    "Write a very long and detailed essay about the history of AI from 1950 "
    "to today, covering major milestones, researchers, and breakthroughs."
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
    # Warmup all three + prime graph caches
    for m in ["Qwen2.5-0.5B-Instruct", "Q32-1", "Yi34"]:
        print(f"warmup {m}")
        req(m, 15)

    pairs = [
        ("Qwen2.5-0.5B-Instruct", "Q32-1"),
        ("Q32-1", "Yi34"),
    ]
    for A, B in pairs:
        print(f"=== {A} <-> {B}, 5 rounds mt=800 ===")
        req(A, 15)  # switch to A at the start of the pair
        for rnd in range(1, 6):
            tA = threading.Thread(target=lambda: req(A, 800))
            tB = threading.Thread(target=lambda: req(B, 800))
            tA.start()
            time.sleep(0.8)
            tB.start()
            tA.join()
            tB.join()
            print(f"  R{rnd}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

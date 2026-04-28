"""Quick sanity check: 3 round concurrent 0.5B ↔ Q32-1 switch.

跑完后 grep 日志应看到:
    D2D scatter-gather: d2d=31246.7MB h2d=0.0MB time=~10.6ms    # 完整命中
    D2D scatter-gather: d2d=601.1MB h2d=0.0MB time=~2.1ms       # 反方向 0.5B
"""

import urllib.request
import json
import time
import threading

URL = "http://localhost:30011/v1/completions"
PROMPT = "Write a long essay about AI with many examples."


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
    req("Qwen2.5-0.5B-Instruct", 15)
    req("Q32-1", 15)
    req("Qwen2.5-0.5B-Instruct", 15)

    for rnd in range(3):
        tA = threading.Thread(target=lambda: req("Qwen2.5-0.5B-Instruct", 150))
        tQ = threading.Thread(target=lambda: req("Q32-1", 400))
        tA.start()
        time.sleep(0.2)
        tQ.start()
        tA.join()
        tQ.join()
        print(f"R{rnd+1}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

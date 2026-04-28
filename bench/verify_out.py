"""Correctness check across model switches.

发 6 条经典 prompt（两个模型各 3 条），确认切换后每个模型输出连贯且符合常识。

启发式标记:
    [OK ]  前 20 字符没有 ascii 控制字符 / 异常 Unicode
    [CHK]  可能乱码，需要肉眼复查

历史 bug 示例 (bump + meta cos_sin_cache 漏 materialize):
    "The capital of France is" => " the largest country in Europe..."   # 答非所问
    "What is 2+3?"              => " The answer is 6."                  # 数字错
修复见 commit `c4676772f fix(bump): materialise meta non-persistent buffers...`
"""

import urllib.request
import json

URL = "http://localhost:30011/v1/completions"


def req(model: str, prompt: str, mt: int = 30) -> str:
    r = urllib.request.Request(
        URL,
        data=json.dumps(
            {"model": model, "prompt": prompt, "max_tokens": mt, "temperature": 0}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["text"]


def main():
    tests = [
        ("Qwen2.5-0.5B-Instruct", "The capital of France is"),
        ("Q32-1", "The capital of France is"),
        ("Qwen2.5-0.5B-Instruct", "What is 2+3?"),
        ("Q32-1", "What is 2+3?"),
        ("Qwen2.5-0.5B-Instruct", "Hello, my name is"),
        ("Q32-1", "Hello, my name is"),
    ]
    for m, p in tests:
        try:
            t = req(m, p, mt=30)
            # Flag suspect chars in first 20 chars (control chars / strange mid-ascii)
            marker = (
                "OK "
                if t and not any(ord(c) > 126 and ord(c) < 0x4e00 for c in t[:20])
                else "CHK"
            )
            print(f"[{marker}] {m:30s} | {p!r} => {t!r}")
        except Exception as e:
            print(f"[ERR] {m}: {e}")


if __name__ == "__main__":
    main()

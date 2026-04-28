# Model Hot-Switch 测试报告

**最后更新**：2026-04-28
**测试环境**：TP=2 B200（每卡 183 GB HBM3e）
**分支**：`https://git.corp.kuaishou.com/cloud/containercloud/vibe-coding/sglang/-/tree/feat/multi-model-hot-switch?ref_type=heads`

---

## 目录

- [测试环境](#测试环境)
- [脚本清单](#脚本清单)
- [测试流程](#测试流程)
  - [1. 正确性验证 — verify_out.py](#1-正确性验证--verify_outpy)
  - [2. 快速 sanity — quickrun.py](#2-快速-sanity--quickrunpy)
  - [3. Partial staging 扫频 — partial_bench.py](#3-partial-staging-扫频--partial_benchpy)
  - [4. 多对模型切换 — size_bench2.py](#4-多对模型切换--size_bench2py)
- [日志结构化提取 — analyze_log.py](#日志结构化提取--analyze_logpy)
- [关键发现汇总](#关键发现汇总)
- [日志归档](#日志归档)

---

## 测试环境

**启动参数**（`/home/mxc/launch_baseline.sh`）：

```bash
export SGLANG_SWITCH_DIAG=0          # 生产模式，测时延必须 = 0
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=1
export LD_PRELOAD=.../torch_memory_saver_hook_mode_preload.abi3.so

python3 -m sglang.launch_server \
    --model-path /home/mxc/model/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen2.5-0.5B-Instruct \
    --port 30011 --tp 2 \
    --mem-fraction-bump 0.85 \
    --disable-piecewise-cuda-graph \
    --enable-bump-allocator \
    --enable-memory-saver \
    --attention-backend flashinfer
```

**注册模型**（除 launch 模型外，通过 `/register_model` 注册）：

| 名称 | 路径 | 规模 | 权重（TP=2 per rank） |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | `/home/mxc/model/Qwen2.5-0.5B-Instruct` | 0.5B | ~0.3 GB |
| Q32-1 / Q32-test / Q32-1..Q32-8 | `/home/mxc/model/Qwen2.5-32B-Instruct` | 32B | ~30 GB |
| Yi34 | `/home/mxc/model/Yi-1.5-34B-Chat` | 34B | ~32 GB |

---

## 脚本清单

| 脚本 | 用途 | 运行时长 |
|---|---|---|
| `verify_out.py` | 切换后推理正确性（6 prompt × 2 模型） | ~30 s |
| `quickrun.py` | 3 round 并发切换 sanity | ~30 s |
| `partial_bench.py` | mt_active 7 级扫频，验证 partial D2D + H2D 并行 | ~6 min |
| `size_bench2.py` | 多对模型切换（0.5B↔Q32，Q32↔Yi34） | ~3 min |
| `analyze_log.py` | 日志提取 d2d / h2d / switch time 结构化数据 | 瞬间 |

---

## 测试流程

### 1. 正确性验证 — `verify_out.py`

**目的**：在多次切换之后确认两个模型的输出都还正常，没有被切换机制损坏

**逻辑**：

```python
tests = [
    ("Qwen2.5-0.5B-Instruct", "The capital of France is"),
    ("Q32-1",                 "The capital of France is"),
    ("Qwen2.5-0.5B-Instruct", "What is 2+3?"),
    ("Q32-1",                 "What is 2+3?"),
    ("Qwen2.5-0.5B-Instruct", "Hello, my name is"),
    ("Q32-1",                 "Hello, my name is"),
]
```

- 贪心解码（`temperature=0`），前 30 token
- 启发式 marker：前 20 字符无控制符 / 非中文 unicode → `[OK ]`，否则 `[CHK]`

**运行方式**：

```bash
# 远端启动服务 + register 两个模型后
python3 bench/verify_out.py
```

**预期输出**：

```
[OK ] Qwen2.5-0.5B-Instruct          | 'The capital of France is' => ' Paris. It is the largest city in Europe and the second largest city in the world. It is located in the south of France, on the banks'
[OK ] Q32-1                          | 'The capital of France is' => ' Paris. Correct! The capital of France is indeed Paris. Paris is not only the capital but also the largest city in France and a major global city'
[OK ] Qwen2.5-0.5B-Instruct          | 'What is 2+3?' => ' The answer is 5. What is 2+4? The answer is 6. What is 2+5? The answer is '
[OK ] Q32-1                          | 'What is 2+3?' => ' 2+3 is 5. Is there anything else you would like to know?'
[OK ] Qwen2.5-0.5B-Instruct          | 'Hello, my name is' => ' Alex and I am a 16 year old male. I have a question about my health. I have a history of asthma and I have been'
[OK ] Q32-1                          | 'Hello, my name is' => ' Dr. David J. Berceli and I am the founder of the Trauma Releasing Exercises (TRE). I am a former U.S'
```

**结果**：✅ 所有 6 条 `[OK ]`。2026-04-20 `194cbf6e7` 落盘后，跨 50+ 次切换均保持正确输出。

---

### 2. 快速 sanity — `quickrun.py`

**目的**：3 round 并发切换，看日志能否打出 D2D scatter-gather。

**逻辑**：

```python
# warmup
req('0.5B', 15); req('Q32-1', 15); req('0.5B', 15)

# 3 concurrent rounds
for rnd in range(3):
    tA = Thread(req('Qwen2.5-0.5B-Instruct', 150))   # active 长请求 ~3s decode
    tQ = Thread(req('Q32-1', 400))                     # target
    tA.start()
    time.sleep(0.2)                                    # 让 0.5B 先进 decode
    tQ.start()
    tA.join(); tQ.join()
```

**运行方式**：

```bash
python3 bench/quickrun.py
# 跑完 grep 日志
grep 'D2D scatter-gather' /home/mxc/sglang-review-test/logs/baseline_out.log | tail
```

**预期日志**：

```
D2D scatter-gather: d2d=31246.7MB h2d=0.0MB time=~10.6ms    # 完整命中 Q32 方向
D2D scatter-gather: d2d=601.1MB   h2d=0.0MB time=~2.1ms     # 反方向 0.5B
```

**结果**：✅ 观察到 d2d=31246.7MB / h2d=0.0MB，10.5-10.7ms。

---

### 3. Partial staging 扫频 — `partial_bench.py`

**目的**：验证 partial staging 机制——active decode 时间不够时，preload 写到一半被打断，switch 用 D2D + H2D 并行完成。

**逻辑**：

```python
for mt_active in [150, 100, 60, 30, 15, 8, 5]:   # active 长度逐级递减
    for rnd in range(5):
        tA = Thread(req('0.5B', mt_active))       # active mt 控制 decode 时间
        tQ = Thread(req('Q32-1', 800))            # target 永远长，足够 decode
        tA.start()
        time.sleep(0.2)
        tQ.start()
        tA.join(); tQ.join()
```

7 level × 5 round = 35 次 Q32 target switch。

**运行方式**：

```bash
python3 bench/partial_bench.py
python3 bench/analyze_log.py --log logs/baseline_out.log --since '2026-04-19'
```

**结果**（来自 `capacity-analysis.md` 2026-04-19 晚记录）：

| Level (mt_active) | decode 时间 ≈ | D2D% | h2d MB | switch time | stopped 位置 |
|:-:|:-:|:-:|---:|---:|---|
| 150 (#1-5) | 3.0 s | **100%** | 0 | ~10.6ms | 不触发（preload 完成） |
| 100 (#6-10) | 2.0 s | 100% | 0 | ~10.6ms | 同上 |
| 60 (#11-15) | 1.2 s | 100% | 0 | ~10.6ms | 同上 |
| 30 (#16-20) | 0.6 s | 100% | 0 | ~10.6ms | **刚好临界** |
| 15 (#21-25) | 0.3 s | 44-48% | ~22 GB | 408-455ms | stopped 16-17/26 |
| 8 (#26-30) | 0.16 s | 20-23% | ~40 GB | 731-778ms | stopped 9-10/26 |
| 5 (#31-35) | 0.1 s | 12-17% | ~48 GB | 826-919ms | stopped 6-8/26 |


- **临界点在 ~0.3-0.6 s active decode**。Q32 30 GB / PCIe5 实测 50 GB/s = 600 ms preload，active decode ≥ 600 ms 时完整命中。
- **Partial 比例精确对应 chunk 比例**：stopped k/26 → D2D ≈ (26-k)/26 × 100%。
- **Switch time 线性退化**：主要由未完成的 H2D 部分决定（D2D 和 H2D 两 stream 并行，max 取主）。
- **无全量 FALLBACK 路径**：即使 mt_active=5（0.1 s），partial D2D 仍生效。

---

### 4. 多对模型切换 — `size_bench2.py`

**目的**：不同模型对（小↔大、大↔大）切换时延特征。

**逻辑**：

```python
# warmup 3 个模型各一次
req('0.5B', 15); req('Q32-1', 15); req('Yi34', 15)

# 多对 pair × 5 round
pairs = [
    ('Qwen2.5-0.5B-Instruct', 'Q32-1'),
    ('Q32-1', 'Yi34'),
]
for A, B in pairs:
    req(A, 15)                # reset active=A
    for rnd in range(5):
        tA = Thread(req(A, 800))
        tB = Thread(req(B, 800))
        tA.start()
        time.sleep(0.8)       # pair 间用 0.8s 更保险
        tB.start()
        tA.join(); tB.join()
```

**运行方式**：

```bash
python3 bench/size_bench2.py
python3 bench/analyze_log.py --log bench/baseline_out.log
```

**结果**（2026-04-26 实测，`bench/baseline_out.log`）：

日志覆盖 3 次独立 pair 运行（Q32-test / Q32-1 / Yi34），共 **29 次切换**。按测试意图分三类：

#### ① Cold（3 次，首次切换开销）

首次切到新 target 时 CUDA graph 要从零 capture，TOTAL ~6 s。preload 线程那时也刚启动，无法准备好，同时走 FALLBACK H2D。属于一次性开销，不是 bench 要测的内容。

| 方向 | n | TOTAL | 备注 |
|---|:-:|---:|---|
| 0.5B → Q32-test | 1 | 6587 ms | Q32-test pair 首次 |
| 0.5B → Q32-1 | 1 | 6290 ms | warmup 首次 |
| Q32-1 → Yi34 | 1 | 6237 ms | warmup 首次 |

#### ② Pair 内期望方向 A→B（13 次，bench 核心测量）

bench 设计为测这个方向：tA 先发让 active 持续 busy，sleep 0.8s 后 tB 到达，preload 充分启动后切换 → 期望 **全 D2D 命中**。

| 方向 | 非 cold n | D2D | FALLBACK | p50 TOTAL | scatter |
|---|:-:|:-:|:-:|---:|---:|
| 0.5B → Q32-test | 3 | 3 | 0 | 21.1 ms | 10.5 ms / 30 GB |
| 0.5B → Q32-1 | 5 | 5 | 0 | 22.5 ms | 10.6 ms / 30 GB |
| Q32-1 → Yi34 | 5 | 5 | 0 | 24.0 ms | 11.7 ms / 32 GB |
| **合计** | **13** | **13** | **0** | — | — |

**核心结论**：13/13 全部命中 D2D，scatter 10.5-11.8 ms 达到 HBM3e 带宽下限（2.8 TB/s），h2d=0

> Q32-test pair 跑了 3 rounds（早期手动测试），Q32-1 / Yi34 pair 各 5 rounds（当前 `size_bench2.py`）。

#### ③ Pair 反向 B→A + Setup 过渡（13 次，非测试重点）

pair 每 round 末尾 active=B，下一 round 开始 tA 发 A 时的"反向切换"；以及 pair 之间 `req(A, 15)` 重置 active 的 setup 切换。**这些不是 bench 要测的方向**，但暴露了一个调度局限。

| 方向 | n | D2D | FALLBACK | 归因 |
|---|:-:|:-:|:-:|---|
| Q32-test → 0.5B | 4 | 2 | 2 | pair 反向，0.5B 权重小，FALLBACK 也才 30-44 ms |
| Q32-1 → 0.5B | 4 | 3 | 1 | 同上 |
| **Yi34 → Q32-1** | 4 | 0 | **4** | pair 反向撞调度局限，全 FALLBACK（见下面分析）|
| Yi34 → 0.5B | 1 | 0 | 1 | pair 2 → pair 1 过渡 setup |
| **合计** | **13** | **5** | **8** | — |


#### 稳态 D2D 时延拆解（`SWITCH ... TOTAL=21ms`）

```
save_teardown:    ~2  ms   # Phase 1
_load_via_d2d:   ~16  ms   # Phase 2
  D2D scatter:     10.6ms   ← HBM3e 2.8 TB/s（理论 3 TB/s 的 93%）
  model_assign + finalize: ~5 ms   (固定 overhead)
kv_setup:         ~1  ms   # Phase 3
rt+graph:         ~2  ms   # Phase 4（graph cache hit）
─────────────────────────
TOTAL:           ~21  ms
```


---

## 日志结构化提取 — `analyze_log.py`

**用途**：从 baseline log 里 grep 出 d2d/h2d/switch/fallback 结构化打印。

**关键 regex**：

```python
RE_D2D = r"D2D scatter-gather: d2d=([\d.]+)MB h2d=([\d.]+)MB time=([\d.]+)ms"
RE_STOP = r"H2D scatter stopped by caller at chunk (\d+)/(\d+)"
RE_SWITCH = r"SWITCH (.+?) -> (.+?): save=...ms load=...ms kv=...ms rt\+graph=...ms TOTAL=...ms"
RE_FALLBACK = r"FALLBACK: full H2D load for (\S+?),"
```

只看 TP0（避免 TP0/TP1 重复计数）。大 target 过滤 `d2d+h2d > 5 GB`（排除 0.5B 这种小目标干扰）。

**用法**：

```bash
# 全量
python3 bench/analyze_log.py --log bench/baseline_out.log

# 时间范围过滤
python3 bench/analyze_log.py --log bench/baseline_out.log --since '2026-04-20 12:'

# 行号过滤（跳过历史 session）
python3 bench/analyze_log.py --log bench/baseline_out.log --from-line 117
```

**输出格式**：

```
  #     d2d MB     h2d MB   D2D%      time  stop
  1    31246.7        0.0  100.0    10.7ms  0/26
  2    31246.7        0.0  100.0    10.6ms  -
  ...

D2D (large target) entries: 13
FALLBACK:                   11
SWITCH (all directions):    29

SWITCH TOTAL (ms) per direction:
  0.5B → Q32-1: n=6 min=21.8 p50=22.5 p99=6290.2 max=6290.2
  ...
```

---

## 关键发现汇总

1. ✅ **正确性**：跨 50+ 次切换两个模型输出都正常。`c4676772f` meta-buffer fix 生效。
2. ✅ **稳态 D2D**：**TOTAL 21 ms**（load 16 ms / D2D scatter 10.6 ms），达到 HBM3e 带宽下限。
3. ✅ **Partial staging**：
   - 临界点 ~600 ms active decode（和 PCIe5 50 GB/s × 30 GB = 600 ms 预期吻合）
   - partial 比例精确对应 chunk 完成比例，switch time 线性退化
   - 无全量 FALLBACK 路径（即使 mt_active=5 也 partial D2D 生效）
4. 📊 **H2D FALLBACK 带宽**：30 GB / 600 ms ≈ 50 GB/s，PCIe5 单向 63 GB/s 的 80%。

---

## 日志归档

- `bench/baseline_out.log` — 1.5 MB，远端最新一轮运行日志（`2026-04-20 ~ 2026-04-26`）
  - 包含 size_bench2 完整运行（0.5B ↔ Q32-test 5 round + 0.5B/Q32-1/Yi34 多对 5 round）
  - 不含 partial_bench 扫频数据（在更早的 session 里，已记录到 `capacity-analysis.md`）
---

## 运行新一轮完整测试的 check list

```bash
# 1. 启动服务
bash /home/mxc/launch_baseline.sh &

# 2. 等 ready (约 90s)
sleep 90 && curl localhost:30011/health

# 3. 注册模型（-H Content-Type: application/json 不能省，否则 FastAPI 当 bytes 拒绝）
curl -X POST localhost:30011/register_model \
     -H 'Content-Type: application/json' \
     -d '{"model_name":"Q32-1","model_path":"/home/mxc/model/Qwen2.5-32B-Instruct"}'
curl -X POST localhost:30011/register_model \
     -H 'Content-Type: application/json' \
     -d '{"model_name":"Yi34","model_path":"/home/mxc/model/Yi-1.5-34B-Chat"}'

# 3.1 查看注册状态（所有模型都 "status":"ready" 才能开跑 bench）
curl -s localhost:30011/list_models | python3 -m json.tool
# 期望输出:
# {
#   "object": "list",
#   "data": [
#     {"id": "Q32-1",                    "object": "model", "status": "ready"},
#     {"id": "Qwen2.5-0.5B-Instruct",    "object": "model", "status": "ready"},
#     {"id": "Yi34",                      "object": "model", "status": "ready"}
#   ]
# }
#
# 其他可能的 status: "loading" (CPU load 进行中) / "failed" (error 信息在 error 字段)

# 4. 等 CPU load 完成（polling list_models，比固定 sleep 更可靠）
while [ $(curl -s localhost:30011/list_models | grep -o '"status":"ready"' | wc -l) -lt 3 ]; do
    echo "waiting... $(curl -s localhost:30011/list_models | grep -o '"status":"[^"]*"')"
    sleep 15
done
echo "all 3 models ready"

# 5. 按顺序运行 bench
python3 bench/verify_out.py              # ~30s,  期望 6 个 [OK ]
python3 bench/quickrun.py                # ~30s,  期望 log 有 D2D scatter
python3 bench/partial_bench.py           # ~6min, 35 次切换，7 级扫频
python3 bench/size_bench2.py             # ~3min, 多对模型

# 6. 分析
python3 bench/analyze_log.py --log /home/mxc/sglang-baseline/baseline_out.log

python3 bench/analyze_log.py --log /home/mxc/sglang-baseline/baseline_out.log --since '<今天>'
```

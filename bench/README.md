# Model Hot-Switch Bench

一套完整的测试资产：启动、注册、切换正确性、H2D / D2D / Partial 性能扫频。

对应分支：`feat/multi-model-hot-switch`（远端 `/home/mxc/sglang-baseline`）
对应 commits：

```
3043b588f  feat(partial-staging): aligned chunk layout + incremental chunk_valid publish
21223d4a8  feat(hot-switch): eager preload next target + raise on preload errors
c4676772f  fix(bump): materialise meta non-persistent buffers and recompute RoPE cache
484378144  feat(hot-switch): multi-model hot-switch with bump allocator + review fixes
```

---

## 目录

```
bench/
├── README.md                 # 本文件
├── partial_bench.py          # Partial staging 7 级扫频
├── size_bench2.py            # 多对模型（0.5B↔32B, 32B↔34B）切换
├── verify_out.py             # 切换后输出正确性检查
├── quickrun.py               # 3 round 快速 sanity check
└── analyze_log.py            # 日志里 d2d/h2d/switch time 结构化提取
```

所有脚本假定 server 在 `localhost:30011`，TP=2。远端执行通过 `python3 tools/claude-proxy/remote-run.py "..."` 代理；本地可以直接 `python3 bench/xxx.py`。

---

## 完整流程

### 1. 启动 server（远端）

```bash
# 启动脚本 /home/mxc/launch_baseline.sh 的关键环境：
export SGLANG_SWITCH_DIAG=0          # 0 = 生产 / 测时延；bounds 或 full 会污染 GPU context
export PATH=/home/mxc/sglang-env/bin:$PATH
export PYTHONSAFEPATH=1
export PYTHONPATH=/home/mxc/sglang-baseline/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=1
export LD_PRELOAD=/home/mxc/sglang-env/lib/python3.12/site-packages/torch_memory_saver_hook_mode_preload.abi3.so
export CUDA_VISIBLE_DEVICES=0,1

/home/mxc/sglang-env/bin/python3 -m sglang.launch_server \
    --model-path /home/mxc/model/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen2.5-0.5B-Instruct \
    --port 30011 --tp 2 \
    --mem-fraction-bump 0.85 \
    --disable-piecewise-cuda-graph \
    --enable-bump-allocator \
    --enable-memory-saver \
    --attention-backend flashinfer \
    > /home/mxc/sglang-review-test/logs/baseline_out.log 2>&1
```

**server args 不要动**：`--tp 2` / `--mem-fraction-bump 0.85` / `--enable-bump-allocator` / `--enable-memory-saver` / `--attention-backend flashinfer` / `--disable-piecewise-cuda-graph` 是 hot-switch 的必要前提。

**启动后**：server 会跑一次 warmup forward + CUDA graph capture，约 60-90s 才能接请求。

```bash
# 后台启动
python3 tools/claude-proxy/remote-run.py "nohup bash /home/mxc/launch_baseline.sh > /tmp/baseline_stdout.log 2>&1 &"

# 轮询等 ready
python3 tools/claude-proxy/remote-run.py "for i in 1 2 3 4 5 6 7 8; do sleep 15; curl -s --max-time 3 localhost:30011/health >/dev/null 2>&1 && echo ready && break; done"
```

### 2. 注册额外模型

`register_model` 返回 success 只代表**开始后台 CPU load**，不代表 load 完成。大模型 CPU pinned 加载需要 30s-120s，必须轮询 `list_models`：

```bash
# 注册 Qwen2.5-32B-Instruct（alias Q32-1）
curl -s -X POST localhost:30011/register_model -H 'Content-Type: application/json' \
    -d '{"model_name":"Q32-1","model_path":"/home/mxc/model/Qwen2.5-32B-Instruct"}'

# 注册 Yi-1.5-34B-Chat
curl -s -X POST localhost:30011/register_model -H 'Content-Type: application/json' \
    -d '{"model_name":"Yi34","model_path":"/home/mxc/model/Yi-1.5-34B-Chat"}'

# 轮询直到所有 status=ready
while true; do
    ready=$(curl -s localhost:30011/list_models | grep -o '"status":"ready"' | wc -l)
    total=$(curl -s localhost:30011/list_models | grep -o '"id":' | wc -l)
    echo "$ready/$total ready"
    [ "$ready" = "$total" ] && break
    sleep 15
done
```

**可用模型清单**（TP=2 验证通过）：

| model_name | path | 权重 (TP=2 每 rank) |
|---|---|---|
| `Qwen2.5-0.5B-Instruct` | `/home/mxc/model/Qwen2.5-0.5B-Instruct` | 0.3 GB |
| `Q32-1` ~ `Q32-8` | `/home/mxc/model/Qwen2.5-32B-Instruct` | 30 GB |
| `Yi34` | `/home/mxc/model/Yi-1.5-34B-Chat` | 33 GB |

**不要用**：`SmolLM2-360M-Instruct` / `Llama-3.2-1B-Instruct`——TP=2 下 `total_num_heads % tp_size != 0` 或 CPU load 失败，会让 scheduler 崩溃。

### 3. 正确性验证（必做第一步）

测试任何性能前先确认切换后模型输出正常，避免性能数字对应错误结果：

```bash
python3 bench/verify_out.py
```

期望输出：每个 `[OK ]` 开头，prompt 回答合理（"The capital of France is" → " Paris..."）。
如果出现 `[CHK]` 或明显答非所问（"The answer is 6."）—— 切换 bug，停下排查，不要继续测时延。

### 4. 快速 sanity check

确认 preload + D2D 机制生效：

```bash
python3 bench/quickrun.py
```

跑 3 round `0.5B ↔ Q32` 并发切换（每 round `0.5B` 先发 150 tokens，sleep 0.2s 后 `Q32` 发 400 tokens）。
随后检查日志：

```bash
grep -aE 'D2D scatter-gather' /home/mxc/sglang-review-test/logs/baseline_out.log | grep TP0 | tail -8
```

期望看到：
```
D2D scatter-gather: d2d=31246.7MB h2d=0.0MB time=10.6ms       # 完整命中
D2D scatter-gather: d2d=601.1MB h2d=0.0MB time=2.1ms          # 0.5B 反方向小权重
```

### 5. Partial staging 扫频（性能核心实验）

验证"active decode 时间 → preload 完成度 → D2D% → switch 时延"的关系：

```bash
python3 bench/partial_bench.py > /tmp/partial_bench.out 2>&1
```

7 级 `mt_active` 扫频（150/100/60/30/15/8/5 tokens），每级 5 round，每 round `0.5B` 先发 → sleep 0.2s → `Q32` 发 mt=800 长请求。

预期：
- `mt_active ≥ 30` (decode ≥ 0.6s)：preload 完整，**100% D2D**，switch scatter **10.6ms**
- `mt_active = 15` (0.3s)：partial，D2D **44-48%**，switch **408-455ms**
- `mt_active = 8`：partial，D2D **20-23%**，switch **731-778ms**
- `mt_active = 5`：partial，D2D **12-17%**，switch **826-919ms**

扫频结束后用 `analyze_log.py` 提取结构化数据：

```bash
python3 bench/analyze_log.py \
    --log /home/mxc/sglang-review-test/logs/baseline_out.log \
    --from-line <bench 开始前的行数>
```

### 6. 多对模型切换（大小模型覆盖）

```bash
python3 bench/size_bench2.py > /tmp/size_bench.out 2>&1
```

测两对：
- `Qwen2.5-0.5B-Instruct ↔ Q32-1`：0.3 GB ↔ 30 GB（100× 权重差）
- `Q32-1 ↔ Yi34`：30 GB ↔ 33 GB（同量级）

每对 5 round `tA sleep(0.8) tB join` 节奏。

**预期**：
- 所有 **稳态 D2D** 都 ~21ms TOTAL（与权重大小无关，scatter 本身 10-18ms + 固定 overhead ~5-10ms）
- **pair 过渡**的首次 switch 可能 FALLBACK（~600ms），因为 pair join 后 active idle 再切

### 7. H2D FALLBACK 路径（降级测量）

无需专门脚本。短请求下自然发生：

```python
# 发请求即切，active 没时间 decode 撑出 preload 窗口
req('Qwen2.5-0.5B-Instruct', mt=5)
req('Q32-1', mt=5)   # → FALLBACK 全量 H2D ~605ms
```

或者在 server log 里找 `FALLBACK: full H2D load for Q32-1` 这行，对应 switch 时延应为 ~610ms（30 GB / ~50 GB/s PCIe5 = 600ms + overhead）。

---

## DIAG 模式开关

```bash
# 生产 / 测时延（必须）
export SGLANG_SWITCH_DIAG=0

# 测 fresh 模型净增显存（只在 pre/post 打探针）
export SGLANG_SWITCH_DIAG=bounds

# 完整 phase-by-phase 探针（debug 用，污染 GPU context）
export SGLANG_SWITCH_DIAG=full
```

**报时延必须 `DIAG=0`**。`bounds` 探针的 `gc.collect()` + `torch.cuda.synchronize()` + `memory_stats()` 干扰 GPU 上下文，D2D scatter 从 10.6ms 抖到 50-80ms。

---

## 显存口径

三层不同的数字，不能混用：

```
driver_used (nvidia-smi)           # capacity planning 用这个
 ≥ pytorch_reserved (allocator pool)
   ≥ pytorch_allocated (tensor 在用)
```

差额 = pytorch 仓库空 block + tms pause 物理页（pytorch 看不到）。

同路径 alias 测试（`Q32-1`..`Q32-8`）下 `driver_used` **不涨**（weights region 复用 + CUDA graph pool 复用），但 `pytorch_allocated` 口径会低估 ~1 GB。

---

## 调度时机（背景知识）

切换路径的三个触发点（`fdf19d416` + `194cbf6e7` 改动后）：

1. **请求入队**：`_add_request_to_queue` → `_prepare_model_switch` → `_switch_queue.append(target)`
2. **每 event-loop tick**：先 `_check_preload()`（提前 prime）→ `_execute_pending_switch()`（满足条件才切）
3. **switch 完成 popleft 后**：finally 里立即 `_check_preload()` 给下一个 target 启 preload

`_execute_pending_switch` 的 4 个触发条件都满足才切：
- `_switch_queue` 非空
- `running_batch` 空（active 当前不在 decode）
- `last_batch` 空
- `waiting_queue` 空（drain 完）

**串行请求**（等 A 返回再发 B）的场景：B 到达时 active 必 idle → 同一 tick 内切 B → preload 没时间 stage → FALLBACK。要想串行也 D2D，只能改 scheduler 让 switch **延迟等 preload**（本 session 没做，代价首请求 TTFT 多 ~600ms）。

---

## 实测基线（baseline v2026-04-20）

### 完整 preload 命中（long request / active busy）

```
D2D scatter-gather: d2d=31246.7MB h2d=0.0MB time=10.6ms     # scatter only
SWITCH Qwen2.5-0.5B-Instruct -> Q32-1:
    save=2.0ms  load=16.0ms  kv=0.8ms  rt+graph=1.8ms  TOTAL=20.6ms
```

- D2D scatter 10.6ms ≈ 30 GB / 2.8 TB/s（HBM3e 理论 3 TB/s 的 93%）
- load 16ms = scatter 10.6ms + model_assign/finalize/overhead 5.4ms
- TOTAL 20.6ms = 完整切换（save + load + kv + rt+graph + parked reqs flush）

### Partial 扫频曲线

| mt_active | decode ≈ | D2D% | h2d MB | switch time | stopped chunk |
|:-:|:-:|:-:|---:|---:|---:|
| 150 / 100 / 60 / 30 | 3.0 / 2.0 / 1.2 / 0.6 s | **100%** | 0 | 10.6ms (scatter) | n/a |
| 15 | 0.3s | 44-48% | ~22 GB | 408-455ms | 16-17 / 26 |
| 8 | 0.16s | 20-23% | ~40 GB | 731-778ms | 9-10 / 26 |
| 5 | 0.1s | 12-17% | ~48 GB | 826-919ms | 6-8 / 26 |

**临界点**：Q32 preload 需 ~600ms（30 GB / 50 GB/s PCIe5 实测）。active decode ≥ 600ms 才能完整命中。

### PCIe5 带宽核对

| 方向 | 数据量 | 时间 | 带宽 | 备注 |
|---|---|---|---|---|
| D2D scatter (HBM 内部) | 30 GB | 10.6ms | **2.8 TB/s** | HBM3e 理论 3 TB/s 的 93% |
| H2D FALLBACK (PCIe5) | 30 GB | ~602ms | **~50 GB/s** | PCIe5 x16 单向 63 GB/s 的 ~80% |

---

## 历史踩坑（避免重复）

1. **SmolLM2 / Llama-3.2-1B 注册成功但切换崩溃**：TP=2 + model config 不兼容 sglang bump 初始化，scheduler 会在第一次切到这些模型时 `RuntimeError: CPU load failed`。**只用 Qwen2.5-0.5B-Instruct / Qwen2.5-32B-Instruct / Yi-1.5-34B-Chat**。

2. **bench 每 round `join()` 后下 round 立即 `start()`**：pair 切换时 round 的第一个 switch 常 FALLBACK。因为 pair join 后 active 刚 idle，下 round 的 target 请求到达时没有并发 active decode。消除方法：用不 join 的持续流（本文没做）。

3. **DIAG=bounds 测时延**：探针污染 GPU context 让 D2D scatter 从 10.6ms 抖到 50-80ms。**所有时延数字必须 DIAG=0 下采集**。

4. **pytorch allocated vs driver_used**：同 path alias 测显存增量时，pytorch 口径看起来涨 ~1 GB，driver 口径不涨。capacity planning 必须用 driver (`nvidia-smi`) 口径。

5. **没 warmup 就测冷切换**：首次切到新模型会触发 CUDA graph capture（~6s），看起来像 switch 时延爆炸。每个 bench 前 `req(m, 15)` warmup 所有模型一遍。

6. **ci_arr off-by-one 导致 847 MB 固定 h2d floor**：早期 `chunk_valid[ci_arr] = True` 用 floor 除法漏标 `chunk_valid[num_chunks-1]`，每 block 最后 22 MB × 38 blocks = 847 MB 永远走 h2d。`194cbf6e7` 改 aligned layout（iter 0 写 leftover rows）修复。现在 h2d=0 MB 在完整 preload 下。

---

## 文件索引

| 文件 | 用途 | 典型运行时间 |
|---|---|---|
| `verify_out.py` | 6 条 prompt × 2 模型输出正确性 | ~20s |
| `quickrun.py` | 3 round 快速切换 sanity | ~30s |
| `partial_bench.py` | 7 level × 5 round partial 扫频 | ~5 min |
| `size_bench2.py` | 2 pair × 5 round 多大小切换 | ~3 min |
| `analyze_log.py` | 从日志提取 D2D/H2D/SWITCH 统计 | 瞬时 |

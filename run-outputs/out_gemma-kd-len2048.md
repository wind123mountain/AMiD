# Run report: `gemma-kd-len2048`

- **Run id**: `gemma-kd-len2048`
- **Server**: HGX47 (`vt-admin`, 8× H200, bare metal, driver 595.71.05)
- **Run root**: `/nvme/annp36-home/work/AMiD`
- **Run input**: [run-inputs/gemma-kd-len2048.md](../run-inputs/gemma-kd-len2048.md)
- **Source revision**: `e59f9ab` (`wind123mountain/AMiD`, branch `nnm2`). The server copy came by rsync, and the sha256 of every run file matched that commit before launch. The server's own `.git` is an older checkout, so `driver.log` prints `rev=4b6d2c0+dirty`.
- **Status**: completed. Both drivers and every phase exited 0. The eval log ends with "Eval Done! (failed models: 0)".
  - Gates:
    - `total_iters` = 2,492 and last step directory `2492/` in all 3 phases: passed.
    - B_global = 64 in every phase: passed.
    - Data and processed-data sha256 equal the manifest: passed.
    - Loss finite throughout (0 nan/inf lines in any `log.txt`): passed.
    - 3 labels with `DONE`, no `FAILED`, 6 `results_*.json` each: passed.
    - MBPP samples without `▁`: passed (0/500 for every model).
    - Dev rougeL at epoch 2 ≥ epoch 0: **failed for `gemma/csd`** (0.78 → 0.30); passed for AMiD and NNM. The plan treats this check as "investigate, do not abort". See Insights.
- **Timezone**: all times are Vietnam time (UTC+7), converted from the server's UTC logs.

| step | start (2026-10-04) | end | elapsed | exit |
|---|---|---|---|---|
| `gemma/csd` | 11:54:45 | 13:28:12 | 1 h 33 min 27 s | 0 |
| `gemma/amid` | 13:28:12 | 15:10:25 | 1 h 42 min 13 s | 0 |
| `gemma/nnm` | 15:10:25 | 17:01:39 | 1 h 51 min 14 s | 0 |
| lm-eval (3 models × 6 tasks) | 17:01:39 | 17:59:35 | 57 min 56 s | 0 |
| **whole chain** (launch at 11:54:42) | 11:54:42 | 17:59:35 | **6 h 04 min 53 s** | 0 |

lm-eval per model: CSD 19 min 19 s, AMiD 19 min 16 s, NNM 19 min 20 s.

## Settings

No paper reference exists for this teacher/student pair on UltraInteract, so this report has no comparison with a paper. The reference is the earlier run `amid-kd-8xh200` ([report](out_amid-kd-8xh200.md)).

- **Data**: `VoCuc/UltraInteract-Infer@3c2fb0d397d3ef2d505f3fe48ad4e30f30733895`, `data/dpo/google/gemma-2-9b-it/generated_train.jsonl` (79,751 records), tokenised with the gemma-2-9b-it chat template. Dev: 200 records sampled from the training records (`random.seed(42)`). The same files and hashes as `amid-kd-8xh200`.
- **Models**: student `google/gemma-2-2b-it@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, teacher `google/gemma-2-9b-it@11c9b309abf73637e4b6f9a3fa1e92e615547819` (fp16).
- **Phases** (LoRA r16 / α32 / dropout 0.05 in all):

  | phase | trainer | `--type` | method parameters | B_global = GPU × per device × grad_acc |
  |---|---|---|---|---|
  | `gemma/csd` | `base_finetune.py` | `adaptive-csd` | `ab`/`pr`, α=0.5, λ=0.5 | 8 × 4 × 2 = 64 |
  | `gemma/amid` | `base_finetune.py` | `adaptive-amid` | `ab`/`pr`, α=0.5, λ=0.5 | 8 × 4 × 2 = 64 |
  | `gemma/nnm` | `finetune.py` | `adaptive-sfkl` | skew α=0.05, NNM ratio 0.2, K=128, 4 layers, d'=256, `--delta-threshold 0.03`, replay `decreasing`, mixed α 0.5 | 8 × 2 × 4 = 64 |

- **Common settings**: lr 1e-4, cosine, no warmup, weight decay 1e-2, clip 1.0, `--kd-ratio 1.0`, seed 10; `--max-length 2048`, `--max-prompt-length 512`; 2 epochs × 1,246 = 2,492 optimizer steps; student generation with adaptive threshold from 0.0, `--loss-eps 0.1`, replay capacity 1000 per rank; DeepSpeed bf16 (`ds_config_zero0_bf16.json`); sampling top-k 0, top-p 1.0, temperature 1.0.
- **Validation protocol**:
  - Dev: before training and after each epoch on the 200 dev records (avg_loss, exact_match, rougeL).
  - lm-eval: `scripts/eval/eval_2.sh` (sha256 `3f94c465d393b63c99d0d42d6f96fdfa336956279b83ac2e05d1227413ff1c7f`), lm-eval 0.4.12 on vLLM 0.17.1, `data_parallel_size=8`, base `google/gemma-2-2b-it@299a856` + LoRA (`max_lora_rank=32`), final checkpoint `2492/` only. GSM8K and GSM-Plus (chat template, multi-turn few-shot), MATH `minerva_math` 4-shot, MMLU-STEM 5-shot, SciQ, MBPP 3-shot without chat template at temperature 0; up to 5,120 new tokens; `spaces_between_special_tokens=True` for every task.

**Differences from `amid-kd-8xh200`** (all planned; none were made during the run):

| setting | `amid-kd-8xh200` | this run |
|---|---|---|
| `--max-length` | 1025 | 2048 |
| NNM per-device batch × grad_acc | 4 × 2 | 2 × 4 |
| CSD phase | not run | run |
| lm-eval `gen_kwargs` | lm-eval default (`spaces_between_special_tokens=False`) | `spaces_between_special_tokens=True` |
| lm-eval base revision | not pinned | `299a856` |

## Main Results

### Dev set (200 records drawn from the training data)

| phase | avg_loss: start → epoch 1 → epoch 2 | rougeL: start → epoch 1 → epoch 2 | exact_match | final threshold |
|---|---|---|---|---|
| `gemma/csd` | 2.898 → 2.922 → 2.930 | 0.78 → 0.81 → 0.30 | 0.0 at all 3 | 0.0 |
| `gemma/amid` | 2.898 → 2.781 → 2.781 | 0.78 → 1.41 → 1.25 | 0.0 at all 3 | 0.0 |
| `gemma/nnm` | 2.875 → 2.734 → 2.734 | 0.49 → 1.78 → 0.79 | 0.0 at all 3 | 0.0 |

The last train-log line of each phase (global iter 2490), and the mean step time per epoch:

| phase | loss | ds_loss | nnm_loss | step time, epoch 0 / epoch 1 (mean) | `amid-kd-8xh200` step time |
|---|---|---|---|---|---|
| `gemma/csd` | −0.0001 | −0.0001 | — | 2.092 s / 2.096 s | — |
| `gemma/amid` | 0.0523 | 0.0523 | — | 2.304 s / 2.304 s | 1.087 s |
| `gemma/nnm` | 1.1844 | 1.1844 | 0.0000 | 2.521 s / 2.535 s | 1.096 s |

Train losses use different objectives, so they cannot be compared across methods.

### lm-eval benchmarks (final checkpoint `2492/`)

Each cell is the value ± its stderr. **No baseline was evaluated** (neither the untrained student nor the teacher), so no number shows a gain or a loss from distillation. The `amid-kd-8xh200` values are the earlier run at `--max-length 1025` with the old detokenisation (reference only).

**Math** (strict-match / flexible-extract; MATH: exact_match / math_verify):

| model | GSM8K (n=1319) | MATH (n=5000) | GSM-Plus (n=10552) |
|---|---|---|---|
| CSD | 0.0326 ± 0.0049 / 0.4867 ± 0.0138 | 0.0342 ± 0.0025 / 0.2450 ± 0.0058 | 0.0299 ± 0.0017 / **0.3444 ± 0.0046** |
| AMiD | 0.0667 ± 0.0069 / 0.4890 ± 0.0138 | 0.0232 ± 0.0021 / **0.2510 ± 0.0058** | 0.0205 ± 0.0014 / 0.3308 ± 0.0046 |
| NNM | 0.0402 ± 0.0054 / **0.4898 ± 0.0138** | 0.0230 ± 0.0021 / 0.2442 ± 0.0058 | 0.0151 ± 0.0012 / 0.3337 ± 0.0046 |
| *ref. `amid-kd-8xh200` AMiD* | *0.0599 / 0.4936* | *0.0256 / 0.2456* | *0.0202 / 0.3356* |
| *ref. `amid-kd-8xh200` NNM* | *0.0447 / 0.4882* | *0.0236 / 0.2432* | *0.0172 / 0.3349* |

**Knowledge and code**:

| model | MMLU-STEM acc (n=3153) | SciQ acc / acc_norm (n=1000) | MBPP pass@1 (n=500) |
|---|---|---|---|
| CSD | 0.4789 ± 0.0086 | 0.906 ± 0.0092 / 0.729 ± 0.0141 | 0.346 ± 0.0213 |
| AMiD | 0.4789 ± 0.0086 | **0.911 ± 0.0090** / 0.758 ± 0.0136 | **0.358 ± 0.0215** |
| NNM | **0.4818 ± 0.0086** | **0.911 ± 0.0090** / **0.786 ± 0.0130** | 0.348 ± 0.0213 |
| *ref. `amid-kd-8xh200` AMiD* | *0.4754* | *0.908 / 0.756* | *0.0 (invalid, `▁` in code)* |
| *ref. `amid-kd-8xh200` NNM* | *0.4821* | *0.911 / 0.782* | *0.0 (invalid, `▁` in code)* |

Bold marks the best of the three models in this run. Strict-match on GSM8K and GSM-Plus is a format measure: only 128 (CSD), 240 (AMiD) and 152 (NNM) of the 2,638 GSM8K sample lines contain the `####` marker that strict-match looks for. Flexible-extract is the usable number.

## Insights

**The detokenisation fix turns Gemma MBPP from 0.0 into a usable score of about 0.35.** No MBPP sample of any model contains `▁` (0/500 each), against 499/500 in `amid-kd-8xh200`. pass@1 is 0.346 (CSD), 0.358 (AMiD) and 0.348 (NNM). Some answers still contain `\r` (84, 316 and 247 of 500), which did not stop them from passing often; the effect of `\r` on pass@1 was not measured. Source: `outputs/eval_results/gemma-kd-len2048/vllm/<label>/google__gemma-2-2b-it/samples_mbpp_*.jsonl`.

**Doubling `--max-length` doubled the training time but did not move any benchmark beyond its stderr.** Step time went from 1.09 s to 2.30 s (AMiD) and 2.53 s (NNM), because the loader pads every sequence to `--max-length`. On the benchmarks that were valid in both runs, AMiD and NNM differ from `amid-kd-8xh200` by at most 0.0054 on GSM8K/GSM-Plus flexible-extract, MATH math_verify, MMLU-STEM and SciQ: for example GSM8K flexible 0.4890 vs 0.4936 (stderr 0.0138) and MATH math_verify 0.2510 vs 0.2456 (stderr 0.0058). This run also changed the detokenisation for every task, so the two runs are not a clean A/B test of the length alone.

**The three methods are statistically tied on every lm-eval task.** GSM8K flexible-extract spans 0.4867–0.4898 (stderr 0.0138), MATH math_verify 0.2442–0.2510 (0.0058), MMLU-STEM 0.4789–0.4818 (0.0086), and MBPP 0.346–0.358 (0.021). The largest gaps are SciQ acc_norm, where NNM (0.786) leads CSD (0.729) by about 4 stderr, and GSM-Plus flexible-extract, where CSD (0.3444) leads AMiD (0.3308) by about 3 stderr.

**CSD is the only phase whose dev loss rose, and its dev rougeL fell sharply at epoch 2.** CSD dev avg_loss went 2.898 → 2.922 → 2.930 while AMiD and NNM fell to 2.781 and 2.734. CSD rougeL dropped from 0.81 to 0.30 at epoch 2, the sanity check this report flags. The rise (0.032) stayed below `--loss-eps 0.1`, so the threshold stayed 0. The CSD train loss ends at −0.0001. On lm-eval the CSD model is not worse than the others (see the previous paragraph), so the dev drop did not carry over to the benchmarks. The dev set overlaps the training data and its exact_match is 0.0 everywhere, so it is a weak signal. Source: `results/gemma-kd-len2048/gemma2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4/log.txt`.

**Student generation never ran in the real run.** The adaptive threshold stayed 0.0 in all three phases, so no step generated, and the step time was flat across both epochs (for example CSD 2.092 s vs 2.096 s). The generation path was exercised only in the pre-run smoke tests, where AMiD reached 141.7 GB of 143.8 GB per GPU. In the real run, AMiD also held 141.7 GB per GPU (allocator reservation) and finished without OOM.

**The NNM regulariser again contributes almost nothing.** 240 of 249 NNM train-log lines show `nnm_loss: 0.0000`; the largest is 0.0018. `amid-kd-8xh200` showed the same (238 of 249 lines). So NNM here behaves almost like skew forward KL with a replay buffer.

**The final checkpoint is not the best by dev rougeL in any of the 3 phases.** rougeL peaks after epoch 1 in all phases (0.81, 1.41, 1.78) and drops at epoch 2 (0.30, 1.25, 0.79). lm-eval used only `2492/`; the `1246/` checkpoints are kept.

## Artifacts

All paths are relative to `/nvme/annp36-home/work/AMiD` on HGX47.

| phase | final checkpoint | `adapter_model.bin` sha256 | `<save_path>` size |
|---|---|---|---|
| `gemma/csd` | `results/gemma-kd-len2048/gemma2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4/2492` | `8415cb05e41f02c3e3dc9adbd97c58089843e91408b26aafa594148e1f7d3e3a` | 157M |
| `gemma/amid` | `results/gemma-kd-len2048/gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4/2492` | `e1f8f582d4fe4e7322f8842f5d8d9dabd9ae70edb0cbcedd48e8b8aef8f75bb5` | 156M |
| `gemma/nnm` | `results/gemma-kd-len2048/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492` | `445e6bfc74a65c8dec2649169fc16715563feb8fe3e0856c8b798d5908069dd4` | 156M |

- **Intermediate checkpoints**: `<save_path>/1246/` for every phase (not evaluated).
- **Training metrics**: `<save_path>/log.txt`; dev answers in `<save_path>/eval/<epoch>/answers.jsonl`.
- **Logs and state**: `runs/gemma-kd-len2048/{gemma,eval}/` (2.7M): `driver.log`, `<phase>.log`, `exit_code`, `<phase>.exit_code`, `<phase>.ckpt`, `data.sha256`, `processed_data.sha256`, `eval/eval_2.log`, `eval/eval_2.exit_code`.
- **lm-eval outputs**: `outputs/eval_results/gemma-kd-len2048/` (622M): `vllm/<label>/` with 18 `results_*.json`, 90 `samples_*.jsonl` and `DONE`; per-model logs in `logs/<label>.log`. The sha256 of the 18 results JSON files, concatenated in sorted path order, is `6ae4d55dab585f16df839ef80915b0f9e6b764aae7abdee7b65548a62dad853a`.
- **Earlier runs**: nothing of `amid-kd-8xh200` (`results/gemma2-2b-it#*`, `outputs/eval_results/{vllm,logs}/`) was moved or overwritten.
- **Disk**: `/nvme` has 5.9T free.
- **GPU release**: all 8 GPUs at 0 MiB and 0% after the run; no compute apps and no leftover `torchrun`, `ray`, `vllm` or `lm_eval` processes of this run.
- **Upload**: none. The run input lists no upload; nothing was sent to Drive or Hugging Face.

## Limitations

- **Single seed (10), final checkpoint only.** The `1246/` checkpoints were not evaluated, although dev rougeL favours them in all 3 phases.
- **No baseline.** Neither the untrained student nor the teacher was evaluated.
- **Not a clean comparison with `amid-kd-8xh200`.** This run changed `--max-length`, the NNM micro-batch and the eval detokenisation together.
- **Student generation was not exercised in training**, because the threshold stayed 0. Results say nothing about the generation path at length 2048 beyond the 3-step smoke tests.
- **`\r` remains in some MBPP answers** (84–316 of 500); its effect on pass@1 was not measured.
- **Strict-match understates Gemma** on GSM8K and GSM-Plus because most answers lack the `####` marker.
- **The dev set overlaps the training data**, and dev exact_match is 0.0 everywhere.
- **8 GPUs**: 8 per-rank replay buffers (`deque(maxlen=1000)`); each rank sees 1/8 of each epoch.
- **Not pinned**: lm-eval benchmark datasets load from the Hub through lm-eval's task configs. `eval_2.sh` patches lm-eval's `hendrycks_math` assertion.
- **MBPP executed model-written code** on the server with only lm-eval's timeout.

## Next Steps (not run)

- Evaluate the untrained `gemma-2-2b-it` (and, if budget allows, `gemma-2-9b-it`) with the same `eval_2.sh`, to get a baseline.
- Run lm-eval on the `1246/` checkpoints, where dev rougeL peaked.
- Re-score `amid-kd-8xh200` with the detokenisation fix, to separate the effect of `--max-length` from the eval change.
- Investigate the `\r` in MBPP answers and the near-zero `nnm_loss`.
- Upload checkpoints and eval outputs if wanted, as was done for `amid-kd-8xh200`.

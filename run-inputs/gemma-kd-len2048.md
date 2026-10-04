# Project

- `name`: `AMiD`
- `run_id`: `gemma-kd-len2048`
- `local_path`: `/Users/savoxism/Documents/GitHub/AMiD`
- `ssh_host`: `vt-admin` (HGX47, user `vt_admin`, 8× H200)
- `remote_path`: `/nvme/annp36-home/work/AMiD`
- `source_revision`: `e59f9ab` on `wind123mountain/AMiD` branch `nnm2` (pushed 2026-10-04; on top of `d3451a8`, the merge of PR #1). The server copy was synced by rsync and its run files hash the same as this commit:

  | file | sha256 |
  |---|---|
  | `run_gemma.sh` | `6593a15ffa8d38451da62ba1c01e5abaf8a326288ff6e452b5337b64d9f52a9f` |
  | `run_eval.sh` | `5dac1c70797142d8ac0059f6228bf8813c497858af8466e7afb227ee5534ceb5` |
  | `scripts/run_8xh200_common.sh` | `bbdccb77ac70d02412282a186b69a11870331d1b55eee26371883731d1ba5842` |
  | `scripts/eval/eval_2.sh` | `3f94c465d393b63c99d0d42d6f96fdfa336956279b83ac2e05d1227413ff1c7f` |
  | `scripts/csd/train_gemma2_2B_it.sh` | `3e21fd670a150f511a9607d114fc0d79d86050bc9a3194a6a8ef97e6ea13b12e` |
  | `scripts/amid/train_gemma2_2B_it.sh` | `cbe8700df775982abbd8b0ae0c3c691c93e708d298706c38d58eee8dc28a3d93` |
  | `scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh` | `0f8d1875a4607d52dd986397e5e33d300b6b37f4a83a6c12783fd1b5d600723d` |
  | `scripts/watch_gemma-kd-len2048.sh` | `d244ed470fa69b6524aba2c7ebf6e986d26a6fa771aff3cd2fd25fb5f0f73a08` |

- `entrypoints`: `run_gemma.sh`, then `run_eval.sh` (runs `scripts/eval/eval_2.sh`). Both source `scripts/run_8xh200_common.sh` and default to `RUN_ID=gemma-kd-len2048`.
- `environment`: the same `.venv` as `amid-kd-8xh200` (Python 3.10.22; torch 2.10.0+cu128, transformers 4.57.3, deepspeed 0.19.7, peft 0.18.1, vllm 0.17.1, lm_eval 0.4.12, tokenizers 0.22.2, ray 2.59.0).

## What changed since `amid-kd-8xh200`

From the collaborator's `upstream/nnm2` (`5341c44`), kept as they wrote them:
- `--max-length` 1025 → **2048** in all three Gemma scripts (`--max-prompt-length` stays 512).
- Gemma NNM micro-batch 4 → **2**, so its grad accumulation goes 2 → 4. B_global stays 64.
- `run_gemma.sh` now trains **CSD** as well: csd → amid → nnm.

Fixes made for this run:

| # | problem | fix |
|---|---|---|
| 1 | Upstream `run_gemma.sh` was the 2-GPU version (`CUDA_VISIBLE_DEVICES=0,1`), with no pins, GPU gate, exit codes or checkpoint check. | Back on the 8×H200 driver: `run_phase` for csd, amid, nnm. Only the two Gemma pins are checked, so a moved Qwen `main` cannot stop this run. |
| 2 | The scripts' save paths are the same as `amid-kd-8xh200`'s. `run_phase` would have renamed those checkpoints (already on HF) to `.failed_*`. | The three Gemma scripts read `SAVE_PATH=${SAVE_PATH:-<old default>}`; `run_phase` passes `results/gemma-kd-len2048/...`. Standalone use keeps the old default. |
| 3 | `eval_2.sh` pointed Gemma CSD and AMiD at `..._8_1e-4/2492`, which no script writes (the scripts write `_4_`). | Checkpoints come from the `runs/$RUN_ID/gemma/<phase>.ckpt` records (only validated checkpoints); without `RUN_ID`, the scripts' default `_4_` paths. Labels say `_4_`. |
| 4 | Upstream `eval_2.sh` ignored `lm_eval` failures (no exit code, no `DONE`/`FAILED`) and always exited 0. | `DONE`/`FAILED` markers, a model with `DONE` is skipped, exit 1 if any task failed. |
| 5 | Gemma output had `▁` (U+2581) in place of spaces after a newline: MBPP pass@1 0.0 in `amid-kd-8xh200`. Cause: lm-eval passes `spaces_between_special_tokens=False` to vLLM, and vLLM then returns an added token that follows another added token as its raw string. In Gemma, `\n` and the space runs (`▁▁`, `▁▁▁`, …) are added tokens. | `eval_2.sh` passes `spaces_between_special_tokens=True` in `--gen_kwargs` for every task. |
| 6 | `eval_2.sh` loaded `google/gemma-2-2b-it` without a revision. | `revision=299a8560…` in `--model_args`. |
| 7 | Results of different runs shared `outputs/eval_results/vllm/<label>`, so `DONE` from one run could skip another. | With `RUN_ID` set, results go to `outputs/eval_results/gemma-kd-len2048/{vllm,logs}/`. |
| 8 | `run_eval.sh` required the three Qwen phases. | It requires `gemma/csd`, `gemma/amid`, `gemma/nnm`. |

# Schedule

- `start_at`: not launched. Launch only after the user approves this plan and after the GPU check in `# Commands`.
- `timezone`: every time in this file is Vietnam time (UTC+7). The server logs in UTC; convert when reporting.
- `max_runtime`: an estimate, not a measurement.
- No generation (threshold stays 0, as in every Gemma phase of `amid-kd-8xh200`): about 1.5–2 h per phase, because the loader pads every sequence to 2,048 tokens instead of 1,025 (that run took about 49 min per phase). Training is then about 5–6 h. Eval is about 20–40 min per model, because answers can be longer, so about 1–2 h. **Total about 6–8 h.**
  - If the adaptive threshold rises (it goes up by 0.1 each time the dev loss rises by ≥ 0.1 after an epoch), some steps also generate up to 1,536 tokens. The smoke tests, where every step generates, spent about 90 s per optimizer step; 2,492 such steps would take about 60 h per phase. The real run starts at threshold 0 and can only move after epoch 1, so it should sit far below this bound.
- There is no enforced time limit. Record the real step time from the first 100 steps of `gemma/csd` and update the estimate.

# Data

- `input`: `VoCuc/UltraInteract-Infer@3c2fb0d397d3ef2d505f3fe48ad4e30f30733895`. This is still the dataset's latest commit (checked 2026-10-04 11:13). File: `data/dpo/google/gemma-2-9b-it/generated_train.jsonl`, responses generated by the teacher.
  - Upstream changed `tools/generate_vllm.py` (temperature 0.0, `max_tokens` 5120) but did not upload new data, and this run does not regenerate it. It trains on the same pinned file as `amid-kd-8xh200`.
- `train`: all 79,751 records, already tokenised by `tools/process_data_ultraInteract.py` with the gemma-2-9b-it tokenizer and chat template (`--max-prompt-length 512`). The processing does not truncate responses; the loader cuts each sequence at `--max-length`, now 2048, so long responses keep more tokens than in `amid-kd-8xh200`.
- `validation`: the same 200 records (`random.seed(42)`, drawn from the training records).
- `benchmark`: dev ROUGE-L / exact match during training, then `scripts/eval/eval_2.sh` (GSM8K, MATH `minerva_math` 4-shot, MMLU-STEM 5-shot, SciQ, MBPP 3-shot without chat template, GSM-Plus).
- `manifest` (sha256, checked on the server 2026-10-04 11:19; equal to `amid-kd-8xh200`):
  - `data/dpo/google/gemma-2-9b-it/generated_train.jsonl`: `0c0679f9251a5f8f3983ddf01a568cabc47187e59971cc7a0e288befbab9fc30`
  - `processed_data/ultraInteract/google/gemma-2-9b-it/`:
    - `train_0.bin`: `25db4acce56291e0c50b38168d679384e24afb5881fc4d1d95d7dec07ffcc4f3`
    - `train_0.idx`: `bf72af5d4f68c78317f514844d648f88d884f3a73940fc4d4454af11cbadc611`
    - `valid_0.bin`: `adb31190bb4508400da003cb1164690feba444a8968777dbdf1ab7897fa98854`
    - `valid_0.idx`: `7d153f6595d9257a06090a445aa96239f3767dca937f2df9e4da585734c5af40`
  - `run_gemma.sh` writes `runs/gemma-kd-len2048/gemma/{data,processed_data}.sha256`; they must equal these values. Processing is skipped while `train_0.idx` and `valid_0.idx` exist. Do not delete `processed_data/`: rebuilding it changes the record order and the hashes.

# Model

Weights are in `HF_HOME=/nvme/annp36-home/.cache/huggingface`. `hf_prepare` checks that `main` resolves to the pinned SHA and that every shard exists. Training runs online (transformers 4.57.3 calls the Hub for large-vocab tokenizers even offline), and `check_pins` fails a phase with exit 5 if a cached `main` ref moved.

| role | repo_id | revision |
|---|---|---|
| student | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| teacher | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` (fp16) |

- `task`: knowledge distillation, gemma-2-9b-it → gemma-2-2b-it, LoRA on the student (r 16, α 32, dropout 0.05).
- `method`: 3 phases, one at a time.

  | order | phase | script | trainer | `--type` | parameters |
  |---|---|---|---|---|---|
  | 1 | `gemma/csd` | `scripts/csd/train_gemma2_2B_it.sh` | `base_finetune.py` | `adaptive-csd` | `ab`/`pr`, α 0.5, λ 0.5 |
  | 2 | `gemma/amid` | `scripts/amid/train_gemma2_2B_it.sh` | `base_finetune.py` | `adaptive-amid` | `ab`/`pr`, α 0.5, λ 0.5 |
  | 3 | `gemma/nnm` | `scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh` | `finetune.py` | `adaptive-sfkl` | skew α 0.05, NNM ratio 0.2, K 128, 4 layers, d' 256, 500 centroid batches, `--delta-threshold 0.03`, replay `decreasing`, mixed α 0.5 |

  All phases: `--student-gen`, adaptive threshold from 0.0, `--loss-eps 0.1`, replay capacity 1000; DeepSpeed `ds_config_zero0_bf16.json`.
- `seed`: 10.
- `optimizer / schedule`: lr 1e-4, cosine, no warmup, weight decay 1e-2, clip 1.0, `--kd-ratio 1.0`, `--max-length 2048`, `--max-prompt-length 512`. Sampling: top-k 0, top-p 1.0, temperature 1.0.
- `epochs`: 2. 1,246 steps per epoch, `total_iters` 2,492 per phase.
- `batch sizes`: B_global = N_GPU × B_device × G_accum = 64 in every phase.

  | phase | this run | eval batch |
  |---|---|---|
  | csd | 8 × 4 × 2 = 64 | 16 |
  | amid | 8 × 4 × 2 = 64 | 16 |
  | nnm | 8 × 2 × 4 = 64 | 32 |

## GPU policy

- All 8 H200 in one torchrun DDP job per phase; phases never overlap: csd → amid → nnm → eval.
- Eval also uses all 8: vLLM `data_parallel_size=8` through ray, one model at a time.
- `require_idle_gpus` runs before each phase and before eval: 8 H200 and no compute process within 5 minutes, otherwise exit 3. It never kills or reuses another process.
- State at planning time (2026-10-04 11:13): 8 H200, 0 MiB used, no compute process, no tmux session; `/nvme` 5.9 T free. The smoke tests below then used the GPUs from 11:15 to 11:50 and released them (0 compute processes at 11:51).
- `LOI_NHAN_ZILLEXA.txt` (from upstream) says another team was given this machine. It is unverified. Check with the GPU owner (wind123mountain) before launch; do not launch if anyone else has a job on the GPUs.

# Commands

```bash
cd /nvme/annp36-home/work/AMiD

# preflight
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # must be empty
tmux ls; who                                                             # nobody else running work
sha256sum run_gemma.sh run_eval.sh scripts/run_8xh200_common.sh scripts/eval/eval_2.sh   # must match # Project (the server's .git is an old checkout, 4b6d2c0; files come by rsync)
ls runs/gemma-kd-len2048 results/gemma-kd-len2048 2>/dev/null            # must not exist (fresh run id)

# full run: training, then lm-eval only if all 3 phases exited 0; detached from SSH
setsid nohup bash -c 'bash run_gemma.sh && bash run_eval.sh' > /dev/null 2>&1 < /dev/null &

# lm-eval alone (after run_gemma.sh exited 0)
setsid nohup bash run_eval.sh > /dev/null 2>&1 < /dev/null &

# monitoring (read-only, from the laptop; Ctrl-C stops only the watcher)
ssh -t vt-admin 'cd /nvme/annp36-home/work/AMiD && INTERVAL=10 bash scripts/watch_gemma-kd-len2048.sh'

# stop the run (kills the driver's process group, including torchrun); use eval/run.pid once eval started
kill -TERM -- -$(ps -o pgid= -p $(cat runs/gemma-kd-len2048/gemma/run.pid) | tr -d ' ')

# verification
cat runs/gemma-kd-len2048/{gemma,eval}/exit_code runs/gemma-kd-len2048/gemma/*.exit_code runs/gemma-kd-len2048/eval/eval_2.exit_code
cat runs/gemma-kd-len2048/gemma/*.ckpt runs/gemma-kd-len2048/gemma/*.sha256
grep -h '^dev | avg_loss' results/gemma-kd-len2048/*/*/log.txt
ls outputs/eval_results/gemma-kd-len2048/vllm/*/*/{DONE,FAILED} 2>/dev/null
find outputs/eval_results/gemma-kd-len2048/vllm -name 'results_*.json' | sort
grep -c $'▁' $(find outputs/eval_results/gemma-kd-len2048/vllm -name 'samples_mbpp_*.jsonl')   # must be 0 or near 0
```

- `run_gemma.sh` repeats the model and data checks in seconds (cache hits), then runs the 3 phases through `run_phase`.
- Built-in evaluation: `--do-valid --eval-gen` on the 200 dev records before training and after each epoch; lines `dev | avg_loss …` in `<save_path>/log.txt`, answers in `<save_path>/eval/<epoch>/answers.jsonl`.
- `run_eval.sh` → `eval_2.sh`: 6 lm-eval tasks on the 3 final checkpoints, base `google/gemma-2-2b-it@299a856` plus the LoRA through vLLM (`lora_local_path`, `max_lora_rank=32`). Settings are `eval_2.sh`'s: chat template + `fewshot_as_multiturn`, `max_new_tokens=5120`, greedy for MBPP; now also `spaces_between_special_tokens=True`. MBPP executes model code (`HF_ALLOW_CODE_EVAL=1`). `eval_2.sh` patches the `hendrycks_math` assertion in the installed `lm_eval/api/task.py` (already patched in `.venv`).

## Smoke tests (2026-10-04 11:15–11:50)

Run on the server in a scratch directory, `smoke_gemma-kd-len2048/` (deleted afterwards), with the same code and the same `.venv`, while the GPUs were idle.

1. **Eval detokenization** (MBPP, `--limit 20`, GPU 0, the `amid-kd-8xh200` AMiD checkpoint `2492`):

   | `--gen_kwargs` | samples with `▁` | MBPP pass@1 |
   |---|---|---|
   | as in `amid-kd-8xh200` | 20/20 | 0.00 |
   | `+ spaces_between_special_tokens=True` | 0/20 | 0.45 |

2. **Training, all three scripts on 8 GPUs** at `--max-length 2048` with their own batch settings. Each ran with `SAVE_PATH=<scratch> … --total-iters 3 --init-threshold 1.0`, so student generation (LoRA `generate`, up to 1,536 new tokens) runs on every step. This code path never ran in `amid-kd-8xh200`, where the threshold stayed 0.

   | phase | exit | wall time | peak GPU memory (per GPU, of 143,771 MiB) | dev at step 0 |
   |---|---|---|---|---|
   | csd | 0 | 459 s | 113,747 MiB | loss 2.898, rougeL 0.78 |
   | amid | 0 | 470 s | 141,747 MiB | loss 2.898, rougeL 0.78 |
   | nnm | 0 | 834 s (includes building 128 centroids from 500 batches) | 90,829 MiB | loss 2.875, rougeL 0.49 |

   - No traceback, OOM or NCCL error. The "right-padding was detected" warning also appears in `amid-kd-8xh200`'s logs (from `--eval-gen`); it is not new.
   - **Risk: AMiD reached the card's memory** (141.7 GB) while generating on every step and still finished. In the real run generation is rare, but an OOM late in the run is possible. If it happens: report and ask; do not change length, batch or the allocator settings without approval.
   - Scope: 3 optimizer steps per phase, no checkpoint save (save happens at epoch end), so `check_ckpt` and the `adapter_model.bin` save path are not covered by the smoke test. They are the same code as in `amid-kd-8xh200`, where they passed.

# Outputs

All outputs stay in the project directory. Nothing of `amid-kd-8xh200` is moved or overwritten.

- `output_path`: `runs/gemma-kd-len2048/` (logs, state), `results/gemma-kd-len2048/` (checkpoints), `outputs/eval_results/gemma-kd-len2048/` (lm-eval).
- `checkpoint`: `<save_path>/{1246,2492}/` with `adapter_model.bin`, `adapter_config.json` and the tokenizer files. Final: `<save_path>/2492/`.

  | phase | `save_path` |
  |---|---|
  | `gemma/csd` | `./results/gemma-kd-len2048/gemma2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4` |
  | `gemma/amid` | `./results/gemma-kd-len2048/gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4` |
  | `gemma/nnm` | `./results/gemma-kd-len2048/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` |

- `metrics`: `<save_path>/log.txt`, `<save_path>/eval/<epoch>/answers.jsonl`.
- `logs`: `runs/gemma-kd-len2048/gemma/{driver.log,<phase>.log,current_phase,data.sha256,processed_data.sha256}`, `runs/gemma-kd-len2048/eval/{driver.log,eval_2.log}`.
- `pid_file`: `runs/gemma-kd-len2048/{gemma,eval}/run.pid`
- `exit_code_file`: `runs/gemma-kd-len2048/{gemma,eval}/exit_code` (driver), `runs/gemma-kd-len2048/gemma/<phase>.exit_code`, `runs/gemma-kd-len2048/eval/eval_2.exit_code`.
- `checkpoint_record`: `runs/gemma-kd-len2048/gemma/<phase>.ckpt` (final path, `total_iters`, B_global, size, weight sha256).
- `benchmarks`: `outputs/eval_results/gemma-kd-len2048/vllm/<label>/` (`results_*.json`, `samples_*.jsonl`, `DONE` or `FAILED`), per-model logs in `outputs/eval_results/gemma-kd-len2048/logs/<label>.log`.

  | label | checkpoint |
  |---|---|
  | `gemma-2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4` | `gemma/csd` |
  | `gemma-2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4` | `gemma/amid` |
  | `gemma-2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` | `gemma/nnm` |

- `report`: `run-outputs/out_gemma-kd-len2048.md`, after the run.

# Upload

- Not part of this run. Nothing is uploaded unless the user asks after the run.
- If asked, use the same pattern as `amid-kd-8xh200`: Drive with `rclone` remote `annp_nqd` and `--drive-root-folder-id=root` (the remote's default root is another folder); HF one public repo per final checkpoint, `HF_HUB_DISABLE_XET=1`, never print the token. Ask before writing into an existing Drive folder or HF repo.

# Acceptance

## Expected artifacts

- `runs/gemma-kd-len2048/gemma/exit_code` = 0 and `csd`, `amid`, `nnm` `.exit_code` = 0.
- Each phase passes `check_ckpt` and has a `.ckpt`: logged `--save` equals the `save_path` above; `--n-gpu 8` and B_global 64; ≥ 2 step dirs; non-empty `adapter_model.bin` and `tokenizer_config.json` in the last; ≥ 3 `dev | avg_loss` lines.
- `data.sha256` / `processed_data.sha256` equal the `# Data` manifest.
- `runs/gemma-kd-len2048/eval/exit_code` and `eval_2.exit_code` = 0; 3 labels with `DONE`, no `FAILED`, a `results_*.json` for each of the 6 tasks.

## Expected metrics

- No target numbers. Report per phase: dev `rougeL` / `exact_match` at epochs 0, 1, 2, final train loss (and `nnm_loss`), the adaptive threshold; lm-eval: GSM8K and GSM-Plus strict/flexible, MATH `exact_match`/`math_verify`, MMLU-STEM `acc`, SciQ `acc`/`acc_norm`, MBPP `pass_at_1`, with stderr.
- Sanity checks (investigate, do not abort): `total_iters` 2,492; last step dir `2492`; finite loss; dev rougeL at epoch 2 ≥ epoch 0; MBPP samples without `▁`; no task at 0 or chance level.

## Failure conditions

- GPUs not idle within 5 min → exit 3 before the phase.
- torchrun non-zero → that phase's exit code; the driver stops and eval does not start.
- `check_ckpt` fails → 4. Pinned `main` moved → 5 (do not retry; report).
- `hf_prepare` fails (pin mismatch, missing shard, no HF credential) → before any GPU use.
- `run_eval.sh`: 1 if a phase lacks exit 0 or `.ckpt`, 3 at the GPU gate, 1 if any task failed (see `<label>/FAILED`, `eval_2.log`).
- Traceback, CUDA OOM, NCCL error, `loss: nan/inf`, full disk, dead torchrun worker in `<phase>.log` (the watcher greps these).

## Retry policy

- Rerunning the launch command is safe: done phases are skipped; a failed phase restarts from step 0 after its partial `save_path` is renamed to `<save_path>.failed_<UTC time>` and its log to `<phase>.log.<UTC time>`.
- Retry a failed phase once, unchanged. If it fails again, stop and report.
- OOM at `--max-length 2048`: do not lower the length or batch on your own; report and ask (a change needs a new `run_id`).
- `run_eval.sh` reruns only models without `DONE`.
- Within this `run_id`, never change dataset, seed, batch, objective, sampler or hyperparameters.

## Monitoring

- Watcher: the command in `# Commands` (phase progress, step, loss, lr, ETA, latest dev line, error lines, eval models done/3, GPU table with process owners, free disk).
- While active, a read-only health check every 20 minutes: PID and exit state, optimizer step, finite loss, log/checkpoint mtimes, disk, GPUs, failure patterns. A stall is never inferred from one idle sample.

## Known limitations (must appear in the report)

- Single seed (10). Dev set drawn from training records (fit, not generalisation). Final checkpoint only, not chosen by dev score.
- **Not comparable one-to-one with `amid-kd-8xh200`**: `--max-length` 2048 vs 1025, NNM micro-batch 2 vs 4 (grad acc 4 vs 2), and the eval detokenization fix changes the text of every Gemma answer (spaces instead of `▁`), so GSM8K/MATH/GSM-Plus can move as well as MBPP.
- 8 GPUs: 8 per-rank replay buffers (`deque(maxlen=1000)`), each rank sees 1/8 of an epoch.
- No lm-eval baseline for the untrained student or the teacher.
- lm-eval benchmark datasets are not pinned; cite lm_eval 0.4.12 and the task versions in each `results_*.json`. `eval_2.sh` patches lm_eval's `hendrycks_math` assertion.
- MBPP runs model-written code on the server with only lm-eval's timeout.
- The Qwen blocks in `eval_2.sh` stay commented out (as upstream left them); this run evaluates Gemma only.

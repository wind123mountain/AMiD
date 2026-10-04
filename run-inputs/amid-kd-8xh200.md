# Project

- `name`: `AMiD`
- `run_id`: `amid-kd-8xh200`
- `local_path`: `/Users/savoxism/Documents/GitHub/AMiD`
- `ssh_host`: `vt-admin` (HGX47, user `vt_admin`, 8× H200)
- `remote_path`: `/nvme/annp36-home/work/AMiD`
- `source_revision`: `4b6d2c0` (branch `nnm2`, cloned from `origin`) plus uncommitted changes copied by rsync. The file hashes below match on both sides. Commit before the run so the report can cite a clean revision.
  - modified: `run_gemma.sh`, `run_qwen.sh`, `scripts/download_data.sh`, `scripts/csd/train_qwen2.5_0.5B_it.sh`, `scripts/amid/train_qwen2.5_0.5B_it.sh`, `scripts/distillm-nnm/qwen2.5/train_qwen2.5_0.5B_it_2e.sh`, `scripts/eval/eval_2.sh`
  - new: `run_eval.sh`, `scripts/run_8xh200_common.sh`, `scripts/preflight_amid.sh`, `scripts/watch_amid-kd-8xh200.sh`

  | file | sha256 |
  |---|---|
  | `run_gemma.sh` | `7d7743bc284a362add25009ecda5183dd9c0f78d11d2b8379c5b9f5a46557563` |
  | `run_qwen.sh` | `131aed873ce9d5b038438dbb507759ef0eef1d4226a780e522492eb5a6753171` |
  | `run_eval.sh` | `1aaa12a2304b6c69a78d60bb9e5c1da864efd61faff2e932cca6f8cdd34d47a6` |
  | `scripts/run_8xh200_common.sh` | `1bc59989cc5b838a925339ab7ac34a5dd94f0cc9ba31747fc09c170e9e060821` |
  | `scripts/preflight_amid.sh` | `6256986c8362449d6605108f89f07dc9dea54f2276a528102df083ba8e41002c` |
  | `scripts/watch_amid-kd-8xh200.sh` | `5ae49c7ecade40172fc5867cef77076143ff2b1e741a63fd55bcf2e57db4b65b` |
  | `scripts/download_data.sh` | `2303641f27140de758f6f7db3c5e9b0c14d5cae30eac850c0b3fab1b97f05b45` |
  | `scripts/eval/eval_2.sh` | `29c5b00e43c321633c4db44981f1c695d88ae737c4d90dcbfef2a01106cc61bc` |
  | `scripts/csd/train_qwen2.5_0.5B_it.sh` | `83db0acbd6980fb2403d7dbeb6980b8dbe79bea9dae63fd23f91929a7ae36f09` |
  | `scripts/amid/train_qwen2.5_0.5B_it.sh` | `8b1183ee4e248bfa5c652bc0f9379c7c8121aa800fcfeda9492518d2df4a19e0` |
  | `scripts/distillm-nnm/qwen2.5/train_qwen2.5_0.5B_it_2e.sh` | `32488f659209c34f1e1b50e4341b66f2a621f862411f6fba77addb6a9e2b2cbc` |

- `entrypoints`: `run_gemma.sh`, then `run_qwen.sh`, then `run_eval.sh` (runs `scripts/eval/eval_2.sh`). All three source `scripts/run_8xh200_common.sh`.
- `environment`: `.venv` (Python 3.10.22) built by `uv sync --python 3.10` from `pyproject.toml`.
  - Versions: torch 2.10.0+cu128, transformers 4.57.3, deepspeed 0.19.7, peft 0.18.1, vllm 0.17.1, lm_eval 0.4.12, ray 2.59.0.
  - `uv` 0.12.22 is a standalone binary in `.uv-tool/`, because the server has no `python3-venv`. Its caches are in `/nvme/annp36-home/.cache/uv`.
  - There is no `uv.lock` (it is gitignored), so the resolved versions above are the record.

# Schedule

- `start_at`: 2026-10-03 14:00:49, launched with the chain command in `# Commands` (driver `run_gemma.sh` pid 2254510).
  - `gemma/amid` torchrun started at 14:00:53 on all 8 GPUs.
  - Preflight already ran from 2026-10-03 12:52:35 to 12:56:46 (`runs/amid-kd-8xh200/preflight/preflight.log`).
- `timezone`: every time in this file is Vietnam time (UTC+7). The server logs in UTC, so times are converted.
- `max_runtime`: measured only for the first phase so far.
  - `gemma/amid`: step time 1.09 s. The phase ran from 14:00:53 to 14:49:29 (48.6 min, exit 0), including the final validation and generation.
  - `gemma/nnm`: step time 1.09 s. The phase ran from 14:49:29 to 15:39:20 (49.9 min, exit 0).
  - `qwen/csd`: started 15:39:23. Step time was 0.37 s in epoch 0.
    - At the end of epoch 0, dev avg_loss rose from 1.87 to 3.72, which is more than `--loss-eps 0.1`. So the adaptive threshold went from 0.0 to 0.1, and the student now generates on part of the steps in epoch 1.
    - Step time in epoch 1 is about 4.8 s. This is the scripted algorithm, not a fault.
    - The phase ended at 17:22:08 (1 h 43 min, exit 0). The final dev avg_loss is 3.47 and rougeL 7.17.
    - In the Gemma phases, dev loss did not rise, so the threshold stayed 0.0 and step time stayed flat.
  - `qwen/amid`: ran from 17:22:08 to 19:06:04 (1 h 44 min, exit 0).
    - Dev avg_loss rose from 1.87 to 2.44 after epoch 0, so the threshold rose to 0.1 as in `qwen/csd`.
    - Step time was 0.44 s in epoch 0 and 4 to 7 s in epoch 1. The final dev avg_loss is 2.36 and rougeL 7.04.
  - `qwen/nnm`: first attempt started 19:06:04 and failed at 19:08:37 (see `## Incident log`). The rerun ran from 19:13:18 to 19:39:52 (26.6 min, exit 0).
    - After epoch 0, dev avg_loss rose from 1.87 to 2.56. This script uses `--delta-threshold 0.03`, so the threshold rose only to 0.03, and epoch 1 took about 15 min.
    - The final dev avg_loss is 2.52 and rougeL 8.45. The teacher-centroid pre-pass takes about 2 min on rank 0 only, so GPUs 1 to 7 are idle during it. Each takes about 8 min per epoch without student generation, and up to about 100 min for epoch 1 if the threshold rises as it did in `qwen/csd`.
  - The script does not enforce a time limit.
  - Evaluation started 19:39:52.
    - The two Gemma models took 18.6 and 19.3 min. Qwen CSD started 20:17:45, so the 5 models should end about 21:15.
    - Evaluation ended 21:30:09 (exit 0, failed models: 0). The 3 Qwen models took 23.5, 24.5 and 24.4 min. The whole chain took 7 h 29 min 20 s. Report: `run-outputs/out_amid-kd-8xh200.md`.
  - The estimate made before the run: evaluation time is not measured. It is 5 models × 6 lm-eval tasks with up to 5,120 new tokens. Record the time of the first model from `outputs/eval_results/logs/` and extrapolate.

# Data

- `input`: `VoCuc/UltraInteract-Infer@3c2fb0d397d3ef2d505f3fe48ad4e30f30733895` (dataset, last modified 2026-10-02), downloaded to `./data/dpo`. Total size is 4.9 G.
  - Each record is an UltraInteract prompt with a response generated by the teacher.
  - Each student uses its own teacher's file: `data/dpo/google/gemma-2-9b-it/generated_train.jsonl` and `data/dpo/Qwen/Qwen3-4B-Instruct-2507/generated_train.jsonl`.
- `train`: all 79,751 records of each file.
  - Tokenised by `tools/process_data_ultraInteract.py` with the teacher's tokenizer and chat template.
  - `--max-prompt-length 512`. The training loader cuts sequences at `--max-length 1025`; no samples are dropped.
  - Prompt/response token stats (Qwen3-4B file): prompt mean 175, max 512; response mean 557, max 2,043.
- `validation`: 200 records drawn with `random.sample` (`random.seed(42)`) **from the training records**, so validation overlaps train (see Known limitations).
- `benchmark`: two levels.
  - During training: the built-in dev evaluation (ROUGE-L / exact match on the 200 validation records).
  - After training: `scripts/eval/eval_2.sh` runs lm-eval on each final checkpoint: GSM8K, MATH (`minerva_math`, 4-shot), MMLU-STEM (5-shot), SciQ, MBPP (3-shot, no chat template), GSM-Plus. The benchmark datasets come from the Hub through lm-eval's task configs and are not pinned (see Known limitations).
- `manifest` (sha256):
  - `data/dpo/google/gemma-2-9b-it/generated_train.jsonl`: `0c0679f9251a5f8f3983ddf01a568cabc47187e59971cc7a0e288befbab9fc30`
  - `data/dpo/Qwen/Qwen3-4B-Instruct-2507/generated_train.jsonl`: `3d4bb80aa1760536c0c9dde6377a753ec1f05d7ed5f46b0750c7bbde754a89ea`
  - `processed_data/ultraInteract/google/gemma-2-9b-it/`:
    - `train_0.bin`: `25db4acce56291e0c50b38168d679384e24afb5881fc4d1d95d7dec07ffcc4f3`
    - `train_0.idx`: `bf72af5d4f68c78317f514844d648f88d884f3a73940fc4d4454af11cbadc611`
    - `valid_0.bin`: `adb31190bb4508400da003cb1164690feba444a8968777dbdf1ab7897fa98854`
    - `valid_0.idx`: `7d153f6595d9257a06090a445aa96239f3767dca937f2df9e4da585734c5af40`
  - `processed_data/ultraInteract/Qwen/Qwen3-4B-Instruct-2507/`:
    - `train_0.bin`: `6385ff4f4761ee1bc9ec8e8cc023924864219a6133f5f2e7b1d74e16dfb03c8e`
    - `train_0.idx`: `f6d142eb3330b5f74c8dfc6f3e4a853e4e25ae096680ba4dea228f5ebce29546`
    - `valid_0.bin`: `704271c40969f9368b130874260a3dec8f25053c2c788f2c27a76420726459a9`
    - `valid_0.idx`: `648536b84c3d470d02f37edb82fe8a6718c83ee98712fa888501665c23e5da2d`
  - Each run also writes its own copies to `runs/amid-kd-8xh200/<tag>/{data,processed_data}.sha256`. These must equal the values above.
- Processing uses `imap_unordered`, so the record order in `train_0.bin` is not deterministic. If `processed_data/` is deleted and rebuilt, the hashes change, and the run needs a new `run_id`. The run scripts skip processing while `train_0.idx` and `valid_0.idx` exist.

# Model

The weights are in the shared HF cache: `HF_HOME=/nvme/annp36-home/.cache/huggingface`, under `hub/models--<org>--<name>/snapshots/<sha>`. They were prefetched once by `hf_prepare` with `HF_HUB_DISABLE_XET=1` and no `hf_transfer`. `hf_prepare` checks that `main` resolves to the pinned SHA and that every shard exists.

Training runs **online**, not with `HF_HUB_OFFLINE=1`:
- transformers 4.57.3 calls the Hub (`model_info`) whenever it loads a tokenizer with a vocabulary over 100k from a repo id, even in offline mode. That covers all four models here. The trainer loads the tokenizer by repo id, so with offline mode every phase would fail at `get_tokenizer` with `OfflineModeIsEnabled` (reproduced on the server on 2026-10-03).
- To keep the pin, `run_phase` runs `check_pins` after every phase. It reads `$HF_HOME/hub/models--*/refs/main` for the four repos and fails the phase with exit code 5 if any of them moved away from its pinned SHA.
- The lm-eval step loads `google/gemma-2-2b-it` with `revision=299a856…`. The Qwen checkpoints are local directories.
- HF account: `Savoxism`; it has access to the gated Gemma-2 repos.

| role | repo_id | revision | shards |
|---|---|---|---|
| Gemma student | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 2 |
| Gemma teacher | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | 4 |
| Qwen student | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | 1 |
| Qwen teacher | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | 3 |

- `task`: knowledge distillation from teacher to student on the UltraInteract responses the teacher generated.
- `method`: 5 training phases, one at a time.

  | order | tag / phase | script | trainer | `--type` | parameters |
  |---|---|---|---|---|---|
  | 1 | `gemma/amid` | `scripts/amid/train_gemma2_2B_it.sh` | `base_finetune.py` | `adaptive-amid` | AMiD `ab`/`pr`, α=0.5, λ=0.5, LoRA r16/α32/dropout 0.05 |
  | 2 | `gemma/nnm` | `scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh` | `finetune.py` | `adaptive-sfkl` | skew α=0.05, NNM ratio 0.2, K=128, 4 layers, d'=256, 500 centroid batches, LoRA r16, `--delta-threshold 0.03` |
  | 3 | `qwen/csd` | `scripts/csd/train_qwen2.5_0.5B_it.sh` | `base_finetune.py` | `adaptive-csd` | `ab`/`pr`, α=0.5, λ=0.5, full fine-tune (no LoRA) |
  | 4 | `qwen/amid` | `scripts/amid/train_qwen2.5_0.5B_it.sh` | `base_finetune.py` | `adaptive-amid` | `ab`/`pr`, α=0.5, λ=0.5, full fine-tune |
  | 5 | `qwen/nnm` | `scripts/distillm-nnm/qwen2.5/train_qwen2.5_0.5B_it_2e.sh` | `finetune.py` | `adaptive-sfkl` | same NNM settings as phase 2, full fine-tune |

  - `gemma/csd` (`scripts/csd/train_gemma2_2B_it.sh`) stays commented out, as it was in the original `run_gemma.sh`.
  - Every phase uses student generation (`--student-gen`, adaptive threshold starting at 0.0, `--loss-eps 0.1`) and a replay buffer with capacity 1000. The NNM phases also use `--replay-ratio decreasing --mixed-alpha 0.5`.
  - The teacher is loaded in fp16 (`--teacher-model-fp16`). DeepSpeed uses `configs/deepspeed/ds_config_zero0_bf16.json` (bf16, ZeRO stage 1).
- `seed`: `10` (training). Data processing uses `random.seed(42)`.
- `optimizer / schedule`:
  - lr `1e-4`, cosine decay, no warmup, weight decay `1e-2`, gradient clip `1.0`, `--kd-ratio 1.0`.
  - `--max-length 1025`, `--max-prompt-length 512`.
  - Sampling for student generation and dev generation: `top_k 0`, `top_p 1.0`, temperature `1.0`.
- `epochs`: 2. Steps per epoch are `int(79,751 / 64)` = 1,246, so each phase runs 2,492 optimizer steps. The log line `total_iters` must show 2,492.
- `batch sizes`: B_global = N_GPU × B_device × G_accum = 64 in every phase.

  | tag | before (2 GPUs) | this run (8 GPUs) | eval batch |
  |---|---|---|---|
  | gemma | 2 × 4 × 8 = 64 | **8 × 4 × 2 = 64** | 16 (amid), 32 (nnm) |
  | qwen | 2 × 16 × 2 = 64 | **8 × 8 × 1 = 64** | 64 |

  - For Qwen, 16 per device cannot give 64 on 8 GPUs: the training script rejects any `EFF_BATCH` that does not divide evenly. `run_qwen.sh` therefore exports `BATCH_SIZE=8`, and the three Qwen scripts now read `BATCH_SIZE=${BATCH_SIZE:-16}`. Their default is unchanged.
  - The CSD and AMiD save paths include the per-device batch, so the Qwen paths end in `_8_1e-4`.
  - `check_ckpt` checks B_global = 64 from the command line logged by each phase and writes it to `<phase>.ckpt`.

## GPU policy

- All 8 H200 (`CUDA_VISIBLE_DEVICES=0,…,7`) run in one torchrun DDP job per phase (`--nproc_per_node 8`).
- The phases never overlap: `gemma/amid` → `gemma/nnm` → `qwen/csd` → `qwen/amid` → `qwen/nnm` → `eval`.
- The `eval` step also uses all 8 GPUs. vLLM runs with `data_parallel_size=8` (one ray actor per GPU, `gpu_memory_utilization=0.8`), one model at a time.
- Before every phase and before `eval`, `require_idle_gpus` polls for up to 5 minutes. It needs 8 visible H200 and no compute process; otherwise it exits with code 3. It never kills or reuses another process.
- Server state at planning time (2026-10-03 12:45): 8 H200, 0 MiB used, no compute processes. `/nvme` had 6.3 T free.
- Host driver 595.71.05 / CUDA 13.1. torch uses its bundled CUDA 12.8 runtime.

# Commands

```bash
cd /nvme/annp36-home/work/AMiD

# preflight (CPU only; already passed 2026-10-03 12:52 → 12:56, safe to rerun)
bash scripts/preflight_amid.sh       # HF auth, prefetch + verify 4 pinned models, pinned dataset, tokenisation
tail -1 runs/amid-kd-8xh200/preflight/preflight.log            # must end with "preflight ok"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # must be empty
git status --short                                               # only the files listed in # Project

# full run: Gemma, then Qwen only if Gemma exits 0, then lm-eval only if Qwen exits 0; detached from the SSH session
setsid nohup bash -c 'bash run_gemma.sh && bash run_qwen.sh && bash run_eval.sh' > /dev/null 2>&1 < /dev/null &

# lm-eval alone (after both training drivers exited 0, e.g. after a failed eval)
setsid nohup bash run_eval.sh > /dev/null 2>&1 < /dev/null &

# monitoring (read-only, from the laptop; Ctrl-C stops only the watcher)
ssh -t vt-admin 'cd /nvme/annp36-home/work/AMiD && INTERVAL=10 bash scripts/watch_amid-kd-8xh200.sh'

# stop the whole run (kills the driver's process group, including torchrun)
kill -TERM -- -$(ps -o pgid= -p $(cat runs/amid-kd-8xh200/gemma/run.pid) | tr -d ' ')   # use qwen/run.pid or eval/run.pid once that driver has started

# verification
cat runs/amid-kd-8xh200/{gemma,qwen,eval}/exit_code
cat runs/amid-kd-8xh200/{gemma,qwen,eval}/*.exit_code
cat runs/amid-kd-8xh200/{gemma,qwen}/*.ckpt
grep -h '^dev | avg_loss' results/*/*/log.txt
ls outputs/eval_results/vllm/*/*/DONE outputs/eval_results/vllm/*/*/FAILED 2>/dev/null
find outputs/eval_results/vllm -name 'results_*.json' | sort
```

- Each `run_*.sh` repeats the preflight steps in a few seconds (cache hits), stays online (see `# Model`), and runs its phases through `run_phase`.
- Evaluation is built into training. `--do-valid --eval-gen` generates on the 200 validation records:
  - once before training (epoch 0);
  - after each epoch (`--eval-interval -1` resolves to 1,246 steps).

  Each evaluation writes a `dev | avg_loss … {exact_match, rougeL}` line to `<save_path>/log.txt` and the answers to `<save_path>/eval/<epoch>/answers.jsonl`.
- `run_eval.sh` then runs `scripts/eval/eval_2.sh` with the 6 lm-eval tasks on the 5 final checkpoints:
  - It refuses to start unless all 5 `<phase>.exit_code` files are 0 and every `<phase>.ckpt` exists.
  - It reads each final checkpoint from `final_checkpoint=` in `<phase>.ckpt`, so only checkpoints that passed `check_ckpt` are evaluated.
  - Gemma: base `google/gemma-2-2b-it@299a856` plus the LoRA adapter through vLLM (`lora_local_path`, `max_lora_rank=32`).
  - Qwen: the full checkpoint directory. `qwen/nnm` is saved with its `projectors.*` weights, which vLLM rejects (`AutoWeightsLoader` raises on unknown keys). `eval_2.sh` therefore writes a copy without them to `outputs/vllm_ckpt/<label>/` and evaluates that. Checkpoints without projectors (`qwen/csd`, `qwen/amid`) are used in place.
  - The generation settings are those of `eval_2.sh`: chat template, `max_new_tokens=5120`, greedy for MATH and MBPP. MBPP runs the generated code on the server (`HF_ALLOW_CODE_EVAL=1`, `--confirm_run_unsafe_code`).
  - `eval_2.sh` patches the installed `lm_eval/api/task.py` in `.venv` once: it replaces the `hendrycks_math` assertion `Answer is not a string` with a `str()` conversion.
- `eval_2.sh` was smoke-tested on 2026-10-03 against stand-in checkpoints built the way the trainer saves them (see Acceptance).

# Outputs

All outputs stay inside the project directory.

- `output_path`: `/nvme/annp36-home/work/AMiD/runs/amid-kd-8xh200` (logs, state) and `/nvme/annp36-home/work/AMiD/results` (checkpoints).
- `checkpoint`: one step directory per epoch, `<save_path>/{1246,2492}/`. The final checkpoint is `<save_path>/2492/`.
  - LoRA phases (Gemma) contain `adapter_model.bin` and `adapter_config.json`.
  - Full fine-tune phases (Qwen) contain `pytorch_model.bin`.
  - Every step directory also holds the tokenizer files.

  | phase | `save_path` |
  |---|---|
  | `gemma/amid` | `./results/gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4` |
  | `gemma/nnm` | `./results/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` |
  | `qwen/csd` | `./results/qwen2.5-0.5B-it#csd/ab_pr_0.5_0.5_8_1e-4` |
  | `qwen/amid` | `./results/qwen2.5-0.5B-it#amid/ab_pr_0.5_0.5_8_1e-4` |
  | `qwen/nnm` | `./results/qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` (the path says `lora`, but this phase is a full fine-tune; the name comes from the script) |

- `metrics`: `<save_path>/log.txt` (train lines every 10 steps, plus the `dev` lines) and `<save_path>/eval/<epoch>/answers.jsonl`.
- `logs`: in `runs/amid-kd-8xh200/<tag>/`:
  - `driver.log`;
  - `<phase>.log` (full torchrun stdout/stderr);
  - `data.sha256`, `processed_data.sha256`;
  - `current_phase`.
- `pid_file`: `runs/amid-kd-8xh200/<tag>/run.pid`
- `exit_code_file`: `runs/amid-kd-8xh200/<tag>/exit_code` (whole driver) and `runs/amid-kd-8xh200/<tag>/<phase>.exit_code` (per phase).
- `checkpoint_record`: `runs/amid-kd-8xh200/<tag>/<phase>.ckpt` gives the final checkpoint path, `total_iters`, B_global, directory size and the weight sha256.
- `preflight`: `runs/amid-kd-8xh200/preflight/` (log plus data hashes per teacher).
- `benchmarks`: `outputs/eval_results/vllm/<label>/` holds lm-eval's `results_*.json` and per-sample `samples_*.jsonl` for each task, plus a `DONE` marker (all 6 tasks exited 0) or a `FAILED` file (one line per failed task). Logs per model are in `outputs/eval_results/logs/<label>.log`, and the whole eval log is `runs/amid-kd-8xh200/eval/eval_2.log`.

  | label (`<label>`) | checkpoint |
  |---|---|
  | `gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4` | `gemma/amid` LoRA on `google/gemma-2-2b-it` |
  | `gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` | `gemma/nnm` LoRA on `google/gemma-2-2b-it` |
  | `qwen2.5-0.5B-it#csd/ab_pr_0.5_0.5_8_1e-4` | `qwen/csd` final step directory |
  | `qwen2.5-0.5B-it#amid/ab_pr_0.5_0.5_8_1e-4` | `qwen/amid` final step directory |
  | `qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0` | `qwen/nnm` copy without projectors in `outputs/vllm_ckpt/qwen2.5-0.5B-it#sfkl_nnm/` |

- `report`: `run-outputs/out_amid-kd-8xh200.md`, written after the run ends.

# Upload

- `rclone_dest`: none given, and `rclone` is not installed on HGX47 (checked 2026-10-03). There is no upload; the artifacts stay on the server under `output_path`.
- To upload later, give a destination and install `rclone` or point to a configured binary. The upload then covers each phase's final `<save_path>/2492/`, `<save_path>/log.txt`, `<save_path>/eval/`, `outputs/eval_results/` and `runs/amid-kd-8xh200/`.

# Acceptance

## Expected artifacts

- The `gemma` and `qwen` driver `exit_code` files are 0, and all 5 `<phase>.exit_code` files are 0.
- Each phase passes `check_ckpt`, which writes `<phase>.ckpt`:
  - the logged `--save` equals the `save_path` above;
  - `--n-gpu 8`, and B_global = 64;
  - at least 2 numbered step directories;
  - the last one holds non-empty weights and `tokenizer_config.json`;
  - `log.txt` has at least 3 `dev | avg_loss` lines (epoch 0 plus 2 epochs).
- The data hashes in `runs/amid-kd-8xh200/<tag>/*.sha256` equal the `# Data` manifest.
- Eval smoke test, 2026-10-03 13:53–14:00, on GPUs 0–1 (idle, nothing else running), in a scratch directory that was deleted afterwards:
  - Stand-in checkpoints were built the way the trainer saves them: an untrained LoRA adapter on `google/gemma-2-2b-it` (`adapter_model.bin`), and `Qwen2.5-0.5B-Instruct` with 4 attached projectors saved by `save_pretrained(safe_serialization=False)`. The saved file did contain `projectors.0-3.weight`.
  - The real `scripts/eval/eval_2.sh` ran with `data_parallel_size=2` through ray. GSM8K ran with `--limit 4`, and the other 5 tasks were stubbed.
  - All 5 labels ended `DONE`, with exit code 0. `vllm_ckpt` dropped the 4 projector tensors, and vLLM loaded the copy. The Gemma revision pin and the LoRA adapter reached vLLM.
  - A second run skipped all 5 labels.
  - Not covered: the full task set, MBPP code execution, and 8-way data parallelism.
- `runs/amid-kd-8xh200/eval/exit_code` and `eval_2.exit_code` are 0. All 5 labels have a `DONE` marker and no `FAILED` file. Each label has a `results_*.json` for every one of the 6 tasks.

## Expected metrics

- No target numbers. The repo has no reference results for these teacher/student pairs on UltraInteract, and the AMiD paper reports other datasets and models.
- Report for each phase:
  - dev `rougeL` and `exact_match` at epochs 0, 1 and 2;
  - the final train `loss` / `ds_loss` (and `nnm_loss`);
  - the adaptive threshold from the `dev` lines;
  - the lm-eval table: GSM8K and GSM-Plus `exact_match` (strict and flexible), MATH `exact_match` / `math_verify`, MMLU-STEM `acc`, SciQ `acc` / `acc_norm`, MBPP `pass_at_1`, with their stderr.
- Sanity checks (a failure means investigate, not abort):
  - `total_iters` = 2,492 in every phase log;
  - the last step directory is `2492` (a mismatch is only logged as a warning);
  - loss is finite throughout;
  - dev `rougeL` at epoch 2 ≥ epoch 0 for each phase;
  - no lm-eval score at chance level or 0 for a whole task. That would point to a broken checkpoint load or chat template, not to the method.

## Failure conditions

- Fewer than 8 H200 visible, or any compute process still on the GPUs after 5 minutes → exit code 3 before the phase starts.
- A non-zero torchrun exit → the phase exit code is non-zero, the driver stops, and `run_qwen.sh` does not start.
- Training exits 0 but `check_ckpt` fails → exit code 4.
- Training exits 0 but a cached `main` ref moved away from its pinned SHA during the phase (`check_pins`) → exit code 5. Do not retry: report it, since the phase may have trained on another model revision.
- `run_eval.sh` exits 1 if a phase is missing its exit code 0 or its `.ckpt`, 3 at the GPU gate, and 1 if any lm-eval task failed. The failing task is in `outputs/eval_results/vllm/<label>/FAILED`, and the error is in `eval_2.log`.
- The pinned model SHA differs from `main` on the Hub, or a shard is missing → `hf_prepare` fails before any GPU use.
- Traceback, CUDA OOM, an NCCL error, `loss: nan/inf`, a full disk, or a dead torchrun worker in `<phase>.log`; the watcher greps for these.

## Retry policy

- Rerunning the same launch command is safe:
  - phases with exit code 0 and a valid checkpoint are skipped;
  - a failed phase restarts from step 0 (the code cannot resume);
  - before the restart, its partial `save_path` is renamed to `<save_path>.failed_<UTC time>`;
  - its old log becomes `<phase>.log.<UTC time>`.
- Retry a failed phase once without changes. If it fails again, stop and report.
- `run_eval.sh` is also safe to rerun: models with a `DONE` marker are skipped, and a model with a failed task reruns all 6 tasks.
- Within this `run_id`, do not change the dataset, seed, batch, objective, sampler or hyperparameters.
- A run with any changed parameter gets a new `run_id` and a new run file.

## Incident log

- 2026-10-03 19:08:37: `qwen/nnm` failed with exit 1 after the initial dev evaluation, on all 8 ranks, with `AttributeError: 'Qwen2Model' object has no attribute 'model'`.
  - The cause was `finetune.py:341`, which reached the decoder layers through `model.base_model.model.model.layers`. That path exists only for a PEFT-wrapped student. Phase 2 (Gemma) is LoRA, but phase 5 (Qwen) is a full fine-tune. The failure was deterministic, so the unchanged retry was skipped.
  - Fix, approved by the user: the hook setup now unwraps DeepSpeed, calls `get_base_model()` when the model is a `PeftModel`, then uses `.model.layers`.
    - A CPU test on a tiny Qwen2 config passed. For PEFT, the new path returns the same layer objects as the old one, so phase 2 is unaffected. For a full fine-tune, the old path fails as in the run and the new path returns the decoder layers.
    - No dataset, seed, batch, objective or hyperparameter changed.
    - `finetune.py` sha256 went from unrecorded to `74c90b0d66ad93841211e1c2e098042a6886e021a966850acc1b640f3bb289da`.
    - Phase 2 ran with the old `finetune.py`, which differs only at this line.
  - 19:13:18: relaunched `bash run_qwen.sh && bash run_eval.sh` (driver pid 2472155). `csd` and `amid` were skipped as done. The failed attempt is kept as `…nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0.failed_20261003T121318Z`.

## Monitoring

- The Mac watcher is the command in `# Commands`. It refreshes every `INTERVAL` seconds and shows:
  - each phase's status, with a progress bar, step/total, epoch, loss, lr, elapsed time, ETA and the latest `dev` line;
  - recent error lines;
  - every GPU's utilization, VRAM, temperature and power, and the owner of each GPU process;
  - the eval driver: models done/failed out of 5, the current model and task, and recent failure lines;
  - free space on `/nvme`.
- While the run is active, a read-only health check runs every 20 minutes. It reads PID and exit state, the optimizer step, finite loss, log and checkpoint modification times, disk, and the GPU table, and scans for the failure patterns above. A stall is never inferred from one idle sample.

## Known limitations (must appear in the report)

- Single seed (10). Results are not averaged over seeds.
- The validation set is drawn from the training records, so the 200 dev records are also trained on. Dev ROUGE-L measures fit, not held-out generalisation.
- The final checkpoint is used as-is (the last epoch). The README selects checkpoints by ROUGE-L; with only two epoch checkpoints, that means comparing `1246/` against `2492/` by hand.
- Moving from 2 to 8 GPUs changes per-rank state, although B_global is unchanged:
  - each rank has its own replay buffer (`deque(maxlen=1000)`), so there are 8 buffers instead of 2;
  - each rank sees 1/8 of each epoch.
- For Qwen, the micro-batch also changes from 16 to 8, so the per-micro-batch loss averages over fewer sequences.
- Numbers are not comparable to an earlier 2-GPU run without disclosing these points.
- `gemma/csd` is not run (it stays commented out), so Gemma has no CSD baseline in this run.
- The Qwen student (Qwen2.5-0.5B) is trained on data tokenised with the teacher's (Qwen3-4B-Instruct-2507) tokenizer and chat template. The Gemma pair shares one tokenizer.
- The UltraInteract-Infer dataset was last modified 2026-10-02. It is pinned to `3c2fb0d`, so later upstream changes do not affect this run.
- lm-eval runs on the final checkpoint (`2492/`) only, not on the epoch-1 checkpoint (`1246/`).
- Gemma CSD is not trained, so it is not evaluated either. The original `eval_2.sh` listed it with a `_8_` path that this run does not produce.
- The lm-eval benchmark datasets are not pinned to a revision. The report must cite lm_eval 0.4.12 and the task versions and `n-shot` values from each `results_*.json`.
- `eval_2.sh` modifies the installed `lm_eval/api/task.py` (the `hendrycks_math` assertion patch). Scores from an unpatched lm_eval may differ on MATH.
- MBPP runs model-written code on the server with no sandbox beyond lm-eval's own timeout.
- The labels in the original `eval_2.sh` used other paths (`_8_` for Gemma, `qwen2.5-0.5-it` for Qwen). They were changed to the paths this run writes.
- No lm-eval baseline exists for the untrained students or the teachers in this run, so the table shows the methods relative to each other only.

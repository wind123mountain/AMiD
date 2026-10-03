# Run report: `amid-kd-8xh200`

- **Run id**: `amid-kd-8xh200`
- **Server**: HGX47 (`vt-admin`, 8× H200, bare metal, driver 595.71.05)
- **Run root**: `/nvme/annp36-home/work/AMiD`
- **Run input**: [run-inputs/amid-kd-8xh200.md](../run-inputs/amid-kd-8xh200.md)
- **Source revision**: `4b6d2c0+dirty` (the modified files and their sha256 are listed in the run input)
- **Status**: completed.
  - Every phase and every driver exited 0. The last line of the eval log is "Eval Done! (failed models: 0)".
  - One phase (`qwen/nnm`) failed once, was fixed with the user's approval, and was rerun. See the Limitations section.
  - Gates:
    - `total_iters` = 2,492 in every phase, and the last step directory is `2492/`: passed.
    - B_global = 64 in every phase: passed.
    - Loss finite throughout (0 nan/inf lines in any `log.txt`): passed.
    - Dev rougeL at epoch 2 ≥ epoch 0: passed in all 5 phases.
    - "No lm-eval score at chance level or 0 for a whole task": **failed**. Gemma MBPP is 0.0 for both models, and several Qwen scores are 0 or below chance. See Main Results.
- **Timezone**: all times are Vietnam time (UTC+7), converted from the server's UTC logs.

| step | start | end | elapsed | exit |
|---|---|---|---|---|
| `gemma/amid` | 2026-10-03 14:00:53 | 14:49:29 | 48 min 36 s | 0 |
| `gemma/nnm` | 14:49:29 | 15:39:20 | 49 min 51 s | 0 |
| `qwen/csd` | 15:39:23 | 17:22:08 | 1 h 42 min 45 s | 0 |
| `qwen/amid` | 17:22:08 | 19:06:04 | 1 h 43 min 56 s | 0 |
| `qwen/nnm`, attempt 1 | 19:06:04 | 19:08:37 | 2 min 33 s | 1 (`AttributeError`) |
| `qwen/nnm`, rerun | 19:13:18 | 19:39:52 | 26 min 34 s | 0 |
| lm-eval (5 models × 6 tasks) | 19:39:52 | 21:30:09 | 1 h 50 min 17 s | 0 |
| **whole chain** (launch at 14:00:49) | 14:00:49 | 21:30:09 | **7 h 29 min 20 s** | 0 |

Time per model in lm-eval:

| model | time |
|---|---|
| Gemma AMiD | 18.6 min |
| Gemma NNM | 19.3 min |
| Qwen CSD | 23.5 min |
| Qwen AMiD | 24.5 min |
| Qwen NNM | 24.4 min |

## Settings

No paper reference exists for these teacher/student pairs on UltraInteract, so this report has no comparison with a paper.

- **Data**: `VoCuc/UltraInteract-Infer@3c2fb0d397d3ef2d505f3fe48ad4e30f30733895`.
  - Each student trains on its own teacher's generated responses (79,751 training records).
  - Dev set: 200 records sampled from the training records with `random.seed(42)`.
- **Models**:

  | role | model | revision |
  |---|---|---|
  | Gemma student | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
  | Gemma teacher | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
  | Qwen student | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
  | Qwen teacher | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` |

- **Phases**:

  | phase | trainer | `--type` | method parameters | params trained | B_global = GPU × per device × grad_acc |
  |---|---|---|---|---|---|
  | `gemma/amid` | `base_finetune.py` | `adaptive-amid` | `ab`/`pr`, α=0.5, λ=0.5 | LoRA r16 / α32 / dropout 0.05 | 8 × 4 × 2 = 64 |
  | `gemma/nnm` | `finetune.py` | `adaptive-sfkl` | skew α=0.05, NNM ratio 0.2, K=128, 4 layers, d'=256, `--delta-threshold 0.03` | LoRA r16 | 8 × 4 × 2 = 64 |
  | `qwen/csd` | `base_finetune.py` | `adaptive-csd` | `ab`/`pr`, α=0.5, λ=0.5 | full fine-tune | 8 × 8 × 1 = 64 |
  | `qwen/amid` | `base_finetune.py` | `adaptive-amid` | `ab`/`pr`, α=0.5, λ=0.5 | full fine-tune | 8 × 8 × 1 = 64 |
  | `qwen/nnm` | `finetune.py` | `adaptive-sfkl` | same as `gemma/nnm` | full fine-tune | 8 × 8 × 1 = 64 |

- **Common settings**:
  - **Optimizer and schedule**: lr 1e-4, cosine decay, no warmup, weight decay 1e-2, gradient clip 1.0, `--kd-ratio 1.0`, seed 10.
  - **Sequence lengths**: `--max-length 1025`, `--max-prompt-length 512`.
  - **Steps**: 2 epochs × 1,246 = 2,492 optimizer steps.
  - **Student generation**: adaptive threshold starting at 0.0, `--loss-eps 0.1`. The threshold step is 0.1 for CSD and AMiD and 0.03 for NNM.
  - **Replay buffer**: capacity 1000 per rank. The NNM phases also use `--replay-ratio decreasing --mixed-alpha 0.5`.
  - **Precision**: teacher in fp16; DeepSpeed bf16 with `configs/deepspeed/ds_config_zero0_bf16.json`.
  - **Sampling**: `top_k 0`, `top_p 1.0`, temperature 1.0.
- **Validation protocol**:
  - **Dev**: runs before training and after each epoch on the 200 dev records, giving avg_loss, exact_match and rougeL.
  - **lm-eval**: `scripts/eval/eval_2.sh` (sha256 `29c5b00e43c321633c4db44981f1c695d88ae737c4d90dcbfef2a01106cc61bc`) runs lm-eval 0.4.12 on vLLM 0.17.1 with `data_parallel_size=8`, on the final checkpoint `2492/` only.
    - GSM8K, GSM-Plus: chat template, multi-turn few-shot, up to 5,120 new tokens.
    - MATH (`minerva_math`, 4-shot), MMLU-STEM (5-shot), SciQ: chat template.
    - MBPP: 3-shot, no chat template, temperature 0.
  - **How each model is loaded for lm-eval**:
    - Gemma: base model plus LoRA adapter (`lora_local_path`, `max_lora_rank=32`).
    - Qwen NNM: a copy with the projectors stripped (`outputs/vllm_ckpt/qwen2.5-0.5B-it#sfkl_nnm/`).

## Main Results

### Dev set (200 records drawn from the training data)

| phase | avg_loss: start → epoch 1 → epoch 2 | rougeL: start → epoch 1 → epoch 2 | exact_match | final threshold |
|---|---|---|---|---|
| `gemma/amid` | 2.898 → 2.781 → 2.781 | 0.55 → 1.41 → 0.66 | 0.0 at all 3 | 0.0 |
| `gemma/nnm` | 2.875 → 2.734 → 2.734 | 0.41 → 2.84 → 0.87 | 0.0 at all 3 | 0.0 |
| `qwen/csd` | 1.867 → 3.719 → 3.469 | 5.48 → 6.29 → 7.17 | 0.0 at all 3 | 0.1 |
| `qwen/amid` | 1.867 → 2.438 → 2.359 | 5.48 → 9.57 → 7.04 | 0.0 at all 3 | 0.1 |
| `qwen/nnm` | 1.867 → 2.563 → 2.516 | 5.57 → 8.10 → 8.45 | 0.0 at all 3 | 0.03 |

The last train-log line of each phase (global iter 2490):

| phase | loss | ds_loss | nnm_loss | step time |
|---|---|---|---|---|
| `gemma/amid` | 0.0522 | 0.0522 | — | 1.087 s |
| `gemma/nnm` | 1.1664 | 1.1664 | 0.0000 | 1.096 s |
| `qwen/csd` | −0.0093 | −0.0093 | — | 4.172 s |
| `qwen/amid` | 0.0825 | 0.0825 | — | 4.298 s |
| `qwen/nnm` | 1.6600 | 1.6599 | 0.0005 | 2.063 s |

Train losses use different objectives, so they cannot be compared across methods.

### lm-eval benchmarks (final checkpoint `2492/`)

Each cell is the value ± its stderr. **No baseline was evaluated** (neither the untrained students nor the teachers), so these numbers cannot be read as a gain or a loss from distillation.

**Math** (strict-match / flexible-extract; MATH: exact_match / math_verify):

| model | GSM8K (n=1319, 5-shot) | MATH (n=5000, 4-shot) | GSM-Plus (n=10552, 5-shot) |
|---|---|---|---|
| Gemma AMiD | 0.0599 ± 0.0065 / **0.4936 ± 0.0138** | 0.0256 ± 0.0022 / **0.2456 ± 0.0058** | 0.0202 ± 0.0014 / **0.3356 ± 0.0046** |
| Gemma NNM | 0.0447 ± 0.0057 / 0.4882 ± 0.0138 | 0.0236 ± 0.0021 / 0.2432 ± 0.0057 | 0.0172 ± 0.0013 / 0.3349 ± 0.0046 |
| Qwen CSD | 0.0083 ± 0.0025 / 0.0174 ± 0.0036 | 0.0008 ± 0.0004 / 0.0160 ± 0.0018 | 0.0028 ± 0.0005 / 0.0100 ± 0.0010 |
| Qwen AMiD | 0.0182 ± 0.0037 / **0.0455 ± 0.0057** | 0.0036 ± 0.0008 / **0.0370 ± 0.0027** | 0.0045 ± 0.0006 / **0.0226 ± 0.0014** |
| Qwen NNM | 0.0 / 0.0106 ± 0.0028 | 0.0 / 0.0168 ± 0.0018 | 0.0 / 0.0088 ± 0.0009 |

**Knowledge and code**:

| model | MMLU-STEM acc (n=3153, 5-shot) | SciQ acc / acc_norm (n=1000, 0-shot) | MBPP pass@1 (n=500, 3-shot) |
|---|---|---|---|
| Gemma AMiD | 0.4754 ± 0.0086 | 0.908 ± 0.0091 / 0.756 ± 0.0136 | 0.0 ⚠ invalid |
| Gemma NNM | **0.4821 ± 0.0086** | **0.911 ± 0.0090** / **0.782 ± 0.0131** | 0.0 ⚠ invalid |
| Qwen CSD | 0.2223 ± 0.0074 | 0.677 ± 0.0148 / 0.590 ± 0.0156 | 0.014 ± 0.0053 |
| Qwen AMiD | **0.2290 ± 0.0075** | 0.664 ± 0.0149 / 0.554 ± 0.0157 | **0.106 ± 0.0138** |
| Qwen NNM | 0.2125 ± 0.0073 | **0.704 ± 0.0144** / **0.633 ± 0.0152** | 0.096 ± 0.0132 |

Bold marks the best value within each student family.

How to read some of these numbers:
- **Gemma MBPP 0.0 is a pipeline artifact, not a model result.** In the sample files, 499 of 500 answers contain the SentencePiece character `▁` (U+2581) in place of indentation spaces, for both models. 349 (AMiD) and 225 (NNM) answers also contain `\r`. So no program can run.
- **Gemma GSM8K and GSM-Plus strict-match are a format issue.** Gemma writes answers such as `$<<9*2=18>>18` without the `#### <n>` marker that strict-match looks for. Flexible-extract is the usable number.
- **Some Qwen scores are 0 or below chance**: Qwen NNM strict-match is 0.0 on GSM8K, GSM-Plus and MATH exact_match, and all three Qwen models score below 0.25 on MMLU-STEM, which is the chance level for 4 choices. See Insights.

## Insights

**The three Qwen students degraded during training, while the Gemma students did not.**
- Qwen dev avg_loss rose from 1.867 before training to 2.36–3.47 at the end, in every Qwen phase:
  - CSD: 3.469
  - AMiD: 2.359
  - NNM: 2.516
- Gemma dev avg_loss fell, from 2.898 to 2.781 (AMiD) and from 2.875 to 2.734 (NNM).
- In lm-eval, the Qwen models score:
  - 0.2125–0.2290 on MMLU-STEM, below the 4-choice chance level of 0.25;
  - at most 0.0455 on GSM8K flexible-extract.
- The samples look like a broken model, not a broken parser:
  - a Qwen AMiD GSM8K answer I inspected is a repetition loop ("she sells 16 eggs per day…");
  - a Qwen AMiD MBPP answer I inspected is just `pass`.
- I checked that the saved Qwen tokenizers give the same token ids and chat template as `Qwen/Qwen2.5-0.5B-Instruct`, so the tokenizer does not explain this.
- A full fine-tune at lr 1e-4 on a 0.5B model is the likely cause. The Gemma students use LoRA at the same lr. This is not confirmed, because the untrained Qwen student was not evaluated.

**Among the Qwen students, AMiD scores highest on 5 of the 6 tasks, but every Qwen score is far too low to rank the methods.**
- Qwen AMiD has the highest:
  - GSM8K flexible-extract (0.0455, vs 0.0174 for CSD and 0.0106 for NNM);
  - MATH math_verify (0.037);
  - GSM-Plus flexible-extract (0.0226);
  - MBPP (0.106).
- Qwen NNM has the highest SciQ (0.704 / 0.633).
- The differences between the methods are small:
  - MMLU-STEM differences are within about 2 stderr (0.2223 / 0.2290 / 0.2125, stderr about 0.0075);
  - MBPP for AMiD vs NNM (0.106 vs 0.096) is within 1 stderr.
- Only Qwen CSD is clearly lower on MBPP (0.014).

**Gemma NNM and Gemma AMiD are statistically tied on the valid benchmarks.**
- GSM8K flexible-extract: 0.4936 (AMiD) vs 0.4882 (NNM), stderr 0.0138.
- MATH math_verify: 0.2456 vs 0.2432.
- GSM-Plus: 0.3356 vs 0.3349.
- MMLU-STEM: 0.4754 vs 0.4821, stderr 0.0086.
- SciQ acc_norm: NNM leads, 0.782 vs 0.756 (stderr about 0.013, a gap of about 2 stderr). This is the largest gap.
- On dev, NNM has a slightly lower avg_loss (2.734 vs 2.781) and a higher rougeL at every point.

**The adaptive student-generation threshold explains why the small Qwen phases ran about twice as long as the Gemma phases.**
- After epoch 0, Qwen dev loss rose by more than `--loss-eps 0.1`, so the threshold rose:
  - to 0.1 in CSD and AMiD;
  - to 0.03 in NNM, which uses `--delta-threshold 0.03`.
- The student then generated on part of the steps in epoch 1:
  - CSD and AMiD: step time went from 0.37–0.44 s in epoch 0 to about 4–7 s in epoch 1 (4.17 s and 4.30 s on the last log line);
  - NNM: only 2.06 s at the end, because of its smaller threshold.
- So Qwen CSD and AMiD took about 1 h 43 min each, against 49–50 min for each Gemma phase.
- Gemma dev loss never rose, so its threshold stayed 0.0 and its step time stayed flat at about 1.09 s.
- This is the scripted algorithm, not a fault.

**The NNM regulariser contributes almost nothing by the end of training.**
- The final `nnm_loss` is 0.0000 (Gemma) and 0.0005 (Qwen), while ds_loss is 1.1664 and 1.6599.
- The learned projectors drive the nuclear-norm log-ratio term to about 0. In the final epoch, NNM therefore behaves almost like plain skew-forward-KL with a replay buffer.
- Source: `<save_path>/log.txt` of the two NNM phases.

**The final checkpoint is not the best by dev rougeL in 3 of the 5 phases.**
- Gemma rougeL peaks after epoch 1 and drops at epoch 2:
  - AMiD: 1.41 → 0.66;
  - NNM: 2.84 → 0.87.
- Qwen AMiD also drops, from 9.57 to 7.04.
- lm-eval used only `2492/`. The `1246/` checkpoints are kept on the server.
- Dev exact_match is 0.0 everywhere. The dev set also overlaps the training data, so it measures fit, not generalisation.

## Artifacts

All paths are relative to `/nvme/annp36-home/work/AMiD` on HGX47.

| phase | final checkpoint (`<save_path>/2492/`) | weight file sha256 | dir size |
|---|---|---|---|
| `gemma/amid` | `results/gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4/2492` | `adapter_model.bin` `2cfc8c62c5ef20c7dc800c8b44ab12bb063d056fdea10426ec4926bcfd7c1ffb` | 155M |
| `gemma/nnm` | `results/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492` | `adapter_model.bin` `1fab427fa8ea03c241f46853272ae4ec8fbd09cfae23b2c19729f98c5065b13d` | 155M |
| `qwen/csd` | `results/qwen2.5-0.5B-it#csd/ab_pr_0.5_0.5_8_1e-4/2492` | `pytorch_model.bin` `763565328f757856c2be83dd32f643511ddc09dcbc682ddf900d4f2a10c791e6` | 1.9G |
| `qwen/amid` | `results/qwen2.5-0.5B-it#amid/ab_pr_0.5_0.5_8_1e-4/2492` | `pytorch_model.bin` `c6f44cf1895cba2270e5463ac0bf82b60b8f66dfa3476705dcebe3fd753684be` | 1.9G |
| `qwen/nnm` | `results/qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492` | `pytorch_model.bin` `1df3773db02930e30893e0468cfe8ce871ac897c5784352b1f353fc0295d88b3` | 2.0G |

Dir sizes are for the whole `<save_path>`.

- **Failed attempt of `qwen/nnm`**: kept for diagnosis as `results/qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0.failed_20261003T121318Z`.
- **Training metrics**: `<save_path>/log.txt`, with dev answers in `<save_path>/eval/<epoch>/answers.jsonl`.
- **Logs and state**: `runs/amid-kd-8xh200/{gemma,qwen,eval}/` (4.4M), containing:
  - `driver.log` and per-phase `<phase>.log`;
  - `exit_code` and `<phase>.exit_code`;
  - `<phase>.ckpt`;
  - `data.sha256` and `processed_data.sha256`.
- **lm-eval outputs**: `outputs/eval_results/` (2.5G), containing:
  - `vllm/<label>/` with 30 `results_*.json` (6 per model) and 150 `samples_*.jsonl`;
  - per-model logs in `logs/<label>.log`;
  - the whole eval log in `runs/amid-kd-8xh200/eval/eval_2.log`.

  The sha256 of all 30 results JSON files, concatenated in sorted path order, is `1fa4fe87a277c6b8f6c8dfbaff5219d7caaf0d03f1a3ce820a600bc3104b9559`.
- **vLLM copy of Qwen NNM** (projectors stripped): `outputs/vllm_ckpt/` (958M).
- **Disk**: `/nvme` has 6.2T free.
- **GPU release**: all 8 GPUs are at 0% utilisation and 0 MiB. There are no compute apps and no leftover `torchrun`, `ray`, `vllm` or `lm_eval` processes.
- **Upload**: none. The run input gives no rclone destination, and `rclone` is not installed on the server.

## Limitations

- **Single seed (10), final checkpoint only.** The results are not averaged over seeds, and the `1246/` checkpoints were not evaluated with lm-eval, although dev rougeL favours them in 3 of the 5 phases.
- **No baseline.** Neither the untrained students nor the teachers were evaluated, so no number in this report shows a gain or a loss from distillation.
- **The Gemma pair has no CSD phase.** `gemma/csd` stays commented out in `run_gemma.sh`.
- **`finetune.py` was changed during the run**, with the user's approval.
  - Cause: the first `qwen/nnm` attempt failed with `AttributeError`. The NNM hook walked `model.base_model.model.model.layers`, which exists only for a PEFT (LoRA) model, and `qwen/nnm` is a full fine-tune.
  - Fix: the hook now resolves the layers through `get_unwrapped_student(model)` (plus `get_base_model()` for PEFT). This does not change the Gemma path, but `gemma/nnm` ran on the code before the fix.
  - New sha256 of `finetune.py`: `74c90b0d66ad93841211e1c2e098042a6886e021a966850acc1b640f3bb289da`.
  - Details are in the run input's Incident log.
  - The other `finetune_*.py` variants have the same bug, but this run does not use them.
- **The Gemma MBPP scores (0.0) are invalid.** The generated code contains `▁` and `\r` in place of spaces (likely a detokenisation issue in the vLLM + LoRA path, not yet confirmed), so it says nothing about the models.
- **Gemma strict-match is a format issue.** The answers lack the `####` marker, so GSM8K and GSM-Plus strict-match understate Gemma.
- **The dev set overlaps the training data**, and dev exact_match is 0.0 everywhere.
- **Qwen tokenizer warning.** transformers 4.57.3 prints an "incorrect regex pattern" warning for the Qwen tokenizer. I checked it: the token ids and chat template are identical to the hub tokenizer.
- **Tokenizer mismatch for Qwen.** Qwen data is tokenised with the teacher's (Qwen3-4B) tokenizer and chat template, while the student is Qwen2.5-0.5B.
- **Not pinned:** the lm-eval benchmark datasets are loaded from the Hub through lm-eval's task configs, without a pinned revision.
- **Changes from the earlier 2-GPU setup** (B_global is still 64):
  - 8 per-rank replay buffers instead of 2;
  - each rank sees 1/8 of each epoch;
  - the Qwen micro-batch is 8 instead of 16.

  These numbers are therefore not directly comparable to a 2-GPU run.

## Next Steps (not run)

- Evaluate the untrained students (and, if budget allows, the teachers) with the same `eval_2.sh`, to get a baseline.
- Retrain the Qwen students with a lower lr (for example 1e-5) or with LoRA, as for Gemma. Then check whether the collapse goes away.
- Run lm-eval on the `1246/` checkpoints, at least for Gemma AMiD, Gemma NNM and Qwen AMiD, where dev rougeL peaked at epoch 1.
- Fix Gemma MBPP detokenisation (the `▁` and `\r` in the vLLM LoRA output), then re-score MBPP from the saved samples or rerun it.
- Optionally merge the 3 chat-template tasks with identical arguments into one `lm_eval` call per model, to save about 2–3 min of vLLM start-up. The user declined this for now.
- Upload the artifacts listed in the run input once an rclone destination is given and `rclone` is installed.

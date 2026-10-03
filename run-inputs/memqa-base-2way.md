# Project

- `name`: `ACL-ECHO`
- `run_id`: `memqa-base-2way`
- `local_path`: `/Users/savoxism/Documents/GitHub/ACL-ECHO`
- `ssh_host`: `H200_ANNP36`
- `remote_path`: `/home/annp36/work/ACL-ECHO`
- `source_revision`: `9d56ffe` plus uncommitted working tree (mirrored by tar, file hashes checked on both sides). Commit before the run so the report can cite a clean revision.
- `entrypoint`: `scripts/memqa/run_memqa.sh`

# Schedule

- `start_at`: not scheduled; start manually after the preflight commands below pass. Actual: 2026-09-29 23:09:20 → 2026-09-30 00:44:25 (1 h 35 min; see `run-outputs/out_memqa-base-2way.md`).
- `timezone`: Vietnam time (UTC+7) for every time in this file. The server (`annp36-dev-0`) logs in UTC; times are converted.
- `max_runtime`: estimate of 6 h for 2 generation phases and 2 judge phases. It is extrapolated from the 4-question sample (112 s on one GPU for 2 LongMemEval + 2 LoCoMo questions × 8 samples), not measured, and not enforced by the script.

# Data

- `input` / `benchmark`: `benchmarks/locomo` (LoCoMo-10) and `benchmarks/longmemeval` (LongMemEval-S), used as they are in the repo.
- `train`: none. No memory-QA data is used in training; this is a zero-shot baseline.
- `validation`: none.
- `manifest` (sha256):
  - `benchmarks/locomo/locomo_questions.json`: `bb8b956ada29715226d61b5e645cea89504a3a2a7c02088d9db9fd6f48bed8bd`
  - `benchmarks/longmemeval/longmemeval_questions.json`: `9614a8fb4ded9e824c2bbd352cbebaba9e8e3a36a8e5fcfc573f4e883301a524`
  - Rendered prompts: `runs/memqa-base-2way/prompts/SHA256SUMS`, written by the `prompts` phase.
- LoCoMo: 10 conversations (median about 29 sessions) and 1,982 questions.
  - single_hop 841, adversarial 446, temporal_reasoning 321, multi_hop 282, open_domain 92.
  - The adversarial questions are abstention questions with GT `No information available`.
  - Source is `snap-research/locomo@3eb6f2c`. The repo's `annotation_issues` flag 5 questions; they are kept and scored as given.
- LongMemEval-S: 500 questions, each with its own haystack (median 48, max 62 sessions).
  - multi-session 133, temporal-reasoning 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30.
  - 30 questions are abstentions.
  - `metadata.json` names `longmemeval_oracle.json` (`xiaowu0162/longmemeval-cleaned@98d7416`) as the source. However, the haystacks are S-sized (median about 107K tokens), not oracle-sized. Treat the data as LongMemEval-S and report the inconsistent metadata.

## Prompt rendering (`scripts/memqa/build_prompts.py`)

Every model gets the same prompt, so the baselines compare directly with LoongRL-trained checkpoints later.

1. **Template**: `qwen_orz_boxed` from `verl/utils/chat_template.py`, the prompt LoongRL trains and evaluates with (paper Fig. 5).
   - The system prompt asks the model to reason in `<think>…</think>` and answer in `\boxed{}`.
   - The assistant turn is prefilled with `<think>`.
   - The template is applied with each model's own tokenizer.
   - For Qwen3 this template replaces its native `enable_thinking` template. Qwen3 therefore reasons inside the same `<think>` prefill, and its native chat format is not tested here.
2. **Layout**: the history comes first and the question last.
   - This matches LoongRL training prompts, where the passages come before `Question:`.
   - It also lets vLLM prefix caching reuse one LoCoMo history across about 200 questions.
   ```
   The following is the conversation history between {A} and {B}, organized into sessions in chronological order.

   ### Session {id} ({session date})
   {Role}: {text} [shared a photo: {blip_caption}]
   ...

   Current date: {question_date}
   In the question, "I", "me" and "my" refer to the User.      # LongMemEval only
   Question: {question}

   {INSTRUCTIONS}
   ```
3. **Content choices**:
   - Every session header carries its date, and the prompt includes the question's current date. Temporal questions (321 in LoCoMo, 133 in LongMemEval) cannot be answered without them.
   - Image captions are inlined, because some LoCoMo answers depend on them (1,226 messages have captions).
   - `session_summary` is left out, because it could leak answers.
   - Role names are capitalised (`User:` / `Assistant:`).
4. **INSTRUCTIONS**:
   - Answer only from the history.
   - Relative times such as "yesterday" are relative to the session date.
   - If the history is insufficient, answer `No information available`.
   - Write dates as `7 May 2023` (or `May 2023` / `2023`).
   - Put only the short final answer inside `\boxed{}`.
   - The date format follows how the benchmark writes its GTs. That formatting hint is a deliberate evaluation choice and must be disclosed.
5. **Ground truth**: a list `[GT_answer]`. For abstention questions whose GT differs from it, `No information available` is appended as an alias.
6. **Budget**:
   - Prompt budget is `max_model_len − max_tokens` = 131,072 − 10,240 = 120,832 tokens.
   - If a prompt is over budget, whole sessions are dropped from the oldest end, and the drop is recorded per question (`dropped_sessions`, `gt_session_dropped`).
   - Precheck (prompt sets rendered on CPU, `runs/memqa-base-2way/precheck/token_stats.txt`):

   | benchmark | median prompt tokens | max prompt tokens | truncated | GT session dropped |
   |---|---|---|---|---|
   | LoCoMo | 22,586 | 24,808 | 0 | 0 |
   | LongMemEval-S | 106,696 | 111,108 | 0 | 0 |

   - LoCoMo's longest prompt (24.8K) plus 10,240 output tokens exceeds 32K. YaRN is therefore used for both benchmarks, so each model runs under one fixed configuration.

# Model

Scope is Qwen3-8B only; the Qwen2.5 family is paused. Weights are already in the global HF cache (`~/.cache/huggingface/hub`), so nothing needs to be downloaded. "Base" here means "before our RL".

| role | tag | repo_id | revision (snapshot) |
|---|---|---|---|
| evaluated model | `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| judge | — | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |

- `model_path`: `runs/memqa-base-2way/models/<tag>`. This is a model view: symlinks to the pinned snapshot plus a rewritten `config.json` with `rope_scaling = {rope_type: yarn, factor: 4.0, original_max_position_embeddings: 32768}`, which gives a 131,072-token context. This is the YaRN setting LoongRL uses at eval time. The HF cache is never modified.
  - Qwen3-8B's own config has `max_position_embeddings: 40960`; the YaRN view above is the long-context setting Qwen documents for this model.
  - The judge uses the raw snapshot (`cache_path` in the HF cache), with no YaRN, because its inputs are short.
  - `Qwen/Qwen2.5-{7B,14B}-Instruct` stay in the script's snapshot map. `MODELS_OVERRIDE` can run them later under a new `run_id`.
  - Snapshot paths are used directly because some caches (Qwen3-8B, Qwen2.5-14B) have no `refs/main`. `models` checks that every shard in `model.safetensors.index.json` exists.
  - Runtime: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`.
- `task`: long-context memory QA (inference only, no training). For each benchmark, sample 8 responses per question from Qwen3-8B.
- `method`: score the responses with the two-way substring reward used in LoongRL training (primary), then judge the same saved responses with Qwen3-32B (secondary). That gives 2 generation phases and 2 judge phases.
- `seed`: `20260929`; each question also gets its own seed, `sha256("{seed}:{question_id}")[:8]`, so its samples do not depend on which shard or GPU runs it.
- `batch sizes`: 8 samples per question (`n=8`); vLLM schedules requests by continuous batching (no fixed batch size).
- `epochs`: one pass over each benchmark.

## Generation (`scripts/memqa/generate_vllm.py`, vLLM 0.28.0)

| parameter | value | reason |
|---|---|---|
| `n` | 8 | LoongRL reports the average pass@1 over 8 samples |
| `temperature` / `top_p` | 0.6 / 0.95 | LoongRL training and eval sampling |
| `top_k` | -1 | LoongRL sampling; Qwen3's own card recommends 20, a known deviation |
| `max_tokens` | 10,240 | LoongRL eval output length |
| `max_model_len` | 131,072 | YaRN ×4 |
| `seed` | `20260929` | run seed, plus the per-question seed above |
| `enable_prefix_caching` | true | a shared history is prefilled once per shard |
| `gpu_memory_utilization` | 0.9 | one model per GPU with TP=1; the sample showed 108 GiB of KV cache (787K tokens, 6 concurrent 131K requests) |

- The prompt is passed as token ids, and each shard asserts that its length equals the token count recorded at build time.
- Only the last `\boxed{}` in the response is scored (`last_boxed_only_string`), so reasoning text around it is ignored. A response with no `\boxed{}` scores 0.

## Grader (`scripts/memqa/score.py`, primary)

- It calls `compute_score` from `verl/utils/reward_score.py` by file path. This is the same code as the RL reward, with the same env:
  - `REWARD_CALC_TYPE=pure_exact_match`
  - `REWARD_TWO_WAY=1`
  - `REWARD_MIN_PRED_CHARS=2`
  - `REWARD_MIN_PRED_TOKEN_RATIO=0.0`
  - `REWARD_RELATIVE_TIME_ONE_WAY=1`
- **Primary metric**: `avg@8`, the mean over questions of the fraction of samples scored 1. A sample scores 1 when the normalised last `\boxed{}` answer `a` and ground truth `y` satisfy `y ⊆ a` or `a ⊆ y`, with a token-boundary match on the `a ⊆ y` side. That side also rejects:
  - empty predictions or predictions under 2 characters;
  - predictions made only of stop words or uninformative words;
  - any prediction when the GT is a relative time expression (contains `before`, `after`, `ago`, `earlier`, `later`, `prior`, `week(s)`, `weekend(s)`). Such GTs contain their anchor date, so a prediction naming only the anchor (`25 May 2023` for `The sunday before 25 May 2023`) would otherwise score 1; the pre-run sample showed exactly this. The guard applies to 108/1,982 LoCoMo GTs (97 temporal_reasoning) and 31/500 LongMemEval GTs (21 temporal-reasoning; the rest are durations such as `4 weeks` and rubric text).
- Reported in each metrics file:
  - overall, by question type, non-abstention, abstention, and overall excluding `single-session-preference`;
  - raw counts (`correct_samples/samples`);
  - reference-only numbers: `pass@8` and the one-way (`y ⊆ a`) score;
  - diagnostics: boxed rate, rate of samples truncated by length, response and prompt token statistics, and dropped-session counts.

## Judge (`scripts/memqa/judge.py`, secondary)

- Judge model: `Qwen/Qwen3-32B@9216db5`, bf16, one copy per GPU (TP=1), with pairs split round-robin across 8 shards.
- Decoding: greedy (`temperature=0`), `max_tokens=32`, Qwen3 thinking disabled (`enable_thinking=False`), `max_model_len=8192`.
- The judge sees the question, the benchmark's original `GT_answer` (without the abstention alias), and the model's final answer. The final answer is the text after the last `</think>` (a short explanation plus `\boxed{}`), cut to its last 8,000 characters.
  - The judge never sees the reasoning, so it cannot credit an answer the model only considered.
  - A response with neither `</think>` nor `\boxed{}` scores 0 without a judge call.
  - Identical (question, final answer) pairs are judged once and the verdict is reused.
- Prompt (user-provided, used verbatim):
  - system: `You are an impartial judge evaluating the quality of an AI assistant's answer to a question. Compare the model's response against the reference answer. Focus on factual correctness and equivalence.`
  - user: the `[[yes]]`/`[[no]]` template with `[User Question]`, `[The Start of Reference Answer]` … `[The End of Reference Answer]`, and `[The Start of Model's Response]` … `[The End of Model's Response]` blocks, ending with `Is the model response correct? Answer [[yes]] or [[no]] only.`. The full text is in `JUDGE_PROMPT_TEMPLATE` in `scripts/memqa/judge.py`.
- Verdict: the last `[[yes]]` or `[[no]]` in the output. Anything else is `invalid`, counted as no, and reported as `invalid_rate`.
- Metrics (`metrics/<tag>_<bench>.judge.json`):
  - judge `avg@8` and `pass@8`, overall, by type, non-abstention and abstention;
  - per-sample agreement with two-way (`both_correct`, `both_wrong`, `judge_only`, `substring_only`, `agreement_rate`);
  - diagnostics: unique pairs judged, samples without an answer, verdict counts.

## GPU policy

- All 8 H200 are used, one vLLM process per GPU (`CUDA_VISIBLE_DEVICES=k`, TP=1), with prompts split round-robin over the list sorted by conversation.
- Phases run one after another: generation LoCoMo → generation LongMemEval → two-way scoring (CPU) → judge LoCoMo → judge LongMemEval.
- Before `check` and before every phase, `require_idle_gpus` aborts with exit code 3 if any compute process is on the GPUs or fewer than 8 H200 are visible. The run never shares GPUs with another job.

# Commands

```bash
cd /home/annp36/work/ACL-ECHO

# preflight (no GPU use except the check phase)
bash scripts/memqa/run_memqa.sh setup      # project-local .venv-bench with vllm==0.28.0 -> runs/memqa-base-2way/logs/env.txt
bash scripts/memqa/run_memqa.sh models     # shard check + YaRN model views
bash scripts/memqa/run_memqa.sh prompts    # 2 prompt files + prompts/SHA256SUMS
bash scripts/memqa/run_memqa.sh check      # idle-GPU gate + 2-prompt LongMemEval smoke on GPU0

# full run (models -> prompts -> check -> run -> score -> judge)
nohup bash scripts/memqa/run_memqa.sh all > runs/memqa-base-2way/logs/all.log 2>&1 &

# monitoring (read-only, from the laptop)
ssh -t H200_ANNP36 'cd /home/annp36/work/ACL-ECHO && INTERVAL=10 bash scripts/watch_memqa-base-2way.sh'

# verification
cat runs/memqa-base-2way/exit_code
sha256sum -c runs/memqa-base-2way/prompts/SHA256SUMS
sha256sum -c runs/memqa-base-2way/metrics/SHA256SUMS
cat runs/memqa-base-2way/logs/score.log runs/memqa-base-2way/logs/judge.log
```

To rerun only the two-way scoring (for example after a grader change), run `bash scripts/memqa/run_memqa.sh score`, which does not use the GPUs. `judge` reuses saved verdicts from phases whose `exit_code` is 0 and only re-summarizes.

# Outputs

All outputs stay inside the project directory.

- `output_path`: `/home/annp36/work/ACL-ECHO/runs/memqa-base-2way`
- `checkpoint`: none, since this is inference only. Model views are in `runs/memqa-base-2way/models/<tag>`.
- `prompts`: `prompts/<tag>_<bench>.jsonl`, `prompts/SHA256SUMS`
- `sample`: `sample/` holds the pre-run examples, which are not part of the metrics:
  - Qwen2.5-7B on LoCoMo conv-26 q0 and q5;
  - `sample/qwen3-8b/`: Qwen3-8B on those two plus LongMemEval `gpt4_59149c77` and `6a1eabeb`, with a judge smoke test.
- `generations`: `generations/<tag>/<bench>/shard_{0..7}.jsonl`, plus `shard_k.log`, `pids`, `exit_code`
- `judge`: `judge/<tag>/<bench>/shard_{0..7}.jsonl` (verdict and raw judge output per unique pair), plus `shard_k.log`, `pids`, `exit_code`
- `metrics`: `metrics/<tag>_<bench>.json` and `.per_question.jsonl` (two-way), `metrics/<tag>_<bench>.judge.json` and `.judge.per_question.jsonl` (judge), `metrics/SHA256SUMS`
- `logs`: `logs/driver.log`, `logs/all.log`, `logs/env.txt`, `logs/prompts.log`, `logs/score.log`, `logs/judge.log`, `logs/smoke_<tag>.{log,jsonl}`
- `pid_file`: `runs/memqa-base-2way/run.pid`
- `exit_code_file`: `runs/memqa-base-2way/exit_code` (whole run), plus `generations/<tag>/<bench>/exit_code` and `judge/<tag>/<bench>/exit_code` (per phase)
- `report`: `run-outputs/out_memqa-base-2way.md`, written after the run ends

# Upload

- `rclone_dest`: none given, so no upload. Artifacts stay on the server under `output_path` until a destination is provided.

# Acceptance

## Expected artifacts

- 2 two-way metrics files and 2 judge metrics files, each with its per-question JSONL.
- `prompts/SHA256SUMS` and `metrics/SHA256SUMS` both verify.
- Whole-run `exit_code` is 0, and all 4 phase `exit_code` files (2 generation, 2 judge) are 0.

## Expected metrics

- No target numbers. This run *measures* the pre-RL baselines, and LoongRL does not report LoCoMo or LongMemEval.
- Every metrics file must pass these checks:
  - question counts are LoCoMo 1,982 and LongMemEval 500;
  - `samples` = 8 × questions;
  - every generation's `prompt_sha256` matches its prompt file, which `score.py` asserts;
  - `questions_with_gt_session_dropped` = 0.
- Sanity checks (a failure means investigate, not abort):
  - `boxed_rate` ≥ 0.9;
  - `length_truncated_rate` ≤ 0.05 (the sample had 0 of 32, with a maximum of 5,405 response tokens);
  - judge `invalid_rate` ≤ 0.01;
  - judge `avg@8` ≥ two-way `avg@8` is expected, because the judge accepts equivalent forms such as `7` for `7 days`. A large `substring_only` count means the judge is stricter than expected and should be inspected.

## Failure conditions

- GPUs busy or fewer than 8 H200 → exit code 3 before any generation.
- Any shard exits with a non-zero code → the phase `exit_code` is non-zero, and the driver aborts without starting later phases.
- `score.py` raises when questions are missing, sample counts are wrong, or prompts do not match. `judge.py summarize` raises when a judged pair has no verdict.
- OOM, NCCL errors, NaN output, or a full disk in `shard_*.log`; the watcher greps for these.

## Retry policy

- Rerunning `all` is safe:
  - phases whose `exit_code` is 0 are skipped;
  - a failed phase resumes per shard, skipping question ids already written;
  - prompts are rebuilt deterministically, and their hashes must equal the first `SHA256SUMS`.
- Retry a failed phase once without changes. If it fails again, stop and report; do not change the generation parameters within this `run_id`.
- A run with any changed parameter (prompt, sampling, context length, grader) gets a new `run_id` and a new run file.

## Known limitations (must appear in the report)

- `single-session-preference` GTs are free-text rubrics, so substring matching scores them near 0 by construction. `overall_excluding_preference` is reported next to `overall`.
- A relative-time GT can only be matched by restating it (`The Sunday before 25 May 2023`); a correctly resolved date (`21 May 2023`) scores 0. The prompt's date-format hint pushes the model toward the resolved form, so temporal scores on these GTs are a lower bound.
- The `a ⊆ y` side gives credit for a partial answer to a list question (e.g. one item of a 3-item GT). The `y ⊆ a` side, which is the paper's reward, can be gamed by long answers that list candidates. `reference_one_way` isolates the second effect.
- The date-format hint and the abstention alias are evaluation choices that favour substring matching. Numbers are not comparable to LLM-judge scores in the LoCoMo or LongMemEval papers.
- The LongMemEval metadata says "oracle" while the data is S-sized; see `# Data`.
- Qwen3-8B runs outside its native chat template, with `top_k=-1` instead of the recommended 20.
- Two prompt issues seen in the samples are left unchanged in this run:
  - LoCoMo's synthetic `Current date` sometimes pulls Qwen3 away from the session date.
  - Number-only answers (`7` for `7 days`) score 0 under two-way because of the 2-character minimum; the judge credits them.
- Judge caveats:
  - The judge is the same model family as the evaluated model, so self-preference bias is possible.
  - Its prompt accepts a response that "contains" the reference, so a hedged answer listing several candidates can pass.
  - Judge numbers are not comparable to the official LongMemEval GPT-4o judge, which uses per-type prompts.
  - In the smoke test the judge accepted `20 May 2023` (a Saturday) for `The sunday before 25 May 2023` (21 May) in 5 of 8 samples, so it can miss off-by-one-day date errors.

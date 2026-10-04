---
name: ssh-experiment-runner
description: Migrate projects to SSH GPU servers, run durable sequential H200 training/evaluation jobs, monitor health and progress, and upload validated artifacts. Use for remote ML experiments; not for local-only runs.
---

# SSH Experiment Runner

## Run input

- Require one `run-inputs/<run_id>.md` file with these sections:
  - `# Project`: `name`, `run_id`, `local_path`, `ssh_host`, `remote_path`, `source_revision`.
  - `# Schedule`: `start_at`, `timezone`, `max_runtime`.
  - `# Data`: input, train, validation, benchmark, and manifest paths.
  - `# Model`: `repo_id`, `revision`, `model_path` or `cache_path`, task, method, seed, batch sizes, epochs, and GPU policy.
  - `# Commands`: exact preflight, training, evaluation, and verification commands in execution order.
  - `# Outputs`: unique output, checkpoint, log, PID, and exit-code paths.
  - `# Upload`: rclone destination and included artifacts.
  - `# Acceptance`: expected artifacts and metrics, failure conditions, and retry policy.
- Ask only for missing fields that materially change the run; never invent paths, commands, revisions, or upload destinations.

## Run output

- After a terminal success or failure, write `run-outputs/out_<run_id>.md`; never overwrite an existing report without asking, and ask before writing a partial report for an unfinished run.
- Use only that run's metrics, validation dumps, logs, gates, and artifacts; never invent or extrapolate values, and disclose any discovered deviation from the run input.
- Include these sections in order:
  1. **Header:** run id, server, run root, link to `run-inputs/<run_id>.md`, status (completed/failed, exit code, gates passed), phase timestamps, and elapsed time.
  2. **Settings:** model and revision, actual configuration, method, objective, optimizer, learning-rate schedule, steps, global batch, and validation protocol; when a paper reference exists, add a table comparing the actual run with the paper and explain each deviation.
  3. **Main Results:** tables by metric family, with final values, relevant baselines or paper references, reproducible raw counts, and secondary graders clearly labelled as reference only.
  4. **Insights:** paragraphs that interpret the result tables; begin each paragraph with a **bold hook sentence** and support every claim with numbers and artifact paths.
  5. **Artifacts:** metric files, validation dumps, checkpoints, logs, hashes, large-directory sizes, GPU release state, and upload status.
  6. **Limitations:** constraints such as single seed, checkpoint selection, missing baselines, and known bugs.
  7. **Next Steps:** possible follow-up work, explicitly marked as not run.

## Environment and Hugging Face

- For every Hugging Face operation, use the project virtual environment and never install globally; disable shell tracing, set `HF_HUB_DISABLE_XET=1`, and do not use XET or `HF_HUB_ENABLE_HF_TRANSFER`; run `hf auth whoami`, then authenticate if needed from `HF_TOKEN` or `${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}` by calling `huggingface_hub.login` through the project Python so the secret never enters arguments or logs; never store or print a token literal, and stop to ask the user if neither credential source exists; prefetch each pinned model/revision once, validate the local snapshot, then set `HF_HUB_OFFLINE=1` for training and evaluation.

## Migration

- Inspect local and remote changes first; preserve user files and never use `--delete` unless explicitly requested.
- Prefer `git pull --ff-only` for a clean tracked revision; otherwise sync explicit paths while excluding secrets, environments, caches, logs, datasets, and checkpoints as appropriate.
- After migration, delete AppleDouble files only inside the resolved remote project root: `find "$REMOTE_PROJECT" -type f -name '._*' -print -delete`.
- Verify source revision, required file hashes, virtual environment, model revision, free disk, and launch scripts before consuming GPUs.

## H200 training

- Inspect GPU ownership, processes, utilization, memory, and model type before launch.
- Every fine-tuning/training job must use all H200 GPUs in one sequential DDP job; never split them across concurrent runs or silently fall back to fewer GPUs, and if any required GPU belongs to another user or process, wait or request direction rather than taking, stopping, or reusing it without explicit authorization.
- Record and verify

$$
B_{\mathrm{global}}=N_{\mathrm{GPU}}\times B_{\mathrm{device}}\times G_{\mathrm{accum}}.
$$

- Use a durable launcher with unique run/output paths, `nohup`, PID file, explicit exit-code file, structured logs, and checkpoint validation.
- Run only one training phase at a time; gate dependent phases on exit code `0` and valid artifacts.
- Never change the dataset, seed, batch, objective, sampler, or hyperparameters during retry without explicit approval.

## Monitoring

- Create a read-only health monitor every 20 minutes for PID/exit state, optimizer step, finite loss/gradient, log/checkpoint modification time, disk, and every GPU's owner, utilization, VRAM, temperature, and power.
- Scan for traceback, CUDA OOM, NCCL failure, NaN/Inf, worker death, and disk-full errors; do not infer a stall from one idle sample.
- Create a Mac Terminal watcher that refreshes every 30 seconds and shows a progress bar, phase, overall percent, step/total, epoch, loss, learning rate, elapsed time, ETA, latest metric, and all GPU health values.
- Always return one directly runnable command in this form: `ssh -t SSH_HOST 'cd REMOTE_PATH && INTERVAL=10 bash scripts/watch_RUN_ID.sh'`.
- `Ctrl-C` must stop only the watcher, never the remote run.

## Artifacts and upload

- Require exit code `0`, complete checkpoints, parseable metrics, and expected hashes before upload.
- Check `rclone listremotes` on the server. If the requested remote is unavailable or unauthenticated, prompt the user; never replace the destination silently.
- Upload only artifacts listed in the run Markdown with `rclone copy OUTPUT_PATH RCLONE_DEST --progress` after validation.
- Verify the uploaded destination with rclone listing/checks and report local path, remote destination, size, hashes, and upload status.
- Preserve failed or partial artifacts for diagnosis unless the user explicitly requests deletion.

## Completion

- In one concise completion sentence, report the run configuration, GPU count, effective global batch, elapsed time, final checkpoint, metrics, failures or retries, output path, rclone destination, GPU release state, and `run-outputs/out_<run_id>.md` path.

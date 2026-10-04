# Shared driver for run_gemma.sh / run_qwen.sh on the 8×H200 server (sourced, not run).
# Every phase is one torchrun DDP job over all 8 GPUs. Phases run strictly one after another,
# and a phase starts only after the previous one exited 0 with a valid checkpoint.

RUN_ID=${RUN_ID:-amid-kd-8xh200}
N_GPU=8
EFF_BATCH=64
EPOCHS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER

DATA_REPO=VoCuc/UltraInteract-Infer
export DATA_REVISION=3c2fb0d397d3ef2d505f3fe48ad4e30f30733895
# Both teachers' tokenizers are needed by process_data_ultraInteract.sh, so both runs prefetch all four.
PINNED_MODELS=(
    google/gemma-2-2b-it=299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
    google/gemma-2-9b-it=11c9b309abf73637e4b6f9a3fa1e92e615547819
    Qwen/Qwen2.5-0.5B-Instruct=7ae557604adf67be50417f59c2c2f167def9a775
    Qwen/Qwen3-4B-Instruct-2507=cdbee75f17c01a7cc42f958dc650907174af0554
)

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# init_run TAG: run dir, pid file, driver log and a whole-run exit code written on any exit.
init_run() {
    RUN_DIR="runs/${RUN_ID}/$1"
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/run.pid" ] && kill -0 "$(cat "$RUN_DIR/run.pid")" 2>/dev/null; then
        echo "driver already running (pid $(cat "$RUN_DIR/run.pid"))" >&2
        exit 2
    fi
    echo $$ > "$RUN_DIR/run.pid"
    rm -f "$RUN_DIR/exit_code"
    trap 'echo $? > "$RUN_DIR/exit_code"' EXIT
    trap 'exit 143' TERM INT
    exec >> "$RUN_DIR/driver.log" 2>&1
    log "start run_id=$RUN_ID tag=$1 host=$(hostname) rev=$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"
}

# hf_prepare REPO=SHA ...: authenticate, then prefetch each pinned model once and check that the
# snapshot `main` resolves to is the pinned one.
hf_prepare() {
    set +x
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if ! hf auth whoami >/dev/null 2>&1; then
        python - <<'EOF' || return 1
import os, sys
from huggingface_hub import login
tok = os.environ.get("HF_TOKEN")
path = os.environ.get("HF_TOKEN_FILE", os.path.expanduser("~/.config/huggingface/token"))
if not tok and os.path.isfile(path):
    tok = open(path).read().strip()
if not tok:
    sys.exit("no HF credential: set HF_TOKEN or HF_TOKEN_FILE")
login(token=tok, add_to_git_credential=False)
EOF
    fi
    python - "$@" <<'EOF' || return 1
import json, os, sys
from huggingface_hub import snapshot_download
for spec in sys.argv[1:]:
    repo, rev = spec.split("=")
    path = snapshot_download(repo, revision="main",
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    got = os.path.basename(path)
    if got != rev:
        sys.exit(f"{repo}: main is {got}, pinned {rev}; update the run file before training")
    idx = os.path.join(path, "model.safetensors.index.json")
    shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
    missing = [s for s in shards if not os.path.isfile(os.path.join(path, s))]
    if missing:
        sys.exit(f"{repo}@{rev}: missing shards {missing}")
    print(f"model ok {repo}@{rev} ({len(shards)} shards) {path}")
EOF
}

# prepare_data TEACHER: pinned dataset download, then tokenisation if the teacher's split is absent.
prepare_data() {
    local teacher=$1 out="./processed_data/ultraInteract/$1"
    bash ./scripts/download_data.sh || return 1
    sha256sum "./data/dpo/$teacher/generated_train.jsonl" | tee "$RUN_DIR/data.sha256"
    if [ ! -s "$out/train_0.idx" ] || [ ! -s "$out/valid_0.idx" ]; then
        bash ./scripts/process_data_ultraInteract.sh || return 1
    fi
    ( cd "$out" && sha256sum train_0.bin train_0.idx valid_0.bin valid_0.idx ) | tee "$RUN_DIR/processed_data.sha256"
}

# check_pins: every pinned repo's cached `main` ref still points at the pinned SHA. Training loads the
# models online by repo id, so a `main` that moved upstream during a phase shows up here.
check_pins() {
    local spec repo rev ref bad=0
    for spec in "${PINNED_MODELS[@]}"; do
        repo=${spec%%=*}; rev=${spec#*=}
        ref=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${repo//\//--}/refs/main" 2>/dev/null)
        [ "$ref" = "$rev" ] || { log "pin check failed: $repo main=${ref:-missing}, pinned $rev"; bad=1; }
    done
    return $bad
}

# require_idle_gpus: exit 3 unless 8 H200 are visible with no compute process. Waits up to
# 5 min first, so processes from the phase that just ended can release their memory.
require_idle_gpus() {
    local n busy i
    for i in $(seq 30); do
        n=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200)
        busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
        [ "$n" -eq "$N_GPU" ] && [ "$busy" -eq 0 ] && return 0
        sleep 10
    done
    log "GPU gate failed: $n H200 visible, $busy compute processes"
    nvidia-smi
    exit 3
}

# check_ckpt NAME SAVE: a phase is valid when B_global = 64 on 8 GPUs, every epoch saved a step
# dir, the last one holds non-empty weights and a tokenizer, and the dev evals were logged.
check_ckpt() {
    local name=$1 save=$2 plog="$RUN_DIR/$1.log" ng bs ga total last nsteps weights ndev
    ng=$(grep -o -- ' --n-gpu [0-9]*' "$plog" | head -1 | awk '{print $2}')
    bs=$(grep -o -- ' --batch-size [0-9]*' "$plog" | head -1 | awk '{print $2}')
    ga=$(grep -o -- ' --gradient-accumulation-steps [0-9]*' "$plog" | head -1 | awk '{print $2}')
    total=$(grep -m1 '^total_iters' "$plog" | awk '{print $2}')
    if [ "$(grep -o -- ' --save [^ ]*' "$plog" | head -1 | awk '{print $2}')" != "$save" ]; then
        log "$name: --save in $plog differs from $save"; return 1
    fi
    if [ "$ng" != "$N_GPU" ] || [ $((${ng:-0} * ${bs:-0} * ${ga:-0})) -ne "$EFF_BATCH" ]; then
        log "$name: B_global check failed (n_gpu=$ng batch=$bs grad_acc=$ga)"; return 1
    fi
    nsteps=$(find "$save" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | wc -l)
    last=$(find "$save" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n | tail -1)
    weights=$(find "$save/$last" -maxdepth 1 -size +0 \( -name 'adapter_model.*' -o -name 'pytorch_model*.bin' -o -name 'model*.safetensors' \) 2>/dev/null)
    ndev=$(grep -c '^dev | avg_loss' "$save/log.txt" 2>/dev/null)
    if [ "$nsteps" -lt "$EPOCHS" ] || [ -z "$weights" ] || [ ! -s "$save/$last/tokenizer_config.json" ] || [ "${ndev:-0}" -lt $((EPOCHS + 1)) ]; then
        log "$name: checkpoint check failed (step dirs=$nsteps last=$last weights=${weights:-none} dev evals=${ndev:-0})"; return 1
    fi
    {
        echo "save_path=$save"
        echo "final_checkpoint=$save/$last total_iters=$total"
        echo "B_global=$((ng * bs * ga)) = $ng GPU x $bs per device x $ga grad_acc"
        du -sh "$save"
        sha256sum $weights
    } > "$RUN_DIR/$name.ckpt"
    [ "$last" = "$total" ] || log "$name: warning, last step dir $last != total_iters $total"
}

# run_phase NAME SCRIPT SAVE: one 8-GPU training phase with its own log and exit-code file.
# A phase with exit code 0 and a valid checkpoint is skipped; a failed attempt's save dir is
# kept beside the new one as SAVE.failed_<UTC time>.
run_phase() {
    local name=$1 script=$2 save=$3 plog="$RUN_DIR/$1.log" pexit="$RUN_DIR/$1.exit_code" ts rc
    if [ "$(cat "$pexit" 2>/dev/null)" = 0 ] && check_ckpt "$name" "$save" >/dev/null; then
        log "$name: done earlier, skipping"; return 0
    fi
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    [ -e "$save" ] && mv "$save" "$save.failed_$ts" && log "$name: kept previous attempt as $save.failed_$ts"
    [ -f "$plog" ] && mv "$plog" "$plog.$ts"
    rm -f "$pexit"
    require_idle_gpus
    log "$name: start $script -> $save"
    echo "$name" > "$RUN_DIR/current_phase"
    SAVE_PATH="$save" bash "$script" > "$plog" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && ! check_ckpt "$name" "$save"; then rc=4; fi
    if [ $rc -eq 0 ] && ! check_pins; then rc=5; fi
    echo $rc > "$pexit"
    log "$name: exit $rc"
    return $rc
}

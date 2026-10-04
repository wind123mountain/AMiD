#!/bin/bash
# Read-only watcher for run_gemma.sh + run_eval.sh (run_id gemma-kd-len2048). Ctrl-C stops only this loop.
# Usage: ssh -t vt-admin 'cd /nvme/annp36-home/work/AMiD && INTERVAL=10 bash scripts/watch_gemma-kd-len2048.sh'

cd "$(dirname "$0")/.."
RUN_ID=${RUN_ID:-gemma-kd-len2048}
INTERVAL=${INTERVAL:-30}
trap 'echo; exit 0' INT TERM

vn() { TZ=Asia/Ho_Chi_Minh date -d "@$1" '+%m-%d %H:%M:%S'; }
hms() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }
bar() { local n=$(($1 * 40 / 100)); printf '[%s%s] %3d%%' "$(printf "%${n}s" | tr ' ' '#')" "$(printf "%$((40 - n))s" | tr ' ' '.')" "$1"; }

# phase list in launch order, read from the active run_phase lines of each run file
phases() { grep -E '^run_phase ' "run_$1.sh" | awk '{print $2}'; }

render() {
    local now total_ph=0 done_ph=0 frac=0 tag ph dir st line cur tot ep loss lr start el eta save metric pid
    now=$(date +%s)
    echo "AMiD $RUN_ID   $(TZ=Asia/Ho_Chi_Minh date '+%Y-%m-%d %H:%M:%S') giờ Việt Nam   refresh ${INTERVAL}s"
    echo
    for tag in gemma; do
        dir="runs/$RUN_ID/$tag"
        pid=$(cat "$dir/run.pid" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then st="running (pid $pid)"
        elif [ -f "$dir/exit_code" ]; then st="exited $(cat "$dir/exit_code")"
        else st="not started"; fi
        echo "== $tag driver: $st"
        for ph in $(phases $tag); do
            total_ph=$((total_ph + 1))
            line=""; cur=0; tot=0
            [ -f "$dir/$ph.log" ] && line=$(grep '^train | epoch' "$dir/$ph.log" | tail -1)
            if [ -n "$line" ]; then
                cur=$(sed -E 's/.*global iter: *([0-9]+)\/ *([0-9]+).*/\1/' <<< "$line")
                tot=$(sed -E 's/.*global iter: *([0-9]+)\/ *([0-9]+).*/\2/' <<< "$line")
                ep=$(sed -E 's/.*epoch +([0-9]+).*/\1/' <<< "$line")
                loss=$(sed -E 's/.*\| loss: ([^ ]+).*/\1/' <<< "$line")
                lr=$(sed -E 's/.*\| lr: ([^ ]+).*/\1/' <<< "$line")
            fi
            if [ -f "$dir/$ph.exit_code" ]; then
                if [ "$(cat "$dir/$ph.exit_code")" = 0 ]; then done_ph=$((done_ph + 1)); st="done"
                else st="FAILED exit $(cat "$dir/$ph.exit_code")"; fi
                printf '  %-5s %s\n' "$ph" "$st"
            elif [ -f "$dir/$ph.log" ] && [ "$(cat "$dir/current_phase" 2>/dev/null)" = "$ph" ]; then
                start=$(grep -E "\] $ph: start" "$dir/driver.log" | tail -1 | sed -E 's/^\[([^]]+)\].*/\1/')
                start=$(date -d "$start" +%s 2>/dev/null || stat -c %Y "$dir/$ph.log")
                el=$((now - start)); eta="?"
                [ "$cur" -gt 0 ] && eta=$(hms $((el * (tot - cur) / cur)))
                [ "$tot" -gt 0 ] && frac=$((cur * 100 / tot))
                save=$(grep -o -- ' --save [^ ]*' "$dir/$ph.log" | head -1 | awk '{print $2}')
                metric=$(grep '^dev | avg_loss' "$save/log.txt" 2>/dev/null | tail -1 | cut -c1-110)
                printf '  %-5s %s  step %s/%s  epoch %s  loss %s  lr %s\n' "$ph" "$(bar $frac)" "$cur" "$tot" "${ep:--}" "${loss:--}" "${lr:--}"
                printf '        elapsed %s  ETA %s  log updated %s\n' "$(hms $el)" "$eta" "$(vn "$(stat -c %Y "$dir/$ph.log")")"
                [ -n "$metric" ] && echo "        latest: $metric"
                grep -aiE 'Traceback|out of memory|NCCL error|No space left|loss: -?(nan|inf)|exitcode *: *-?[1-9]' "$dir/$ph.log" | tail -2 | sed 's/^/        !! /' | cut -c1-140
            else
                printf '  %-5s pending\n' "$ph"
            fi
        done
    done
    local running_part=$(( frac ))
    [ "$done_ph" -eq "$total_ph" ] && running_part=0
    echo
    echo "overall $(bar $(( (done_ph * 100 + running_part) / (total_ph > 0 ? total_ph : 1) )))  phases done $done_ph/$total_ph"

    # lm-eval (run_eval.sh -> scripts/eval/eval_2.sh): models finished, current model and task
    dir="runs/$RUN_ID/eval"
    pid=$(cat "$dir/run.pid" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then st="running (pid $pid)"
    elif [ -f "$dir/exit_code" ]; then st="exited $(cat "$dir/exit_code")"
    else st="not started"; fi
    echo "== eval driver: $st"
    if [ -f "$dir/eval_2.log" ]; then
        printf '  models done %s/3, failed %s   log updated %s\n' \
            "$(grep -cE '=== (Xong|Bỏ qua)' "$dir/eval_2.log")" "$(grep -c '=== LỖI' "$dir/eval_2.log")" \
            "$(vn "$(stat -c %Y "$dir/eval_2.log")")"
        if [ "$(cat "$dir/current_phase" 2>/dev/null)" = eval_2 ] && [ ! -f "$dir/eval_2.exit_code" ]; then
            echo "  now: $(grep -E '=== Bắt đầu: ' "$dir/eval_2.log" | tail -1 | sed -E 's/.*Bắt đầu: (.*) ===/\1/')  $(grep -E '^>>> \[' "$dir/eval_2.log" | tail -1)"
        fi
        grep -aE '!! FAILED|Traceback|out of memory|No space left' "$dir/eval_2.log" | tail -2 | sed 's/^/  !! /' | cut -c1-140
    fi
    echo
    echo "GPU  util  mem(GiB)        temp  power(W)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits |
        awk -F', *' '{printf "%-3s  %3s%%  %6.1f/%-6.1f  %3sC  %6.0f\n", $1, $2, $3/1024, $4/1024, $5, $6}'
    echo "GPU processes (owner pid used MiB):"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r p m; do
        [ -n "$p" ] && echo "  $(ps -o user= -p "$p" 2>/dev/null || echo '?') $p $m"
    done
    echo "disk /nvme: $(df -h /nvme | awk 'NR==2{print $4" free ("$5" used)"}')"
}

while true; do
    out=$(render 2>&1)
    clear
    echo "$out"
    sleep "$INTERVAL"
done

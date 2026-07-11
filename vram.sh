#!/bin/bash

measure_vram_peak() {
    local SCRIPT_TO_RUN="${1:-run.sh}"
    local LOGFILE="${2:-vram_log.txt}"
    local DURATION="${3:-600}"

    > "$LOGFILE"

    echo "==> Chạy: $SCRIPT_TO_RUN | Log: $LOGFILE | Duration: ${DURATION}s"

    # Chạy script cần đo ở background, tự kill sau DURATION giây
    timeout "${DURATION}s" bash "$SCRIPT_TO_RUN" &
    local TRAIN_PID=$!

    # Map gpu_uuid -> index vật lý (0,1,2,...)
    local -A GPU_INDEX_OF_UUID
    while IFS=',' read -r idx uuid; do
        idx=$(echo "$idx" | xargs)
        uuid=$(echo "$uuid" | xargs)
        GPU_INDEX_OF_UUID["$uuid"]="$idx"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        # Toàn bộ PID con (đệ quy) của script đang chạy
        local CHILD_PIDS
        CHILD_PIDS=$(pstree -p "$TRAIN_PID" 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))')

        # gpu_uuid,pid,used_memory -> lọc theo PID con -> cộng dồn trong cùng 1 GPU
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
            | awk -F',' -v pids="$CHILD_PIDS" '
                BEGIN {
                    n = split(pids, arr, " ")
                    for (i = 1; i <= n; i++) valid[arr[i]] = 1
                }
                {
                    gsub(/^[ \t]+|[ \t]+$/, "", $1)
                    gsub(/^[ \t]+|[ \t]+$/, "", $2)
                    gsub(/^[ \t]+|[ \t]+$/, "", $3)
                    if ($2 in valid) mem[$1] += $3
                }
                END {
                    for (uuid in mem) print uuid "," mem[uuid]
                }
            ' | while IFS=',' read -r uuid mem; do
                local idx="${GPU_INDEX_OF_UUID[$uuid]:-unknown}"
                echo "gpu${idx},${mem}" >> "$LOGFILE"
            done

        sleep 1
    done

    wait "$TRAIN_PID"

    echo "==> Peak VRAM used theo từng GPU (MiB):"
    sort -t',' -k1,1 -k2,2n "$LOGFILE" | awk -F',' '
        { if ($2+0 > peak[$1]+0) peak[$1] = $2 }
        END { for (g in peak) print g": "peak[g]" MiB" }
    ' | sort
}


measure_vram_peak "scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_4.sh" "vram_nnm.log"
measure_vram_peak "scripts/csd/train_qwen2.5_1.5B_it.sh" "vram_csd.log"
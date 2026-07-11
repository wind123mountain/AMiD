#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Nhận path từ tham số khi chạy (nếu có), nếu không có thì dùng path mặc định
BASE_PATH=${1:-"results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_no_train_proj"}

echo "🔍 Đang tìm checkpoint trong: $BASE_PATH"

LATEST_CKPT=$(find "$BASE_PATH" -mindepth 1 -maxdepth 1 -type d -regex "$BASE_PATH/[0-9]+" | sort -V | tail -n 1)

# Kiểm tra xem có tìm thấy thư mục nào không
if [ -z "$LATEST_CKPT" ]; then
    echo "❌ Lỗi: Không tìm thấy thư mục checkpoint nào trong $BASE_PATH"
    exit 1
fi

# Xóa dấu '/' thừa ở cuối path (nếu có) để dòng lệnh gọn gàng hơn
LATEST_CKPT=${LATEST_CKPT%/}

echo "✅ Đã tìm thấy checkpoint mới nhất: $LATEST_CKPT"
echo "🚀 Bắt đầu chạy script merge..."

# 3. Chạy lệnh python
python tools/merge_model.py \
  --base_model "$BASE_MODEL" \
  --adapter "$LATEST_CKPT" \
  --output "$BASE_PATH"

echo "🎉 Hoàn tất!"
set -e
 
if [ $# -eq 0 ]; then
    VARIANTS=(bnm bnmm erank)
else
    VARIANTS=("$@")
fi
 
for V in "${VARIANTS[@]}"; do
    echo ""
    echo "############################################"
    echo "# Running variant: $V"
    echo "############################################"
    bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_1e_ablation.sh "$V"
done
 
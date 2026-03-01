#!/bin/bash
# Pareto frontier sweep: train one model per distance_penalty value.
# Run from /mnt/spatial_lm on the Ubuntu machine.
#
#   bash pareto_sweep.sh
#
# Results land in pareto_p{penalty}.log and checkpoint_pareto_p{penalty}.pt

MICROMAMBA="$HOME/.local/bin/micromamba"
ENV="/mnt/research/micromamba/envs/spatial_lm"
PY="$MICROMAMBA run -p $ENV python -u"

DATA=data/python_stdlib.txt
STEPS=10000
SEQ=256
BATCH=64
EVAL=1000

for p in 0.0 0.1 0.3 0.5 1.0 2.0 3.0; do
    echo "============================================"
    echo "  penalty=$p  ($(date))"
    echo "============================================"
    $PY train.py \
        --data        $DATA \
        --seq_len     $SEQ \
        --batch_size  $BATCH \
        --max_steps   $STEPS \
        --eval_interval $EVAL \
        --distance_penalty $p \
        --checkpoint  "checkpoint_pareto_p${p}.pt" \
        2>&1 | tee "pareto_p${p}.log"
    echo ""
done

echo "Pareto sweep complete — run pareto_plot.py to visualize."

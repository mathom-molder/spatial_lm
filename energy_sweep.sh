#!/bin/bash
# Energy weight sweep: fix distance_penalty=0.1 (best Pareto point),
# vary energy_weight to add metabolic pressure.
#
#   bash energy_sweep.sh
#
# Results land in energy_ew{weight}.log and checkpoint_energy_ew{weight}.pt

MICROMAMBA="$HOME/.local/bin/micromamba"
ENV="/mnt/research/micromamba/envs/spatial_lm"
PY="$MICROMAMBA run -p $ENV python -u"

DATA=data/code.txt
STEPS=10000
SEQ=256
BATCH=64
EVAL=1000

for ew in 0.001 0.005 0.01 0.05 0.1; do
    echo "============================================"
    echo "  energy_weight=$ew  ($(date))"
    echo "============================================"
    $PY train.py \
        --data          $DATA \
        --seq_len       $SEQ \
        --batch_size    $BATCH \
        --max_steps     $STEPS \
        --eval_interval $EVAL \
        --distance_penalty 0.1 \
        --energy_weight $ew \
        --checkpoint    "checkpoint_energy_ew${ew}.pt" \
        2>&1 | tee "energy_ew${ew}.log"
    echo ""
done

echo "Energy sweep complete — run energy_plot.py to visualize."

#!/usr/bin/env bash
# Resumable single-GPU experiment queue. Re-running skips completed runs (marker files).
#   WANDB_ENTITY=... WANDB_PROJECT=spectral-muon-nanogpt bash sweep.sh
# Remove a marker in sweeps/done/ to force a rerun.
set -u
mkdir -p sweeps/done sweeps/logs

run() {
    name="$1"; shift
    if [ -f "sweeps/done/${name}.done" ]; then
        echo "[skip] ${name}"
        return
    fi
    echo "[run ] ${name}: $*"
    if torchrun --standalone --nproc_per_node=1 train_gpt2.py "$@" 2>&1 | tee "sweeps/logs/${name}.log"; then
        touch "sweeps/done/${name}.done"
    else
        touch "sweeps/done/${name}.fail"
        echo "[FAIL] ${name} (marker: sweeps/done/${name}.fail)"
    fi
}

TAUS="1e-4 2e-4 4e-4 1e-3 3e-3"
FAM="--optim=Muon --rms_match=0 --warmup_iters=0"

# ---- Tier 1: Muon endpoint + top arm, seeds 0,1 ----
for S in 0 1; do
    run "conv-s${S}"          $FAM --backend=conv --seed=${S}
    for T in $TAUS; do
        run "top-tau${T}-s${S}"  $FAM --backend=targeted --arm=top --tau=${T} --seed=${S}
    done
done

# ---- Tier 2: bottom arm, seeds 0,1 ----
for S in 0 1; do
    for T in $TAUS; do
        run "bot-tau${T}-s${S}"  $FAM --backend=targeted --arm=bot --tau=${T} --seed=${S}
    done
done

# ---- Tier 3: third seed for the whole family ----
run "conv-s2" $FAM --backend=conv --seed=2
for A in top bot; do
    for T in $TAUS; do
        run "${A}-tau${T}-s2" $FAM --backend=targeted --arm=${A} --tau=${T} --seed=2
    done
done

# ---- Tier 4: study B (AdamW cells; warmup matched = 250) ----
for S in 0 1; do
    run "adamw-wd0.01-s${S}"  --optim=AdamW --weight_decay=0.01 --warmup_iters=250 --seed=${S}
    run "adamw-wd0-s${S}"     --optim=AdamW --weight_decay=0    --warmup_iters=250 --seed=${S}
    run "muon-warm-s${S}"     --optim=Muon --backend=conv --rms_match=0 --warmup_iters=250 --seed=${S}
done

# ---- Not yet runnable (code missing) ----
# NSGD endpoint: needs the nsgd backend (power-iteration spectral-norm normalize) added to zeropower_backends
# for S in 0 1 2; do run "nsgd-s${S}" $FAM --backend=nsgd --seed=${S}; done
# Muon+wd cells (study B4): needs decoupled wd in Muon.step
# for S in 0 1; do run "muon-wd0.01-s${S}" --optim=Muon --backend=conv --rms_match=0 --warmup_iters=250 --weight_decay=0.01 --seed=${S}; done
echo "queue complete."

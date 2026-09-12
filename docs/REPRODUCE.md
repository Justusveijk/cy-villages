# Reproducing the results

All scripts take `--quick` for a 3-minute laptop smoke test. Real runs need a GPU (Colab L4 was used) and Python 3.11.

## Environment
```
uv venv cy311 --python 3.11 && source cy311/bin/activate
uv pip install "git+https://github.com/pythoncymetric/cymetric.git"
```
On Colab, use `notebooks/cy_v4_colab.ipynb` (installs TF 2.14 + CUDA 11 libs via uv).

## Order of runs
1. `python scripts/sample_overnight.py` — samples 66 bicubic shapes (groups A–D, held-outs, strangers) into `moduli_data_v4/`. CPU, ~1 min/shape on Apple Silicon.
2. `python scripts/moduli_net_v4.py` — specialists A, B; transfer test (Fig 1). Set `E_SPEC=25, E_WARM=10`. ~7 GPU-hours.
3. `python scripts/moduli_net_v5.py` — specialists C, D (A, B reused); routers; strangers. Set `E_GP=0`. ~4 GPU-hours.
4. `python scripts/radius.py --sample-only --base=A` then `python scripts/radius.py --base=A` — radius curve (Fig 2). Same with `--base=D`. ~1.5 GPU-hours each.
5. `python scripts/radius.py --long` — equal-budget test (Fig 3). ~2.5 GPU-hours.

Seeds are fixed in the scripts; tables in the README should reproduce to ~3 decimals (Monte-Carlo σ has ±0.002 noise).

## Outputs
- `results/*.csv` — every fine-tuning curve, one row per (shape, mode).
- `weights/specialist_{A,B,C,D}.weights.h5` — the four specialists (Keras, 4×256 GELU).

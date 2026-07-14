# RecallGuard Cog

[![Deploy to app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?image=ghcr.io/lee101/recall-guard-cog:latest&name=recall-guard&hardware=cpu-auto&idleSeconds=45)

A CPU-first, one-click deployment of the **Certify-then-Rectify** idea from
[HNSW with Accuracy Guarantees Using Graph Spanners](https://arxiv.org/abs/2607.02338).
It runs ordinary HNSW first, applies the paper's post-hoc conformal risk control
(CRC) gate, and escalates rejected queries to an exact vectorized L2 scan.

The result includes the chosen path, neighbors, squared-L2 distances, certificate
threshold and score, calibrated risk bound, acceptance statistics, and timings.
Enable `audit` to measure an accepted fast-path answer against exact retrieval.

## Why the rectifier is exact scan by default

The paper's faster MBV rectifier requires a dataset-specific empirical spanner
stretch bound. A universal default would turn a probabilistic assumption into a
misleading correctness claim. RecallGuard therefore uses the CRC gate unchanged
and a conservative exact scan for escalations. The upstream MBV implementation
remains in this repository for users who have calibrated a stretch bound.

## Run

```bash
cog predict \
  -i dataset_size=5000 \
  -i dimensions=64 \
  -i neighbors=10 \
  -i ef_search=32 \
  -i calibration_queries=200 \
  -i audit=true
```

Or run the built container directly:

```bash
docker run -p 5000:5000 ghcr.io/lee101/recall-guard-cog:latest
curl -s http://localhost:5000/predictions \
  -H 'content-type: application/json' \
  -d '{"input":{"dataset_size":5000,"dimensions":64,"audit":true}}'
```

Pass `vectors_json` as a JSON matrix and `query_json` as a JSON vector to use
your own data. If omitted, the Cog creates a reproducible clustered benchmark
with non-trivial HNSW cases. Inputs are bounded to five million scalar values so
a public endpoint cannot accidentally allocate unbounded memory.

## Inference work

- C++ HNSW extension compiled with `-O3`, OpenMP, and native CPU instructions.
- Contiguous `float32` storage and a squared-norm exact scan that avoids an
  `N x D` difference allocation.
- In-process LRU cache for three calibrated indexes, so repeat predictions skip
  construction and calibration.
- CPU-native Cog image; no CUDA or multi-gigabyte torch dependency.
- Release CI uses Cog's slim Python base, cutting the image from about 2.07 GB
  to 662 MB in the release check while preserving a reproducible source build.
- 45-second app.nz idle window by default, then full scale-to-zero.

## Validate

```bash
python -m pytest -q
cog build -t recall-guard-cog:test
```

## Attribution and license

Apache-2.0, following the upstream research implementation. See `NOTICE` for
authors and app.nz modifications.

---

## Upstream research repository

This repository contains the code for the paper experiments on CRC/LTT-based recall certification for HNSW.

It keeps the paper-relevant code paths:

- `hnswlib/`, `python_bindings/`: augmented C++ core and Python bindings
- `crc_core.py`: shared certifier core (calibration, thresholding, scoring, serialization)
- `run_crc_experiment.py`: one-shot CRC certifier runner
- `run_crc_feature_alpha_sweep.py`: CRC alpha sweep on a fixed split/index
- `run_crc_fixed_alpha_tau_sweep.py`: CRC tau sweep at fixed alpha
- `run_ltt_feature_alpha_sweep.py`: LTT alpha sweep with feature/model options
- `sparse_rectify.py`: rectifier helper (MBV / sparse-L1 backends)
- `scripts/reproduce_certifier_suite.sh`: environment check helper
- `paper/recall_certification.pdf`: companion paper

## Quick Start

Build the Python extension:

```bash
python -m pip install -e .
```

The experiments expect external dataset and index files:

- `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- `gist_base.fvecs`, `gist_query.fvecs`, `gist_groundtruth.ivecs`
- `deep1m_base.fbin`, `deep.query.public.10K.fbin`, `deep1M_groundtruth.ivecs`
- HNSW index files under `indexes/`

## Running the Certifier Experiments

```bash
# 1) Validate environment and inputs
./scripts/reproduce_certifier_suite.sh check

# 2) Base CRC run
python run_crc_experiment.py \
  --fvec-base-file sift_base.fvecs \
  --fvec-query-file sift_query.fvecs \
  --fvec-gt-file sift_groundtruth.ivecs \
  --M 16 --efc 100 --efs 100 --k 100 \
  --tau 0.90 --alpha 0.10 \
  --cal-size 5000 --qcnt 10000 --seed 0 \
  --out-dir crc_runs/sift_crc_base

# 3) CRC alpha sweep on that split/index
python run_crc_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_alpha_sweep \
  --tau 0.90 --alpha-mode adaptive

# 4) LTT alpha sweep on the same split/index
python run_ltt_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_ltt_alpha_sweep \
  --tau 0.90 --epsilon 0.05 \
  --alphas 0.1,0.05,0.01,0.005
```

## Notes

- Generated CSV outputs are not committed; rerun scripts will recreate them into local output directories.

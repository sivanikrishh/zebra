# Handoff for B: regenerate all percepts on Linux + RTX 3090

Everything is regenerable from the repo + the organiser's GZGC archive + a 9.4 MB data bundle
(masks and parquets). Derived images are **internal – not for public display** (CC BY-NC-ND 2.0).
Data layouts: `docs/contracts.md`.

## 1. Environment (Linux)

```bash
git clone <repo> zebrasight && cd zebrasight
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
pytest -q                     # expect all tests to pass (~25 s on a laptop GPU)
```

## 2. Data

```bash
mkdir -p data/raw
cp /path/to/gzgc.coco.tar.gz data/raw/          # organiser archive, 10,433,199,738 bytes
python -m zebrasight.data.unpack                # -> data/raw/gzgc/ (4,948 JPEGs, ~80 s)
python -m zebrasight.data.handoff unpack /path/to/zebrasight_data_handoff.zip
```

The bundle (`cache/zebrasight_data_handoff.zip` on A's laptop, made with
`python -m zebrasight.data.handoff pack`) holds the region label PNGs and
`data/regions.parquet` (+ v1 / pilot tables), so B uses exactly A's masks and splits.

**Region rule v2 (important).** `data/regions.parquet` now uses region rule v2: the face is the real
head (OWLv2 head detection, fallback: front tip of the silhouette), the flank is the trunk between
shoulder and hip (no neck), the rump excludes the tail. Rule v1's "face" was mostly neck, so
**any face percept made before this (pilot store, env-demo store except its 5 regenerated zebras)
is invalid**; those rows are flagged `valid: false`. Some photos have no face row (no usable head);
see `docs/data_report.md` for the shares. To rebuild the masks instead of using the bundle
(≈ 15 min on a 3090; needs `pip install transformers`, downloads OWLv2 once):
`python -m zebrasight.data.masks`, `python -m zebrasight.data.subset`, then
`python -m zebrasight.data.heads run` and `python -m zebrasight.data.regions_v2`.

## 3. One command

```bash
# 10 core conditions (c0-c9; c7 is stored as an alias of c8), flank + face, 16 distances
python -m zebrasight.pipeline.store data/regions.parquet cache/percepts_full.zarr
# add the 7 extra conditions (s25 s50 o20 o40 o60 m2 m5) to the SAME store later, additively
python -m zebrasight.pipeline.store data/regions.parquet cache/percepts_full.zarr --with-extras
```

To run it unattended, use `nohup … > logs/full.log 2>&1 &` (or tmux).
**Resume:** run the same command again. Condition groups already marked complete are skipped, so
an interrupted run continues where it stopped and adding conditions never recomputes old ones.
Progress lines (every 10 crops) show s/crop, time left and peak RAM.

## 4. Expected time and size

Measured on A's laptop (RTX 3050 Laptop 6 GB, i7) with the final code, 6 random subset crops:

| Run | Crops | Percepts | Laptop (measured rate) | RTX 3090 (estimate) | On disk |
|---|---|---|---|---|---|
| core c0–c9 (9 computed + c7 alias) | 2,088 | 300,672 | ~6.6 s/crop → **~3.8 h** | **~1 h** | **~28–31 GB** |
| + extras (16 computed) | 2,088 | 534,528 | 11.7 s/crop → **~6.8 h** | **~1.7 h** | **~47–59 GB** |

- The 3090 figures assume ~4× the laptop GPU. They are an estimate, not a measurement. Part of
  the work runs on the CPU (photon noise, Zarr compression, scene patterns), so the speed-up may be
  smaller than 4×.
- Size ranges: the low end is from the 6 measured crops; the high end is the analytic pixel count ×
  the pilot's measured compression ratio (2.09×). The subset keeps each zebra's largest, sharpest
  views, so crops are larger than average.
- Peak RAM ~2.3 GB, peak GPU < 1 GB per process.

## 5. Read percepts

```python
from zebrasight.pipeline.read_store import PerceptStore
st = PerceptStore("cache/percepts_full.zarr")
rows = st.rows()                         # zebra_id, region, split, ..., conds
p = st.get(rows.key[0], "c2", 9)         # dusty, 31.7 m
```

## 6. The pilot store (available now)

`cache/percepts_pilot.zarr`: 50 zebras, 409 photos, 818 crops (flank + face), conditions c0, c7, c8,
c9 × 16 distances, 4.6 GB. It is **layout 1** and was made BEFORE the speed fixes (storage cap
1024 px, full-resolution S, no working-size cap), so its numbers differ slightly from a full run
for close-range crops. Either migrate it once (no recomputation, ~5–10 min, writes a new store):

```bash
python -m zebrasight.pipeline.store --migrate-v1 cache/percepts_pilot.zarr cache/percepts_pilot_v2.zarr
```

and read it with `PerceptStore`, or read layout 1 directly:

```python
import zarr
root = zarr.open_group("cache/percepts_pilot.zarr", mode="r")
conds = root.attrs["conds"]              # ["c0", "c7", "c8", "c9"]
g = root["rows"]["<ann_id>_flank"]
ci, di = conds.index("c0"), 9
h, w = g["shape"][ci, di]
y = g["y"][ci, di, :h, :w]               # also s, rho, mask; deg_per_px[ci, di]
```

Splits for the pilot are in `data/regions_pilot.parquet` (every pilot zebra has gallery and query
views; no encounter in both).

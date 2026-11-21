# ZebraSight team pack: START HERE

What a zebra's eye sees of another zebra's coat at 2–200 m, under day, dusk, night, haze, dust and
viewing angle, so we can measure which stripe features still identify individuals.
A = data + zebra-eye model, **B = features/metrics**, **V = VR**.

## What's in this zip

| Folder | For | Contents |
|---|---|---|
| `code/` | B, V | The full repo at the latest commit: code, `configs/`, `docs/`, `tests/`, `CLAUDE.md`, `README.md`, `requirements-lock.txt` |
| `data_handoff/` | B | `zebrasight_data_handoff.zip`: region masks (rule v2), `regions.parquet` with gallery/query splits, v1/pilot tables |
| `vr_export/` | V | `vr_export.zip`: 960 textures (5 zebras × flank/face × 6 conditions × 16 distances), shader JSONs, README with every formula, placeholder LUT |
| `figures/` | everyone | Calibration, haze, montages, mask QA (`qa_grid_v2.png`) |

**Not included:** the GZGC photos (10 GB). Download them yourself (section 3).

## 1. B: regenerate all percepts (full detail: `code/docs/handoff_B.md`)

```bash
cd code
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: py -3.11 -m venv .venv; .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev] && pytest -q                         # all tests should pass
mkdir -p data/raw && cp /path/to/gzgc.coco.tar.gz data/raw/
python -m zebrasight.data.unpack                           # photos -> data/raw/gzgc/
python -m zebrasight.data.handoff unpack ../data_handoff/zebrasight_data_handoff.zip
python -m zebrasight.pipeline.store data/regions.parquet cache/percepts_full.zarr               # c0-c9
python -m zebrasight.pipeline.store data/regions.parquet cache/percepts_full.zarr --with-extras # + s/o/m
```

- Resumable: rerun the same command. Expected on an RTX 3090 (estimate): about 1 h and about
  30 GB for c0–c9, about 1.7 h and about 50 GB with the extras.
- Read percepts with `zebrasight.pipeline.read_store.PerceptStore`. Layouts and columns are in
  `code/docs/contracts.md`.
- **Skip rows flagged `valid: false`.** In the pilot store and the env-demo store, the old "face"
  rows (region rule v1) were mostly neck and are flagged invalid. `data/regions.parquet` uses rule
  v2: real heads via OWLv2 (79%), a silhouette-tip fallback (20%), and no face for 0.7% of photos.

## 2. V: the VR pack (full detail: `README.md` inside `vr_export.zip`)

- Textures: one 16-layer PNG set per zebra × region × condition. Pick the two layers around the
  live distance and blend them in **log-distance**, after sRGB-decoding.
- Colours are already the zebra's dichromat view (`params/dichromat_matrix.json`). Dawn = use dusk.
- Live haze: apply it **only to the c0 (clear) textures, never to c1–c3** (their haze is baked in).
  Note that c0 contains no haze at all. Formulas are in `params/haze.json` and the README.
- `params/light.json`, `field_of_view.json` (FOV zones approximate) and `loupe.json`
  (magnification ×1.9 for Quest 3). **The HUD shows the true distance.**
- `vr_lut.json` is a **PLACEHOLDER** (dummy values, proposed schema) until B provides real
  features.

## 3. Photos and licence

- GZGC photos are **not in this zip**: download `gzgc.coco.tar.gz` (10.43 GB) from Wild Me's
  mirror, https://wildbookiarepository.azureedge.net/datasets/gzgc.coco.tar.gz (listed on
  Academic Torrents / LILA), and put it in `code/data/raw/`.
- **Licence:** the dataset's COCO annotations mark every image **CC BY-NC-ND 2.0**; the Academic
  Torrents listing states CDLA-Permissive 1.0. Treat everything made from these photos (masks,
  percepts, textures, montages) as **internal use** until the organisers confirm. Cite
  Parham et al. (2017), *Animal population censusing at scale with citizen science and
  photographic identification*.

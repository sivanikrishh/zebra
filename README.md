# ZebraSight

Simulates what a zebra's eye sees when it looks at another zebra's coat at 16 distances
(2-200 m, log-spaced) under 10 conditions, so we can measure which stripe features still
identify individuals.

Team: A = data + zebra-eye model, B = features/metrics, V = VR.

## Setup

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
pytest
```

Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
pytest
```

`requirements-lock.txt` pins the exact versions that were tested on the Windows dev machine.

## Layout

| Path | Contents |
|---|---|
| `zebrasight/types.py` | `ViewState`, `ViewCond`, `Percept` |
| `zebrasight/pipeline/` | stage registry + `Pipeline.run` |
| `zebrasight/eye/` | geometry and zebra-eye stages (dichromacy, streak, csf, motion, noise) |
| `zebrasight/env/` | scene and atmosphere stages (linear RGB) |
| `zebrasight/{data,features,metrics,plots}` | data loading, B's features/metrics, figures |
| `configs/` | `grid.yaml` (distances x conditions), `eye.yaml` (literature parameters), `melin_table2.yaml` |
| `docs/` | `architecture.md`, `eye_parameters.md` |
| `vr/` | V's VR work |

## Data and licence

Photos: Wild Me **GZGC-IBEIS** (plains zebras, Nairobi National Park, 2015), licensed
**CC BY-NC-ND 2.0** (Attribution, NonCommercial, NoDerivatives), organiser-provided. Put the archive at
`data/raw/gzgc.coco.tar.gz` and run `python -m zebrasight.data.unpack`. Everything derived from
these images (masks, percepts, QA grids, VR textures) is **internal – not for public display**
until use of derived images is confirmed; charts of statistics are fine. See `docs/data_report.md`.

See `docs/architecture.md` for the design rules and `docs/eye_parameters.md` for every
parameter with its citation and evidence grade.

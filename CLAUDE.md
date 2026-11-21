# ZebraSight — project context for Claude

Hackathon project (demo Monday). We simulate what a zebra's eye sees when it looks at
another zebra's coat at 16 distances (2-200 m, log-spaced) and 10 conditions (c0-c9), then
measure which stripe features identify individuals.

- Person A: data + zebra-eye model. B: features/metrics. V: VR.
- No zebra spatial-resolving-power estimate exists yet (the Wits plains zebra project is
  planned, not published), so every eye parameter is a horse proxy. See
  `docs/eye_parameters.md` for values, sweeps, citations and evidence grades (A/B/C).

## Environment

- Native Windows 11 (no WSL), PowerShell, RTX 3050 Laptop 6 GB, 16 GB RAM, i7. No Docker.
- Python 3.11 venv in `.venv` (`py -3.11 -m venv .venv; .venv\Scripts\activate`).
  System Python 3.14 is too new for the CUDA stack; do not use it.
- torch from `https://download.pytorch.org/whl/cu126`, then `pip install -e .[dev]`.
- The same repo must run on Linux: use pathlib everywhere, no hard-coded path separators,
  no OS-specific shell scripts as entry points (use Python).
- Any joblib/multiprocessing code must run under `if __name__ == "__main__":` (Windows spawn).
- Keep GPU memory under 6 GB: batch accordingly.
- Run tests with `pytest`.

## DESIGN RULES

1. **One pipeline, ordered stages:** geometry -> scene -> atmosphere -> eye. Each stage is
   a `torch.nn.Module` with `forward(state) -> state`, registered by name
   (`@register_stage("name")`) and switchable in config. Phase 1 baseline = geometry + eye;
   scene and atmosphere exist as no-op stubs.
2. **Colour is preserved until the eye.** Scene and atmosphere work in linear RGB
   (3 channels), because coloured airlight (haze), shadows and grass backgrounds come later.
   Only the eye converts RGB -> zebra cone signals [L, S] (rod channel at night).
3. **Spatial maps:** the state carries per-pixel visual-field coordinate maps
   (`az_map`, `el_map`, degrees), so visual-streak acuity can vary across the image (needed
   for full scenes and VR). For a single crop: centre position (`field_az_deg`,
   `field_el_deg`) + pixel offsets.
4. **The region mask** (face/flank/rump) travels inside the state and is resampled with
   the image.
5. **IDs are metadata.** `zebra_id`, `image_id`, `session_id`, `region` travel in
   `state.meta`; maths never reads them. `zebra_id` links face and flank of the same animal.
6. **Provenance:** every stage appends the parameter values it used to `state.log`
   (via `Stage.record`).
7. **Tensors:** batched `(B, C, H, W)`; cuda if available else cpu; float32 compute,
   float16 storage (`Percept.for_storage()`).

## Parameters

- Never invent a parameter or citation. Each one in `configs/eye.yaml` has value, sweep,
  unit, citation + DOI/URL and an evidence grade: A = measured in horse, B = derived in a
  published model, C = our assumption (must be swept; goes on the limits slide).
- Mark a citation UNVERIFIED if its DOI could not be confirmed.
- Keep `configs/eye.yaml` and `docs/eye_parameters.md` in sync.

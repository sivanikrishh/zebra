"""Time the geometry + eye pipeline per 256x256 image: ``python -m zebrasight.eye.bench``."""

from __future__ import annotations

import time

import torch

from zebrasight.config import make_cond
from zebrasight.pipeline import Pipeline


def bench(device: str, batch: int, cond_id: str = "c0", reps: int = 10) -> float:
    pipe = Pipeline(["geometry", "eye"], device=device)
    img = torch.rand(batch, 3, 256, 256)
    mask = torch.ones(batch, 1, 256, 256)
    # 0.6 cm/px at 34.4 m is ~0.01 deg/px: the working grid, no resampling
    cond = make_cond(cond_id, 34.4, speed_mps=2.0)
    pipe.run(img, 0.6, mask, cond)  # warm-up (cuFFT plans, matrix precompute)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        pipe.run(img, 0.6, mask, cond)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / batch * 1000.0


def main() -> None:
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for cond_id in ("c0", "c9"):
            for batch in (1, 16):
                ms = bench(device, batch, cond_id)
                print(f"{device:4s} {cond_id} batch {batch:2d}: {ms:7.1f} ms / image")


if __name__ == "__main__":
    main()

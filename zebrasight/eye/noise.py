"""Noise stage: photon (Poisson) + neural (Gaussian) noise, seeded and device-independent.

Photon noise: the mean photon count per pixel is
    N = q0 * luminance_cd_m2 * t_int_s * signal
with q0 = ``noise.photon_scale`` (photons per cd/m2 per second per pixel at signal 1;
grade C, sweep 1e6-1e8). The output is Poisson(N) / (q0 * luminance * t_int), so the mean
is unchanged and relative noise grows as light falls: with q0 = 1e7 the white-level count
is ~4.8e6 by day, ~4.8e4 at dusk and ~16 on a moonless night (SNR ~4).
Neural noise: additive Gaussian, sigma = ``noise.neural_sigma`` x the image mean signal
(grade C, sweep 0-0.05, bounded above by the 0.42 Weber fraction).

Each image gets its own seed, hash(image_id, region, distance_m, cond_id, cond.seed), so noise
differs between images but is reproducible. Random numbers (one uniform for the Poisson
inverse CDF and one normal per pixel) are drawn on the CPU from a torch.Generator and then
moved to the device, so CPU and CUDA runs give the same noise.
"""

from __future__ import annotations

import hashlib

import torch

from zebrasight.registry import Stage, register_stage
from zebrasight.types import ViewState


POISSON_EXACT_BELOW = 50.0  # mean count; above it the normal approximation is used


def poisson_icdf(u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Poisson sample by inverse CDF from one uniform per pixel.

    Exact for lam < POISSON_EXACT_BELOW, normal approximation round(lam + sqrt(lam) z)
    above. Unlike torch.poisson (rejection sampling) it uses a fixed number of random
    draws, so a tiny change in lam (e.g. CPU vs CUDA rounding) cannot desynchronise the
    random stream.
    """
    k = torch.round(lam + torch.sqrt(lam) * torch.special.ndtri(u)).clamp_min(0.0)
    small = lam < POISSON_EXACT_BELOW
    if small.any():
        ls, us = lam[small], u[small]
        p = torch.exp(-ls)
        cdf = p.clone()
        ks = torch.zeros_like(ls)
        kmax = int(POISSON_EXACT_BELOW + 10 * POISSON_EXACT_BELOW**0.5 + 10)
        for n in range(1, kmax + 1):
            below = us > cdf
            if not below.any():
                break
            ks = ks + below.to(ks.dtype)
            p = p * ls / n
            cdf = cdf + p
        k[small] = ks
    return k


def _meta_item(meta: dict, key: str, i: int):
    v = meta.get(key)
    return v[i] if isinstance(v, (list, tuple)) else v


def noise_seed(meta: dict, i: int, distance_m: float, cond_id: str, base_seed: int) -> int:
    """Per-image seed = hash(image_id, region, distance_m, cond_id, base_seed).

    Stable across runs, processes and machines (blake2b, not Python's salted hash()).
    Different images, regions, distances or conditions get independent noise; rerunning the
    same one reproduces it exactly.
    """
    key = repr((_meta_item(meta, "image_id", i), _meta_item(meta, "region", i), round(float(distance_m), 6),
                str(cond_id), int(base_seed)))
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little") & (2**63 - 1)


def add_noise(x: torch.Tensor, luminance: float, t_int_s: float, q0: float, sigma: float,
              seeds: list[int]) -> torch.Tensor:
    """x (B,C,H,W); one seed per batch item."""
    xc = x.detach().to("cpu", torch.float64).clamp_min(0.0)
    scale = q0 * luminance * t_int_s
    out = torch.empty_like(xc)
    for i, seed in enumerate(seeds):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        xi = xc[i]
        u = torch.rand(xi.shape, generator=g, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
        z = torch.randn(xi.shape, generator=g, dtype=torch.float64)
        noisy = poisson_icdf(u, xi * scale) / scale
        if sigma > 0:
            noisy = noisy + sigma * xi.mean(dim=(-2, -1), keepdim=True) * z
        out[i] = noisy
    return out.to(x.device, x.dtype)


@register_stage("noise")
class Noise(Stage):
    def forward(self, state: ViewState) -> ViewState:
        c = state.cond
        key = "temporal.integration_ms_night" if c.light == "night" else "temporal.integration_ms_day"
        t = float(self.param(key)) / 1000.0
        q0 = float(self.param("noise.photon_scale"))
        sigma = float(self.param("noise.neural_sigma"))
        x = state.rod if state.rod is not None else state.cones
        seeds = [noise_seed(state.meta, i, c.distance_m, c.cond_id, c.seed) for i in range(x.shape[0])]
        x = add_noise(x, c.luminance_cd_m2, t, q0, sigma, seeds)
        state = state.evolve(rod=x) if state.rod is not None else state.evolve(cones=x)
        self.record(state, model=self.param("noise.model"), photon_scale=q0, neural_sigma=sigma,
                    integration_ms=t * 1000.0, base_seed=c.seed, seeds=seeds,
                    white_photon_count=q0 * c.luminance_cd_m2 * t)
        return state

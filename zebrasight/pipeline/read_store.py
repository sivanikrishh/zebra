"""Read percepts from a layout-v2 store (see zebrasight/pipeline/store.py).

    from zebrasight.pipeline.read_store import PerceptStore
    st = PerceptStore("cache/percepts_pilot.zarr")
    st.rows()                        # DataFrame: one row per (ann_id, region) with IDs and split
    p = st.get("12345_flank", "c7", 4)   # aliases resolved (c7 -> c8); distance index 0..15
    p["y"], p["s"], p["rho"], p["mask"], p["deg_per_px"], p["distance_m"]

All arrays are cropped to their valid area. ``s`` is at half resolution (2 x deg_per_px) and
None at night. Opening is read-only and loads nothing until ``get`` is called.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr


class PerceptStore:
    def __init__(self, path: str | Path):
        self.root = zarr.open_group(str(path), mode="r")
        if self.root.attrs.get("layout_version") != 2:
            raise ValueError(f"{path}: expected layout_version 2, got {self.root.attrs.get('layout_version')}")
        self.distances = list(self.root.attrs["distances_m"])
        self.aliases = dict(self.root.attrs.get("aliases", {}))
        self._rows = self.root["rows"]

    def rows(self) -> pd.DataFrame:
        recs = []
        for k in self._rows.group_keys():
            g = self._rows[k]
            conds = sorted(c for c in g.group_keys() if g[c].attrs.get("complete") or g[c].attrs.get("alias_of"))
            recs.append({"key": k, **dict(g.attrs), "conds": conds})
        return pd.DataFrame(recs)

    def resolve(self, cond: str) -> str:
        return self.aliases.get(cond, cond)

    def get(self, key: str, cond: str, d_index: int) -> dict:
        g = self._rows[key][self.resolve(cond)]
        h, w = (int(v) for v in g["shape"][d_index])
        out = {
            "y": np.asarray(g["y"][d_index, :h, :w], np.float32),
            "rho": np.asarray(g["rho"][d_index, :h, :w], np.float32),
            "mask": np.asarray(g["mask"][d_index, :h, :w]) > 0,
            "s": None,
            "deg_per_px": float(g["deg_per_px"][d_index]),
            "display_limited": bool(g["display_limited"][d_index]),
            "source_limited": bool(g["source_limited"][d_index]),
            "distance_m": self.distances[d_index],
            "cond": cond,
            "computed_as": self.resolve(cond),
        }
        if "s" in g.array_keys():
            hs, ws = (int(v) for v in g["s_shape"][d_index])
            out["s"] = np.asarray(g["s"][d_index, :hs, :ws], np.float32)
        return out

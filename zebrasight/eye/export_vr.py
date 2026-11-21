"""Export the dichromat matrices for V's VR shader: ``python -m zebrasight.eye.export_vr``."""

from __future__ import annotations

from zebrasight.config import REPO_ROOT, load_config
from zebrasight.eye.dichromacy import export_display_matrices

OUT = REPO_ROOT / "vr" / "shaders" / "dichromat_matrix.json"


def main() -> None:
    cones = load_config("eye").cones
    payload = export_display_matrices(OUT, cones.L_peak.value, cones.S_peak.value, cones.rod_peak.value)
    print(f"wrote {OUT} (neutral point {payload['neutral_point_nm']:.1f} nm)")


if __name__ == "__main__":
    main()

"""Positional observers (Part C). Registry: name -> object with .name, .needs_alignment, .score(g, q).

Positional (oracle-aligned): gabor_phase, gabor_magnitude, orientation_field, aligned_minutiae,
registered_layout_oracle. Controls without position: gabor_pooled, orientation_pooled.
Own alignment: hotspotter_self.
"""

from __future__ import annotations

from zebrasight.positional.observers import (aligned_minutiae, gabor_magnitude, gabor_phase, gabor_pooled,
                                             orientation_field, orientation_pooled)
from zebrasight.positional.observers.existing import hotspotter_self, registered_layout_oracle

OBSERVERS = {o.name: o for o in (gabor_phase, gabor_magnitude, gabor_pooled, orientation_field, orientation_pooled,
                                 aligned_minutiae, registered_layout_oracle, hotspotter_self)}

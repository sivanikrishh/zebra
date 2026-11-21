# Region rule v2 QA

`qa_grid_v2.png`: 24 zebras from the full-run subset (12 chosen as head-down grazing poses by a
silhouette heuristic, 12 random) with face (orange), flank (blue), rump (green) and the chosen
OWLv2 head box (yellow). Regenerate: `python -m zebrasight.data.heads qa`.

Score threshold 0.15: tuned by eye on this set. Head-sized boxes in the front 35% of the body
scored 0.15–0.54; whole-zebra boxes (which OWLv2 returns for "head of a zebra") are rejected by
size, not by score.

By eye: face correct in 21/24 (all 12 grazing poses included); 3 questionable: PZ_0184 (box on a
neighbour's head behind the animal), PZ_1234 and PZ_0081 (silhouette tip catches chest/neck because
the head is not the front-most point). Flank and rump correct in 24/24.

Facing convention verified: for "left" views the animal's head is on the image left.
Source photos: GZGC (CC BY-NC-ND 2.0).

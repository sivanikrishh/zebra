"""VR export pack for V: textures + shader parameters generated from the model's configs.

    python -m zebrasight.vrpack            # params + textures + placeholder LUT + vr_export.zip

Everything is written to vr/export/ (documented in vr/export/README.md). The shader JSONs are
generated from configs/*.yaml and the model's own functions, never hand-typed;
tests/test_vrpack.py fails if they drift from the model.
"""

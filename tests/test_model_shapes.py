"""
Checkpoint smoke test.

[2026.07] detector_final.py now loads a single Desklib binary AI-text
detector via from_pretrained (Hugging Face Hub, or a local snapshot via
DESKLIB_LOCAL_PATH / XPLAGIAX_EVAL_WEIGHTS) instead of three manually placed
ModernBERT weight files. This test no longer gates on local .bin files
existing on disk — it needs network access (or a local snapshot) to actually
load the model, so it stays opt-in:

    RUN_MODEL_SMOKE=1 pytest tests/test_model_shapes.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_SMOKE") != "1",
    reason="model smoke test is opt-in (RUN_MODEL_SMOKE=1) and needs network access "
           "or a local snapshot (DESKLIB_LOCAL_PATH / XPLAGIAX_EVAL_WEIGHTS)",
)


def test_classifier_head_is_binary():
    from app.engine import detector_final as df

    assert df.model.classifier.out_features == 1
    assert len(df.label_mapping) == 2


def test_classify_text_returns_detection_result():
    from app.engine import detector_final as df

    msg, fig, result = df.classify_text("The quick brown fox jumps over the lazy dog.")
    assert fig is None  # generate_plot defaults to False
    assert result.prediction in ("Human", "AI")
    assert 0 <= result.human_percentage <= 100
    assert 0 <= result.ai_percentage <= 100

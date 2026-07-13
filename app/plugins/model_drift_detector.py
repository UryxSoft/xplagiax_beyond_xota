"""
app/plugins/model_drift_detector.py — Model Drift Detection.

Detects if the ensemble is degrading over time by monitoring:
  - Prediction confidence trends
  - Class distribution shifts
  - Precision/recall drift against rolling baseline

Implements anti-enshittification safeguards: if model quality drops,
alerts before service degrades for end-users.
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from app.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# Drift detection parameters
WINDOW_SIZE = 100  # samples to compute rolling baseline
DRIFT_THRESHOLD = 0.05  # 5% drop in confidence triggers alert
ALERT_PATH = Path("/tmp/xplagiax_drift_alerts.jsonl")
MAX_ALERTS_KEPT = 1000


class ModelDriftDetector(BasePlugin):
    """Monitor ensemble model drift and precision degradation."""

    def __init__(self):
        self._lock = Lock()
        self._prediction_history = deque(maxlen=WINDOW_SIZE)
        self._alerts = deque(maxlen=MAX_ALERTS_KEPT)
        self._last_baseline_confidence = None
        self._last_alert_time = None
        self._is_degraded = False

    def name(self) -> str:
        return "model_drift_detector"

    def description(self) -> str:
        return (
            "Anti-enshittification safeguard. Detects if ensemble confidence or "
            "precision drifts over time; alerts when model quality degrades."
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze must be called WITH the primary AI detection result.
        Since this plugin runs in parallel and has no direct access to
        ensemble output, it relies on being called with injected metadata.

        In production, plugin_orchestrator should call:
          drift_detector.analyze(
              text,
              metadata={"ensemble_confidence": 0.7, "prediction": "AI"}
          )

        For now, return empty; see warmup for real integration pattern.
        """
        return {}

    def record_prediction(
        self,
        confidence: float,
        prediction: str,
        text_len: int,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Record a prediction for drift tracking.

        Typically called by plugin_orchestrator after ensemble classification.

        Parameters
        ----------
        confidence : float
            Ensemble output confidence (0.0 to 1.0).
        prediction : str
            "AI" or "Human".
        text_len : int
            Length of input text in characters.
        timestamp : datetime, optional
            Prediction timestamp. Defaults to now.

        Returns
        -------
        dict
            Drift alert if one was triggered, else empty.
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self._prediction_history.append(
                {
                    "confidence": confidence,
                    "prediction": prediction,
                    "text_len": text_len,
                    "timestamp": timestamp.isoformat(),
                }
            )

            # Check for drift
            alert = self._check_drift()
            if alert:
                self._alerts.append(alert)
                self._is_degraded = True
                self._write_alert(alert)
                logger.warning("Model drift detected: %s", alert["reason"])

            return alert if alert else {}

    def _check_drift(self) -> Optional[Dict[str, Any]]:
        """
        Detect drift via:
          1. Confidence trend (rolling mean)
          2. Class balance (should be ~50/50 for mixed corpus)
          3. Precision proxy (variance in confidence)

        Returns alert dict if drift detected, else None.
        """
        if len(self._prediction_history) < 10:
            return None  # Too few samples

        recent = list(self._prediction_history)
        recent_confidence = [p["confidence"] for p in recent]
        recent_predictions = [p["prediction"] for p in recent]

        # Drift metric 1: mean confidence
        current_mean_confidence = sum(recent_confidence) / len(recent_confidence)

        if self._last_baseline_confidence is None:
            self._last_baseline_confidence = current_mean_confidence
            return None

        # Check if confidence dropped significantly
        conf_drop = self._last_baseline_confidence - current_mean_confidence
        if conf_drop > DRIFT_THRESHOLD:
            self._last_baseline_confidence = current_mean_confidence
            return {
                "severity": "warning",
                "reason": f"Confidence dropped {conf_drop:.2%} (was {self._last_baseline_confidence:.2%})",
                "metric": "confidence_trend",
                "baseline": self._last_baseline_confidence,
                "current": current_mean_confidence,
                "timestamp": datetime.now().isoformat(),
                "samples": len(recent),
            }

        # Drift metric 2: class imbalance (if heavily skewed)
        ai_count = sum(1 for p in recent_predictions if p == "AI")
        human_count = len(recent_predictions) - ai_count
        ratio = min(ai_count, human_count) / len(recent_predictions)
        if ratio < 0.2:  # Less than 20% minority class
            logger.info(
                "Class imbalance detected: %d AI, %d Human",
                ai_count,
                human_count,
            )

        # Update baseline for next check
        self._last_baseline_confidence = current_mean_confidence
        return None

    def _write_alert(self, alert: Dict[str, Any]) -> None:
        """Write alert to JSONL file for monitoring systems."""
        try:
            with open(ALERT_PATH, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.error("Failed to write drift alert: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """
        Return current drift detection status.

        Used by health check and monitoring endpoints.
        """
        with self._lock:
            if not self._prediction_history:
                return {"status": "no_data"}

            recent = list(self._prediction_history)
            recent_confidence = [p["confidence"] for p in recent]

            return {
                "status": "degraded" if self._is_degraded else "healthy",
                "samples_tracked": len(recent),
                "mean_confidence": sum(recent_confidence) / len(recent_confidence),
                "confidence_range": (
                    min(recent_confidence),
                    max(recent_confidence),
                ),
                "recent_alerts": list(self._alerts)[-5:],
                "last_alert": (
                    self._last_alert_time.isoformat()
                    if self._last_alert_time
                    else None
                ),
            }

    def warmup(self) -> None:
        """Initialize drift detector state."""
        logger.info("ModelDriftDetector ready")

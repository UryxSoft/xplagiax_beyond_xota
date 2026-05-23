"""
app/tasks.py — Background tasks for Celery.
"""
import time
import logging
from flask import current_app
from app.celery_app import celery

logger = logging.getLogger(__name__)

@celery.task(bind=True, name="analyze_document_task")
def analyze_document_task(self, payload):
    """
    Executes the heavy document analysis.
    payload contains: text, plugins, max_tokens
    """
    t0 = time.perf_counter()
    
    text = payload.get("text", "")
    plugins_requested = payload.get("plugins", ["ai_detection"])
    max_tokens = int(payload.get("max_tokens", 150))
    
    registry = current_app.config["PLUGIN_REGISTRY"]
    timeout = current_app.config.get("PLUGIN_TIMEOUT", 30)

    # 1. Run standard plugins
    results = registry.run(plugins_requested, text, timeout=timeout)

    # 2. Run heavy segmentation
    doc_result = {}
    try:
        import app.engine
        from detector_final import analyze_long_document
        doc_result = analyze_long_document(text, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("analyze_long_document failed in task: %s", exc)

    segments = doc_result.get("segments", [])

    # 3. Merge results
    if "ai_detection" in results and doc_result:
        ai_result = results["ai_detection"]
        if ai_result.get("status") == "ok" and isinstance(ai_result.get("data"), dict):
            ai_result["data"]["segments"] = segments
            ai_result["data"]["overall_summary"] = doc_result.get("overall_summary", {})

    elapsed = time.perf_counter() - t0

    return {
        "status": "ok",
        "word_count": len(text.split()),
        "plugins_requested": plugins_requested,
        "results": results,
        "total_elapsed_ms": round(elapsed * 1000, 1),
    }

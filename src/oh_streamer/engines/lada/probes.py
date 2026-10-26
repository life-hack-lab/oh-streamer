"""Accessors for the FrameRestorer internals the player observes (queues, clip length).

Split out of player/buffer_progress.py, processing/detect_gate.py and player/controller.py (engine-decoupling phase 2);
behavior unchanged. The estimation / gating logic stays with its callers; this is the only place that knows the
attribute names of Lada's pipeline. Pure python (no torch / lada import).
"""


def progress_queues(restorer):
    """(detection queue, mosaic clip queue, restored clip queue) of `restorer`; AttributeError if it has none."""
    return restorer.frame_detection_queue, restorer.mosaic_clip_queue, restorer.restored_clip_queue


def gate_queues(restorer):
    """(detection queue, output frame queue) of `restorer`; AttributeError if it lacks them or max_clip_length."""
    det_q = restorer.frame_detection_queue
    out_q = restorer.frame_restoration_queue
    restorer.max_clip_length
    return det_q, out_q


def detection_queue(restorer):
    """The queue the detector puts one entry per scanned frame on (None if there is none)."""
    return getattr(restorer, "frame_detection_queue", None)


def pipeline_fill(restorer) -> dict:
    """{"feeder"|"clips"|"out": (qsize, maxsize)} for the info panel; AttributeError if the restorer has no such queues."""
    feeder = restorer.mosaic_detector.frame_feeder_queue
    return {"feeder": (feeder.qsize(), feeder.maxsize),
            "clips": (restorer.mosaic_clip_queue.qsize(), restorer.mosaic_clip_queue.maxsize),
            "out": (restorer.frame_restoration_queue.qsize(), restorer.frame_restoration_queue.maxsize)}


def request_cancel(restorer) -> None:
    """Ask the restoration of `restorer`'s current clip to stop (the cancel event of the proxy model built by
    cancel.cancellable_restoration_model; no-op without one). Lives here, not in cancel.py, to stay torch-free."""
    cancel = getattr(restorer.mosaic_restoration_model, "cancel_event", None)
    if cancel is not None:
        cancel.set()


def clear_cancel(restorer) -> None:
    """Re-arm the restoration model's cancel event after request_cancel (the model is shared across restorers)."""
    cancel = getattr(restorer.mosaic_restoration_model, "cancel_event", None)
    if cancel is not None:
        cancel.clear()

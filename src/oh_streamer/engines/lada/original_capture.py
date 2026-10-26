# SPDX-License-Identifier: AGPL-3.0-only
# Wraps lada FrameRestorer._restore_frame at run time (upstream lada, AGPL-3.0). Modified 2026-10-26; see THIRD_PARTY_NOTICES.md, section
# "Modifications to upstream software".
"""Original-frame capture for the Lada engine: patches FrameRestorer._restore_frame.

Moved from processing/compare_stack.py (engine-decoupling phase 2); behavior unchanged. The generic helpers
(capture_boxes, rebuild_original, ...) stay in compare_stack.
"""

from oh_streamer.processing.compare_stack import capture_boxes


def install_original_capture(frame_restorer_cls=None) -> bool:
    """Patch FrameRestorer._restore_frame once so frames get frame._lada_orig when restorer.capture_original is true.
    Returns True if installed now, False if it already was."""
    cls = frame_restorer_cls
    if cls is None:
        from lada.restorationpipeline.frame_restorer import FrameRestorer as cls   # noqa: PLC0415 - heavy import
    if getattr(cls, "_lada_compare_hook", False):
        return False
    orig = cls._restore_frame

    def _restore_frame(self, frame, frame_num, restored_clips):
        if getattr(self, "capture_original", False):
            boxes = [c.boxes[0] for c in restored_clips if c.frame_start == frame_num]
            if boxes:
                capture_boxes(frame, boxes)
        return orig(self, frame, frame_num, restored_clips)

    cls._restore_frame = _restore_frame
    cls._lada_compare_hook = True
    return True

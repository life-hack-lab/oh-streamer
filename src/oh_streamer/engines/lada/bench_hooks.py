"""Lada-internal patches used by the performance check (perf/bench.py).

Moved from perf/bench.py (engine-decoupling phase 2); behavior unchanged. Imports torch / lada lazily, inside the
functions, like the code it came from. Every hook takes the engine session (engine.py LadaSession; capability
BENCHMARK_HOOKS) and works on the FrameRestorer underneath it, which stays inside this package.
"""

import time


def _restorer(session):
    r = session._restorer
    if r is None:
        raise RuntimeError("session is closed")
    return r


def count_restored_clips(session) -> dict:
    """Wrap the restorer's clip restoration so the child can prove injected detections became restored clips."""
    restorer = _restorer(session)
    stats = {"clips": 0, "clip_frames": 0}
    orig = restorer._restore_clip

    def counted(clip, *a, **kw):
        stats["clips"] += 1
        try:
            stats["clip_frames"] += len(clip.frames)
        except (TypeError, AttributeError):
            pass
        return orig(clip, *a, **kw)

    restorer._restore_clip = counted
    return stats


def install_synthetic_detections(session, density: int, region_boxes):
    """Replace the detector's scene builder on this instance (see perf/bench.py install_synthetic_detections).
    region_boxes(frame_idx, density, w, h) supplies the boxes of one frame."""
    import torch
    from lada.restorationpipeline.mosaic_detector import Scene
    det = _restorer(session).mosaic_detector

    def create_or_append(results, scenes, frame_num):
        if density <= 0:
            return
        h, w = int(results.orig_shape[0]), int(results.orig_shape[1])
        dev = results.orig_img.device
        for box in region_boxes(frame_num - det.start_frame, density, w, h):
            t, l, b, r = box
            mask = torch.zeros((h, w, 1), dtype=torch.uint8, device=dev)
            mask[t:b, l:r] = 255
            current = None
            for scene in scenes:
                if scene.belongs(box):
                    current = scene
                    if scene.frame_end == frame_num:
                        scene.merge_mask_box(mask, box)
                    else:
                        scene.add_frame(frame_num, results.orig_img, mask, box)
                    break
            if current is None:
                current = Scene(det.video_meta_data.video_file, det.video_meta_data)
                scenes.append(current)
                current.add_frame(frame_num, results.orig_img, mask, box)

    det._create_or_append_scenes_based_on_prediction_result = create_or_append


def watch_detector_done(session) -> dict:
    """Record when the detector finishes the file (it puts EOF on frame_detection_queue): {"t": perf_counter or None}."""
    from lada.utils.threading_utils import EOF_MARKER
    seen = {"t": None}
    q = _restorer(session).frame_detection_queue
    orig = q.put

    def put(item, *a, **kw):
        if item is EOF_MARKER and seen["t"] is None:
            seen["t"] = time.perf_counter()
        return orig(item, *a, **kw)

    q.put = put
    return seen

"""Pure frame-rate planning plus lightweight tests of the reader/factory seams."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path as _Path
import tempfile
import sys as _sys
from types import SimpleNamespace
import unittest

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.processing.frame_rate import planned_fps, selected_frame
from oh_streamer.player import player_settings as PS


_IO_HOOKS_SOURCE = _SOURCE_ROOT / "oh_streamer" / "engines" / "lada" / "io_hooks.py"
_ADAPTER_SOURCE = _SOURCE_ROOT / "oh_streamer" / "engines" / "lada" / "adapter.py"


def _reader_class():
    hook = next(n for n in ast.parse(_IO_HOOKS_SOURCE.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "install_lada_io_hooks")
    node = next(n for n in hook.body if isinstance(n, ast.ClassDef) and n.name == "HttpVideoReader")

    class ReaderBase:
        def seek(self, _offset_ns):
            pass

    namespace = {"_LADA_VIDEO_READER": ReaderBase}
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(_IO_HOOKS_SOURCE), "exec")
    exec(code, namespace)
    return namespace["HttpVideoReader"], namespace


def _factory_make():
    tree = ast.parse(_ADAPTER_SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RestorerFactory")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "make")
    registrations = []

    class FakeRestorer:
        def __init__(self, *args):
            self.args = args
            self.registered_meta = registrations[-1].meta

    namespace = {
        "FrameRestorer": FakeRestorer,
        "Fraction": Fraction,
        "planned_fps": planned_fps,
        "replace": __import__("dataclasses").replace,
        "register_source": registrations.append,
        "cancellable_restoration_model": lambda model: model,
    }
    code = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(_ADAPTER_SOURCE), "exec")
    exec(code, namespace)
    return namespace["make"], FakeRestorer, registrations


@dataclass
class _VideoMeta:
    video_fps_exact: Fraction
    video_fps: float
    average_fps: float
    frames_count: int
    duration: float
    time_base: Fraction
    start_pts: object
    video_file: str = "source"
    video_width: int = 1920
    video_height: int = 1080


def _reader_output(*, rate, stride, time_base, pts, start_pts=0, target_s=None):
    converted_pts = []

    class FakeFrame:
        def __init__(self, value):
            self.pts = value

        def to_ndarray(self, format):
            if format != "bgr24":
                raise AssertionError(format)
            converted_pts.append(self.pts)
            return self.pts

    class FakePacket:
        def decode(self):
            return iter(FakeFrame(value) for value in pts)

    class FakeContainer:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(thread_type=None, time_base=time_base)])

        def demux(self, _stream):
            return iter((FakePacket(),))

    class FakeLogging:
        ERROR = 0

        @staticmethod
        def restore_default_callback():
            pass

        @staticmethod
        def set_libav_level(_level):
            pass

    class FakeAv:
        logging = FakeLogging()
        error = SimpleNamespace(InvalidDataError=type("InvalidDataError", (Exception,), {}))
        ERROR = 0

    class FakeTorch:
        @staticmethod
        def from_numpy(value):
            return value

    reader_cls, namespace = _reader_class()
    meta = _VideoMeta(rate, float(rate), float(rate), len(pts), 1.0, time_base, start_pts)
    if stride != 1:
        meta.restoration_stride = stride
    namespace.update({
        "_IO_SOURCES": {"source": (meta, None)},
        "av": FakeAv,
        "torch": FakeTorch,
        "is_url": lambda _path: False,
        "selected_frame": selected_frame,
        "log": lambda *_args: None,
    })
    reader = reader_cls()
    reader.file = "source"
    reader.container = FakeContainer()
    if target_s is not None:
        reader.seek(round(target_s * 1e9))
    result = [(frame, frame_pts) for frame, frame_pts in reader.frames()]
    return [frame_pts for _frame, frame_pts in result], converted_pts


class PlannedFpsTest(unittest.TestCase):
    def test_original_mode_preserves_exact_rate(self):
        for fps in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001), Fraction(60)):
            with self.subTest(fps=fps):
                self.assertEqual(planned_fps(fps, "original"), fps)

    def test_thirty_mode_caps_by_integer_stride(self):
        self.assertEqual(planned_fps(Fraction(60), "30"), Fraction(30))
        self.assertEqual(planned_fps(Fraction(60000, 1001), "30"), Fraction(30000, 1001))
        self.assertEqual(planned_fps(Fraction(120), "30"), Fraction(30))

    def test_thirty_mode_keeps_rates_at_or_below_thirty(self):
        for fps in (Fraction(24000, 1001), Fraction(25), Fraction(30000, 1001), Fraction(30)):
            with self.subTest(fps=fps):
                self.assertEqual(planned_fps(fps, "30"), fps)


class SelectedFrameTest(unittest.TestCase):
    def test_selects_first_frame_then_each_stride(self):
        self.assertEqual(
            [i for i in range(8) if selected_frame(i, 3)],
            [0, 3, 6],
        )

    def test_stride_one_selects_every_frame(self):
        self.assertTrue(all(selected_frame(i, 1) for i in range(8)))


class ReaderSamplingTest(unittest.TestCase):
    def test_sixty_fps_reader_converts_only_alternate_frames(self):
        got, converted = _reader_output(rate=Fraction(30), stride=2, time_base=Fraction(1, 60),
                                        pts=range(6))
        self.assertEqual(got, [0, 2, 4])
        self.assertEqual(converted, got)  # dropped frames never reach ndarray conversion

    def test_fractional_sixty_fps_reader_uses_exact_reduced_metadata(self):
        got, converted = _reader_output(rate=Fraction(30000, 1001), stride=2,
                                        time_base=Fraction(1001, 60000), pts=range(6))
        self.assertEqual(got, [0, 2, 4])
        self.assertEqual(converted, got)

    def test_seeked_readers_choose_the_same_stride_frames(self):
        kwargs = dict(rate=Fraction(30), stride=2, time_base=Fraction(1, 60),
                      pts=range(12), target_s=0.11)
        first, _ = _reader_output(**kwargs)
        second, _ = _reader_output(**kwargs)
        self.assertEqual(first, [6, 8, 10])
        self.assertEqual(second, first)

    def test_missing_and_string_start_pts_are_safe(self):
        missing, _ = _reader_output(rate=Fraction(30), stride=2, time_base=Fraction(1, 60),
                                    pts=range(6), start_pts=None)
        string, _ = _reader_output(rate=Fraction(30), stride=2, time_base=Fraction(1, 60),
                                   pts=range(900, 906), start_pts="900")
        self.assertEqual(missing, [0, 2, 4])
        self.assertEqual(string, [900, 902, 904])


class RestorerFactoryTest(unittest.TestCase):
    def test_reduced_and_original_modes_always_derive_from_native_metadata(self):
        make, _fake_restorer, registrations = _factory_make()

        class Factory:
            device = "cpu"
            max_clip_length = 240
            res_name = "model"
            det_model = res_model = pad_mode = object()
            restoration_fps = "original"

        Factory.make = make

        native = _VideoMeta(Fraction(60), 60.0, 60.0, 601, 10.02, Fraction(1, 90000), 17)
        source = SimpleNamespace(src="source", meta=native, native_meta=native)
        factory = Factory()

        factory.make(source)
        self.assertEqual(source.meta.video_fps_exact, Fraction(60))
        self.assertEqual(source.meta.frames_count, 601)

        factory.restoration_fps = "30"
        factory.make(source)
        reduced = source.meta
        self.assertEqual(reduced.video_fps_exact, Fraction(30))
        self.assertEqual(reduced.average_fps, 30.0)
        self.assertEqual(reduced.frames_count, 301)
        self.assertEqual(reduced.restoration_stride, 2)
        self.assertIs(registrations[-1].meta, reduced)
        self.assertEqual((reduced.duration, reduced.time_base, reduced.start_pts),
                         (native.duration, native.time_base, native.start_pts))

        factory.restoration_fps = "original"
        factory.make(source)
        restored = source.meta
        self.assertEqual(restored.video_fps_exact, Fraction(60))
        self.assertEqual(restored.average_fps, 60.0)
        self.assertEqual(restored.frames_count, 601)
        self.assertEqual(restored.restoration_stride, 1)
        self.assertIs(registrations[-1].meta, restored)
        self.assertIs(source.native_meta, native)
        self.assertEqual((restored.duration, restored.time_base, restored.start_pts),
                         (native.duration, native.time_base, native.start_pts))


class RestorationFpsSettingTest(unittest.TestCase):
    def test_menu_validation_and_persisted_value(self):
        self.assertEqual(PS.validate("restoration_fps", "30"), "30")
        with self.assertRaises(ValueError):
            PS.validate("restoration_fps", "60")
        menu = PS.build_menu(dict(PS.DEFAULTS), True, ["v4-fast"], ["basicvsrpp-v1.2"],
                             True, "h264_nvenc")
        item = next(item for item in menu if item["key"] == "restoration_fps")
        self.assertEqual(item["kind"], "restart")
        self.assertEqual([option["v"] for option in item["options"]], ["original", "30"])
        self.assertEqual(PS.parse_request('1:' + json.dumps({"restoration_fps": "30"}), menu),
                         {"restoration_fps": "30"})

        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory:
            path = _Path(directory) / "settings.json"
            PS.save_file(path, PS.updated_file({}, {"restoration_fps": "30"}))
            saved, warning = PS.load_file(path)
        self.assertIsNone(warning)
        self.assertEqual(saved, {"restoration_fps": "30"})


if __name__ == "__main__":
    unittest.main()

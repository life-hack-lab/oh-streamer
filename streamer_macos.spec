# Experimental streamer-specific spec. Run packaging/build_macos.py on a prepared Mac.
import os
from pathlib import Path
import sys

if sys.platform != "darwin":
    raise RuntimeError("streamer_macos.spec must be built on macOS")

ROOT = Path(SPECPATH).resolve()
sys.path.insert(0, str(ROOT / "packaging"))
from build_macos import resource_inventory
from macos_player_bundle import bundle_player
sys.path.insert(0, str(ROOT / "src"))
from oh_streamer.branding import APP_NAME
from PyInstaller.utils.hooks import collect_submodules

LADA_SOURCE = Path(os.environ["OH_STREAMER_PACKAGING_LADA_SOURCE"]).resolve()
SOURCE_ROOT = ROOT / "src"
sys.path[:0] = [str(SOURCE_ROOT), str(LADA_SOURCE)]
hidden = [
    "oh_streamer.player.oh_player",
    "oh_streamer.queue.queue_app",
    "oh_streamer.queue.queue_worker",
    "oh_streamer.perf.__main__",
    "oh_streamer.sources.resolvers.resolve_browser",
    "oh_streamer.sources.http_transport",
    "oh_streamer.platform.platform_support",
    "webview.platforms.cocoa",
]
hidden += collect_submodules("oh_streamer")
hidden += collect_submodules("lada", filter=lambda name: not (name == "lada.gui" or name.startswith("lada.gui.")))
a = Analysis([str(ROOT / "packaging" / "frozen_entry.py")],
             pathex=[str(SOURCE_ROOT), str(LADA_SOURCE)], binaries=[], datas=resource_inventory(ROOT),
             hiddenimports=hidden, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=["gi", "lada.gui"], noarchive=False)
pyz = PYZ(a.pure)
# Keep console streams: queue_worker and resolver use stdout as machine-readable IPC.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="oh-streamer", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="oh-streamer")
app = BUNDLE(coll, name=f"{APP_NAME}.app", icon=str(ROOT / "src" / "oh_streamer" / "resources" / "icons" / "app.icns"),
             bundle_identifier="local.oh.streamer", info_plist={"NSHighResolutionCapable": True})

# Add the dedicated GUI subprocess only after PyInstaller has laid out the app;
# passing it as data would let BUNDLE relocate its code/resources. This also
# reseals the outer ad-hoc signature after signing the nested helper inside-out.
bundle_player(Path(app.name), Path(os.environ["OH_STREAMER_PACKAGING_MPV"]),
              ROOT / "src/oh_streamer/resources/icons/app.icns", APP_NAME,
              Path(os.environ["OH_STREAMER_PACKAGING_MPV_NOTICES"]),
              vulkan_icd=Path(os.environ["OH_STREAMER_PACKAGING_MPV_VULKAN_ICD"])
              if os.environ.get("OH_STREAMER_PACKAGING_MPV_VULKAN_ICD") else None)

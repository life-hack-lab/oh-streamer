# Third-party notices

Copyright (C) 2026 life-hack-lab. Oh Streamer is licensed under the GNU Affero General Public License, version 3 only
(SPDX: `AGPL-3.0-only`); see the `LICENSE` file. Source: https://github.com/life-hack-lab/oh-streamer

(zh-TW) 本檔列出 Oh Streamer 隨安裝程式附帶、首次啟動時下載、以及由使用者要求下載的第三方軟體與模型的授權，
並說明我們對上游軟體所做的修改。Oh Streamer 本身以 AGPL-3.0-only 授權。這不是法律意見。

This file is installed next to `LICENSE` in the install folder. It is not legal advice.

## Written offer of corresponding source (GPL components)

For the GPL programs bundled in the installer (mpv and ffmpeg/ffprobe, section (a)), the corresponding source code
(upstream source and build recipe) is attached to the matching GitHub Release of https://github.com/life-hack-lab/oh-streamer
and stays available for at least three years after that release. You can also ask for it through
https://github.com/life-hack-lab/oh-streamer/issues.

Packages installed at first run (section (b)) are downloaded by the app directly from PyPI and download.pytorch.org;
this project does not redistribute them. Their source is available from each project, for example PyAV and the FFmpeg
build inside its wheels (including libx264 and libx265): https://github.com/PyAV-Org/PyAV and
https://github.com/PyAV-Org/pyav-ffmpeg.

## (a) Bundled in the installer

| Component | Version | License | Upstream |
|---|---|---|---|
| mpv (`mpv.exe`, shinchiro build) | v0.41.0-1227-ge471496c5 (build 20261021) | GPL-2.0-or-later | https://github.com/mpv-player/mpv ; https://github.com/shinchiro/mpv-winbuild-cmake |
| ffmpeg, ffprobe (gyan.dev `full_build`, static) | 9.0.2 | GPL-3.0 (the build is licensed as GPLv3) | https://www.gyan.dev/ffmpeg/builds/ ; https://github.com/GyanD/codexffmpeg ; FFmpeg commit 946fcce07b |
| uv | 0.12.23 | MIT OR Apache-2.0 | https://github.com/astral-sh/uv |
| python-build-standalone (CPython) | 3.13.17, release 20261021 | PSF-2.0 (plus bundled libraries; see `licenses/python-LICENSE.txt`) | https://github.com/astral-sh/python-build-standalone |
| RIFE code in `third_party/rife` (vs-rife v5.7.0, commit 784a6d0b87bc) and RIFE weights `flownet_v4.6.pkl` | v4.6 | MIT (Practical-RIFE by hzwer; vs-rife by HolyWu) | https://github.com/hzwer/Practical-RIFE ; https://github.com/HolyWu/vs-rife |
| Silkscreen font (Regular, Bold; embedded in the UI CSS) | n/a | OFL-1.1, (c) 2001 The Silkscreen Project Authors | https://github.com/googlefonts/silkscreen |
| Visual C++ runtime DLLs (msvcp140, msvcp140_1, vcomp140) | 14.44.35211 | Microsoft redistributable terms (app-local) | https://aka.ms/vs/17/release/vc_redist.x64.exe |
| WebView2 Evergreen bootstrapper (run only when the runtime is missing) | 1.3.299.15 | Microsoft proprietary terms | https://go.microsoft.com/fwlink/p/?LinkId=2124703 |
| lada wheel (restoration engine), built from upstream commit `20cb34a` | 0.11.1.dev0+20cb34a | AGPL-3.0 (= AGPL-3.0-only) | https://github.com/ladaapp/lada ; https://codeberg.org/ladaapp/lada |
| lada model weights (detection v4 accurate and fast, restoration generic v1.2) bundled with the installer | HF revision bcf461d4 | AGPL-3.0 | https://huggingface.co/ladaapp/lada |

ffmpeg and mpv are separate programs started as child processes. AGPL-3.0 and GPLv3 permit combining (AGPL section 13).
Individual license texts are installed under `licenses/`.

## (b) Installed at first run by the environment setup

On first launch the app runs `uv sync` against `uv.lock` (105 packages) and downloads the Python packages from PyPI and
download.pytorch.org. Packages with non-permissive or notable licenses:

| Package | Version | License |
|---|---|---|
| ultralytics | 8.4.4 | AGPL-3.0 |
| pystray | 0.19.5 | LGPL-3.0 |
| zeroconf | 0.151.5 | LGPL-2.1-or-later |
| av (PyAV) | 16.1.0 | BSD-3-Clause package; the wheel bundles FFmpeg libraries including GPL libx264 and libx265, plus LGPL libraries |
| opencv-python | 4.12.0.88 | Apache-2.0 (the wheel bundles LGPL FFmpeg) |
| certifi | 2026.10.23 | MPL-2.0 |
| tqdm | 4.67.1 | MPL-2.0 AND MIT |
| torch, torchvision, torchaudio (cu128 or cpu) | 2.8.0 / 2.9.1 | BSD-style; the CUDA wheels bundle NVIDIA CUDA and cuDNN libraries under the NVIDIA EULA |
| pyatv | 0.18.0 | MIT |
| playwright | 1.63.0 | Apache-2.0 (bundles a Node driver; no browser is downloaded by the app) |

Other dependencies are under permissive licenses; see `uv.lock` and each package's metadata.

At run time the app may also download the llama.cpp server (build b11617; MIT) with its `cudart` zip (NVIDIA CUDA EULA)
from https://github.com/ggml-org/llama.cpp/releases when you enable local translation subtitles.

## (c) Models downloaded at your request

Oh Streamer does not distribute any subtitle model. You download them at run time, directly from Hugging Face, pinned to a
fixed revision and verified by SHA-256. Oh Streamer does not redistribute them; their licenses apply between you and the
model authors.

| Model (catalog id) | Hugging Face repo | License as stated by the model card |
|---|---|---|
| whisper-ja-1.5B (default ASR) | efwkjn/whisper-ja-1.5B | The model card declares no license. Its rights status is unclear; check with the author before any use beyond personal use. |
| anime-whisper | litagin/anime-whisper | MIT |
| sakura-galtransl-7b-v3.7 (default translation) | SakuraLLM/Sakura-GalTransl-7B-v3.7 | CC-BY-NC-SA-4.0 |
| sakura-galtransl-14b-v3.8 | SakuraLLM/Sakura-GalTransl-14B-v3.8 | CC-BY-NC-SA-4.0 |
| sakura-7b-qwen2.5-v1.0 | SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF | CC-BY-NC-SA-4.0 |
| lada weights (if fetched separately) | https://huggingface.co/ladaapp/lada | AGPL-3.0 |

Non-commercial notice: the Sakura models are licensed CC-BY-NC-SA-4.0. They may not be used commercially (for example
paid translation services, paid patches or paid subtitle work). Use them only for non-commercial purposes.

## (d) Modifications to upstream software (AGPL-3.0 section 5a)

Modified by the Oh Streamer authors (life-hack-lab). Date of this notice: 2026-10-26.

| Upstream | Modification | Where |
|---|---|---|
| torch / mmengine weight loading | Sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (only if unset) so lada weights load on torch 2.6 or newer | `src/oh_streamer/engines/lada/runtime_patches.py` |
| ultralytics (AGPL-3.0) | Sets `YOLO_OFFLINE=true` (no telemetry or DNS lookups); wraps `ultralytics.utils.nms.non_max_suppression` so the default `max_time_img` is 0.3 instead of 0.05 | `runtime_patches.py` |
| lada `lada.utils.video_utils` | Replaces reader/metadata functions with our versions (HTTP options, frame skipping before the target frame) | `src/oh_streamer/engines/lada/io_hooks.py` |
| lada `FrameRestorer._restore_frame` | Wrapped to keep the original frame for the compare view | `src/oh_streamer/engines/lada/original_capture.py` |
| lada restoration model | Hooks to cancel an in-progress clip restoration quickly, and benchmark hooks | `src/oh_streamer/engines/lada/cancel.py`, `bench_hooks.py` |
| lada wheel `lada/__init__.py` | One line: `VERSION` is baked as `<base>+<short commit>` when the wheel is built | `packaging/build_lada_wheel.py` |

These changes are applied at run time from our own code; the lada and ultralytics sources are otherwise unmodified.
The wheel is built from upstream commit `20cb34a20a83c72c87a991d2c949032c70085b16` by `packaging/build_lada_wheel.py`,
which is in the public repository.

## (e) Trademarks

Oh Streamer is not affiliated with or endorsed by Apple, NVIDIA, Microsoft or Groq. Apple, AirPlay, NVIDIA, CUDA,
Microsoft, Windows, WebView2 and Groq are trademarks of their respective owners.

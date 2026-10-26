"""Render the app icon (rounded dark square, RGB-offset "Oh!" wordmark over an RGB band) into the packaged resources.

Design (decided 2026-10-14, lab page: testdata/icon_experiment/index.html):
  * >= 40 px: Arial Black "Oh!" drawn as red/green/blue copies offset horizontally and added together (the queue
    window's pixel logo), band underneath as wide as the ink, 12 % of the icon tall, 6 % gap.
  * <= 32 px: the offset smears small glyphs, so those sizes use hand-hinted bitmap letters in plain white; the RGB
    survives only in the band.

Outputs (committed, so macOS builds do not need the Windows font):
  src/oh_streamer/resources/icons/app.ico   Windows exe + tray (16-256 px, every size drawn natively)
  src/oh_streamer/resources/icons/app.icns  macOS bundle (Apple grid: 824 px tile centred on a 1024 canvas)
  src/oh_streamer/resources/icons/app.png   256 px tile for non-Windows trays

Run on Windows from the streamer folder:  ..\\lada\\.venv\\Scripts\\python.exe tools\\make_app_icon.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "oh_streamer" / "resources" / "icons"
FONT_FILE = Path(r"C:\Windows\Fonts\ariblk.ttf")

TEXT = "Oh!"
BG = (0x13, 0x17, 0x25)
WHITE = (255, 255, 255)
RGB = [(0xFF, 0x47, 0x66), (0x36, 0xDB, 0x86), (0x4A, 0x86, 0xFF)]
RADIUS = 0.22      # corner radius / icon size
PAD = 0.14         # horizontal room left and right of the wordmark
TEXT_H = 0.36      # max ink height / icon size
BAND = 0.12        # band height / icon size
GAP = 0.06         # text-to-band gap / icon size
OFFSET = 4 / 256   # RGB offset / icon size (min 1 px)
SS = 4             # supersampling for the vector sizes
PIXEL_MAX = 32     # sizes up to this use the bitmap letters

ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]

# Hand-hinted bitmap letters: 7 rows (16-20 px), 10 rows (24 px), 14 rows (32 px). Counters stay >= 1/3 of the width.
GLYPHS = {
    7: {
        "O": [".###.", "##.##", "##.##", "##.##", "##.##", "##.##", ".###."],
        "h": ["##...", "##...", "####.", "##.##", "##.##", "##.##", "##.##"],
        "!": ["##", "##", "##", "##", "##", "..", "##"],
    },
    10: {
        "O": [".#####.", "##...##", "##...##", "##...##", "##...##", "##...##", "##...##", "##...##", "##...##", ".#####."],
        "h": ["##.....", "##.....", "##.....", "##.###.", "###..##", "##...##", "##...##", "##...##", "##...##", "##...##"],
        "!": ["##", "##", "##", "##", "##", "##", "##", "..", "##", "##"],
    },
    14: {
        "O": ["..#####..", ".#######."] + ["###...###"] * 10 + [".#######.", "..#####.."],
        "h": ["###......"] * 4 + ["###.####.", "#########", "####..###"] + ["###...###"] * 7,
        "!": ["###"] * 9 + ["..."] * 2 + ["###"] * 3,
    },
}


def _band(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    for i, color in enumerate(RGB):
        x0, x1 = x + round(i * w / 3), x + round((i + 1) * w / 3)
        draw.rectangle((x0, y, x1 - 1, y + h - 1), fill=color)


def _tile(size: int) -> Image.Image:
    """Rounded background, antialiased by supersampling."""
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle((0, 0, size * SS - 1, size * SS - 1), radius=round(size * SS * RADIUS),
                                          fill=BG + (255,))
    return big.resize((size, size), Image.LANCZOS)


def _fit_font(width: float, height: float) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    px = int(height * 1.6)
    while px > 4:
        font = ImageFont.truetype(str(FONT_FILE), px)
        box = font.getbbox(TEXT)
        if box[2] - box[0] <= width and box[3] - box[1] <= height:
            return font, box
        px -= 1
    raise RuntimeError("wordmark does not fit")


def vector_icon(size: int) -> Image.Image:
    off = max(1, round(size * OFFSET)) * SS          # whole output pixels, so the fringes stay crisp
    s = size * SS
    font, box = _fit_font(s * (1 - 2 * PAD), s * TEXT_H)
    tw, th = box[2] - box[0], box[3] - box[1]
    band, gap = max(SS, round(size * BAND) * SS), round(size * GAP) * SS
    top = (s - (th + gap + band)) // 2
    left = (s - tw) // 2
    origin = (left - box[0], top - box[1])

    def layer(color: tuple[int, int, int], dx: int) -> Image.Image:
        im = Image.new("RGB", (s, s), (0, 0, 0))
        ImageDraw.Draw(im).text((origin[0] + dx, origin[1]), TEXT, font=font, fill=color)
        return im

    # additive mix onto the background: overlaps become white, the edges keep a single colour (the logo's 'lighter')
    art = Image.new("RGB", (s, s), BG)
    for color, dx in zip(RGB, (-off, 0, off)):
        art = ImageChops.add(art, layer(color, dx))
    _band(ImageDraw.Draw(art), left, top + th + gap, tw, band)
    shape = Image.new("L", (s, s), 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, s - 1, s - 1), radius=round(s * RADIUS), fill=255)
    art.putalpha(shape)
    return art.resize((size, size), Image.LANCZOS)


def pixel_icon(size: int) -> Image.Image:
    rows = 14 if size >= 32 else 10 if size >= 24 else 7
    glyphs = GLYPHS[rows]
    space = 2 if size >= 24 else 1
    w = sum(len(glyphs[c][0]) for c in TEXT) + space * (len(TEXT) - 1)
    band, gap = max(2, round(size * BAND)), max(1, round(size * GAP))
    x0, y0 = (size - w) // 2, (size - (rows + gap + band)) // 2
    out = _tile(size)
    d = ImageDraw.Draw(out)
    x = x0
    for ch in TEXT:
        for ry, row in enumerate(glyphs[ch]):
            for rx, cell in enumerate(row):
                if cell == "#":
                    d.point((x + rx, y0 + ry), fill=WHITE + (255,))
        x += len(glyphs[ch][0]) + space
    _band(d, x0, y0 + rows + gap, w, band)
    return out


def icon(size: int) -> Image.Image:
    return pixel_icon(size) if size <= PIXEL_MAX else vector_icon(size)


def mac_icon(size: int) -> Image.Image:
    """Apple icon grid: the tile is 824/1024 of the canvas, centred, transparent margin."""
    tile = round(size * 824 / 1024)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # small mac sizes are downscaled from a large render: macOS shows them rarely and the bitmap set does not fit
    art = vector_icon(max(tile, 128)).resize((tile, tile), Image.LANCZOS) if tile < 128 else vector_icon(tile)
    canvas.alpha_composite(art, ((size - tile) // 2, (size - tile) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--preview", action="store_true", help="also write one PNG per size for inspection")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = [icon(s) for s in ICO_SIZES]
    frames[-1].save(args.out / "app.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1])
    frames[-1].save(args.out / "app.png")
    mac = [mac_icon(s) for s in ICNS_SIZES]
    mac[-1].save(args.out / "app.icns", format="ICNS", append_images=mac[:-1])
    if args.preview:
        for s, im in zip(ICO_SIZES, frames):
            im.save(args.out / f"preview_{s}.png")
        mac[-1].save(args.out / "preview_mac_1024.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

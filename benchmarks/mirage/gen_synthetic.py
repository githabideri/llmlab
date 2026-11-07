#!/usr/bin/env python3
"""Generate the synthetic corpus for the Mirage evaluation (deterministic, per-image seed).

Categories (GT = exact ground truth written to the manifest):
  shapes-simple   1-4 non-overlapping shapes (bbox, name, color, shape)
  shapes-crowded  6-12 overlapping shapes
  shapes-small    one small target in a plain field
  shapes-pairs    exactly 2 shapes, spatial-relation label (a relative to b)
  shapes-counts   1-6 shapes (counting task)
  shapes-faint    low-contrast shapes (adversarial)
  shapes-blank    empty canvases (negative controls)
  noise / gradient  texture-only negatives
  doc             rendered text documents, exact GT text
  ui-*            fake app windows / mobile screens / dialogs / HUD, exact GT

All geometry is in pixel coordinates of the rendered image.
Usage: python gen_synthetic.py   (writes images/ + images/synthetic-manifest.jsonl)
"""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

def relp(path):
    """repo-relative path for the manifest (the committed corpus uses
    paths relative to this directory; keep regenerating them that way)"""
    return os.path.relpath(path, os.path.dirname(OUT))

SHAPES = ["square", "circle", "triangle", "rectangle", "ellipse", "diamond"]
COLORS = {
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (40, 160, 60),
    "yellow": (230, 210, 30),
    "orange": (230, 120, 20),
    "purple": (140, 40, 180),
    "brown": (120, 70, 30),
    "pink": (230, 120, 170),
    "gray": (120, 120, 120),
}


def font(size=28):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_shape(d, shape, box, fill):
    x0, y0, x1, y1 = box
    if shape == "square" or shape == "rectangle":
        d.rectangle([x0, y0, x1, y1], fill=fill)
    elif shape == "circle" or shape == "ellipse":
        d.ellipse([x0, y0, x1, y1], fill=fill)
    elif shape == "triangle":
        d.polygon([(x0 + (x1 - x0) / 2, y0), (x0, y1), (x1, y1)], fill=fill)
    elif shape == "diamond":
        d.polygon([(x0 + (x1 - x0) / 2, y0), (x1, y0 + (y1 - y0) / 2),
                   (x0 + (x1 - x0) / 2, y1), (x0, y0 + (y1 - y0) / 2)], fill=fill)


def _intersects(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def make_shapes(kind, seed, n=None, size=(512, 512), overlap=False,
                bg=(245, 245, 245), faint=False, fixed_shapes=None):
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    palette = list(COLORS.keys())
    if fixed_shapes:
        specs = fixed_shapes
    else:
        specs = []
        for _ in range(n):
            s = rng.choice(SHAPES)
            c = rng.choice(palette)
            if kind == "shapes-small":
                side = rng.randint(16, 28)
            elif kind == "shapes-faint":
                side = rng.randint(60, 120)
            elif overlap:
                side = rng.randint(70, 160)
            else:
                side = rng.randint(90, 180)
            specs.append((s, c, side))
    placed = []
    objects = []
    for s, c, side in specs:
        dx, dy = side, int(side * (1.2 if s in ("rectangle", "ellipse") else 1.0))
        box = None
        for _tries in range(300):
            x0 = rng.randint(4, max(5, w - dx - 4))
            y0 = rng.randint(4, max(5, h - dy - 4))
            box = (x0, y0, x0 + dx, y0 + dy)
            if not overlap and all(not _intersects(box, p) for p in placed):
                break
        fill = COLORS[c]
        if faint:
            fill = tuple(int(bg[k] * 0.97 + fill[k] * 0.03) for k in range(3))
        draw_shape(d, s, box, fill)
        placed.append(box)
        objects.append({"name": f"{c} {s}", "bbox": [int(v) for v in box],
                        "color": c, "shape": s})
    return img, {"objects": objects}


def make_doc(seed, n_lines=6, font_size=30, lines=None, title="Invoices and notes"):
    rng = random.Random(seed)
    w, h = 900, 1200
    img = Image.new("RGB", (w, h), (252, 250, 246))
    d = ImageDraw.Draw(img)
    f_title = font(int(font_size * 1.7))
    f_body = font(font_size)
    y = 40
    d.text((50, y), title, font=f_title, fill=(20, 20, 20))
    y += int(font_size * 2.6)
    words_bank = ("order", "total", "due", "item", "price", "client", "amount",
                  "24", "137.50", "invoice", "number", "8842", "street", "Vienna",
                  "date", "2026", "net", "vat", "paid", "balance")
    text_lines = []
    for i in range(n_lines):
        if lines is not None and i < len(lines):
            line = lines[i]
        else:
            k = rng.randint(6, 14)
            line = " ".join(rng.choice(words_bank) for _ in range(k))
        text_lines.append(line)
        d.text((50, y), line, font=f_body, fill=(30, 30, 30))
        y += int(font_size * 1.5)
    return img, {"text": [title] + text_lines}


def make_ui(kind, seed):
    """Fake UI screens with exact ground truth of elements and text."""
    rng = random.Random(seed)
    w, h = (480, 960) if kind == "mobile" else (900, 600)
    img = Image.new("RGB", (w, h), (248, 248, 250))
    d = ImageDraw.Draw(img)
    f = font(18)
    elements = []
    texts = []

    if kind == "window":
        d.rectangle([0, 0, w, 44], fill=(60, 60, 66))
        for i, t in enumerate(["File", "Edit", "View", "Help"]):
            d.text((12 + i * 52, 12), t, font=f, fill=(240, 240, 240))
            elements.append({"name": "menu item", "text": t,
                             "bbox": [12 + i * 52, 8, 12 + i * 52 + 44, 36]})
            texts.append(t)
        for i, t in enumerate(["Save", "Open", "Export"]):
            x = 12 + i * 90
            d.rectangle([x, 60, x + 80, 96], fill=(90, 90, 100), outline=(40, 40, 46))
            d.text((x + 20, 68), t, font=f, fill=(240, 240, 240))
            elements.append({"name": "button", "text": t, "bbox": [x, 60, x + 80, 96]})
            texts.append(t)
        d.rectangle([12, 110, w - 12, h - 12], fill=(255, 255, 255), outline=(180, 180, 185))
        elements.append({"name": "content pane", "bbox": [12, 110, w - 12, h - 12]})
    elif kind == "mobile":
        d.rectangle([0, 0, w, 90], fill=(30, 30, 34))
        d.text((16, 30), "12:44", font=f, fill=(250, 250, 250))
        texts.append("12:44")
        elements.append({"name": "status bar clock", "text": "12:44", "bbox": [10, 20, 80, 60]})
        d.rectangle([0, 90, w, 160], fill=(210, 60, 60))
        d.text((20, 105), "Home screen", font=font(24), fill=(255, 255, 255))
        texts.append("Home screen")
        for i in range(8):
            x = 30 + (i % 4) * 110
            y = 200 + (i // 4) * 140
            d.rounded_rectangle([x, y, x + 96, y + 96], 18,
                                fill=rng.choice(list(COLORS.values())))
            elements.append({"name": "app icon", "bbox": [x, y, x + 96, y + 96]})
    elif kind == "dialog":
        d.rectangle([0, 0, w, h], fill=(200, 200, 205))
        d.rectangle([150, 150, w - 150, h - 150], fill=(250, 250, 250), outline=(120, 120, 126))
        d.text((180, 180), "Confirm action?", font=font(24), fill=(20, 20, 20))
        texts.append("Confirm action?")
        elements.append({"name": "dialog title", "text": "Confirm action?",
                         "bbox": [180, 170, 460, 210]})
        for i, t in enumerate(["OK", "Cancel"]):
            x = 220 + i * 160
            d.rectangle([x, 420, x + 120, 464], fill=(90, 90, 100))
            d.text((x + 30, 430), t, font=f, fill=(250, 250, 250))
            elements.append({"name": "button", "text": t, "bbox": [x, 420, x + 120, 464]})
            texts.append(t)
    elif kind == "terminal":
        d.rectangle([0, 0, w, h], fill=(16, 18, 22))
        lines = ["$ uptime", " 09:41:12 up 3 days, load average: 0.42",
                 "$ df -h /", "Filesystem  Size  Used  Avail",
                 "/dev/sda1  50G   32G    14G", "$ echo done", "done"]
        for i, line in enumerate(lines):
            d.text((20, 20 + i * 40), line, font=f, fill=(120, 220, 120))
            texts.append(line)
        elements.append({"name": "prompt line", "bbox": [20, 20, 300, 56]})
    elif kind == "chart":
        d.rectangle([0, 0, w, 40], fill=(70, 70, 78))
        d.text((12, 10), "Weekly throughput", font=f, fill=(240, 240, 240))
        texts.append("Weekly throughput")
        vals = [40, 70, 55, 90, 62, 78, 45]
        for i, v in enumerate(vals):
            x = 60 + i * 110
            d.rectangle([x, h - 80 - v * 4, x + 70, h - 80], fill=(60, 120, 220))
            elements.append({"name": "bar", "bbox": [x, h - 80 - v * 4, x + 70, h - 80]})
        for i, t in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            d.text((85 + i * 110, h - 66), t, font=f, fill=(60, 60, 60))
            texts.append(t)
    elif kind == "settings":
        d.text((40, 20), "Settings", font=font(28), fill=(20, 20, 20))
        texts.append("Settings")
        for i, (label, on) in enumerate([("Notifications", True), ("Dark mode", False),
                                         ("Auto update", True), ("Telemetry", False)]):
            y = 80 + i * 70
            d.text((40, y), label, font=f, fill=(30, 30, 30))
            texts.append(label)
            d.rounded_rectangle([w - 140, y, w - 70, y + 34], 17,
                                fill=(60, 160, 90) if on else (180, 180, 185))
            elements.append({"name": "toggle", "state": "on" if on else "off",
                             "bbox": [w - 140, y, w - 70, y + 34]})
    elif kind == "hud":
        d.rectangle([0, 0, w, h], fill=(40, 60, 40))
        d.rectangle([20, h - 60, 220, h - 30], fill=(150, 30, 30))
        elements.append({"name": "health bar", "bbox": [20, h - 60, 220, h - 30]})
        cx, cy = w // 2, h // 2
        d.line([cx - 14, cy, cx + 14, cy], fill=(240, 240, 240), width=2)
        d.line([cx, cy - 14, cx, cy + 14], fill=(240, 240, 240), width=2)
        elements.append({"name": "crosshair", "bbox": [cx - 14, cy - 14, cx + 14, cy + 14]})
        for i in range(5):
            x = 20 + i * 60
            d.rectangle([x, h - 120, x + 50, h - 70], outline=(200, 200, 200), width=2)
            elements.append({"name": "inventory slot", "bbox": [x, h - 120, x + 50, h - 70]})
        d.ellipse([w - 170, 10, w - 20, 160], outline=(220, 220, 180), width=2)
        elements.append({"name": "minimap", "bbox": [w - 170, 10, w - 20, 160]})
    return img, {"elements": elements, "text": texts}


PLAN = [
    ("shapes-simple", 1, dict(n=2)), ("shapes-simple", 2, dict(n=3)),
    ("shapes-simple", 3, dict(n=4)), ("shapes-simple", 4, dict(n=1)),
    ("shapes-simple", 5, dict(n=3)), ("shapes-simple", 6, dict(n=2)),
    ("shapes-simple", 7, dict(n=4)), ("shapes-simple", 8, dict(n=3)),
    ("shapes-crowded", 11, dict(n=8, overlap=True)),
    ("shapes-crowded", 12, dict(n=10, overlap=True)),
    ("shapes-crowded", 13, dict(n=12, overlap=True)),
    ("shapes-crowded", 14, dict(n=9, overlap=True)),
    ("shapes-crowded", 15, dict(n=8, overlap=True)),
    ("shapes-crowded", 16, dict(n=11, overlap=True)),
    ("shapes-crowded", 17, dict(n=10, overlap=True)),
    ("shapes-crowded", 18, dict(n=12, overlap=True)),
    ("shapes-small", 21, dict(n=1)), ("shapes-small", 22, dict(n=1)),
    ("shapes-small", 23, dict(n=1)),
    ("shapes-pairs", 31, dict(fixed_shapes=[("square", "red", 120), ("circle", "blue", 120)])),
    ("shapes-pairs", 32, dict(fixed_shapes=[("circle", "green", 100), ("triangle", "yellow", 100)])),
    ("shapes-pairs", 33, dict(fixed_shapes=[("rectangle", "orange", 140), ("diamond", "purple", 90)])),
    ("shapes-pairs", 34, dict(fixed_shapes=[("ellipse", "pink", 110), ("square", "brown", 110)])),
    ("shapes-counts", 41, dict(n=1)), ("shapes-counts", 42, dict(n=2)),
    ("shapes-counts", 43, dict(n=3)), ("shapes-counts", 44, dict(n=5)),
    ("shapes-counts", 45, dict(n=6)),
    ("shapes-faint", 51, dict(n=3, faint=True)),
    ("shapes-faint", 52, dict(n=2, faint=True)),
    ("shapes-blank", 61, dict(n=0, bg=(255, 255, 255))),
    ("shapes-blank", 62, dict(n=0, bg=(0, 0, 0))),
    ("shapes-blank", 63, dict(n=0, bg=(128, 128, 128))),
    ("shapes-blank", 64, dict(n=0, bg=(200, 200, 200))),
]
NOISE_SEEDS = [71, 72]
GRAD_SEEDS = [73]


def main():
    manifest = []
    for kind, seed, kw in PLAN:
        img, gt = make_shapes(kind, seed, **kw)
        path = os.path.join(OUT, f"syn-{kind}-{seed}.png")
        img.save(path)
        if kind == "shapes-pairs":
            (a, b) = gt["objects"]
            ax = (a["bbox"][0] + a["bbox"][2]) / 2
            ay = (a["bbox"][1] + a["bbox"][3]) / 2
            bx = (b["bbox"][0] + b["bbox"][2]) / 2
            by = (b["bbox"][1] + b["bbox"][3]) / 2
            if abs(ax - bx) > abs(ay - by):
                rel = "left of" if ax < bx else "right of"
            else:
                rel = "above" if ay < by else "below"
            gt["relation_a_to_b"] = rel
        manifest.append({"id": f"syn-{kind}-{seed}", "kind": "synthetic",
                         "subkind": kind, "path": relp(path), "gt": gt,
                         "size": list(img.size)})
    for seed in NOISE_SEEDS:
        rng = random.Random(seed)
        w = h = 512
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(h):
            for x in range(w):
                v = rng.randint(0, 255)
                px[x, y] = (v, v, v)
        path = os.path.join(OUT, f"syn-noise-{seed}.png")
        img.save(path)
        manifest.append({"id": f"syn-noise-{seed}", "kind": "synthetic",
                         "subkind": "noise", "path": relp(path),
                         "gt": {"objects": []}, "size": [w, h]})
    for seed in GRAD_SEEDS:
        w = h = 512
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(h):
            for x in range(w):
                px[x, y] = (int(255 * x / w), int(255 * y / h), 128)
        path = os.path.join(OUT, f"syn-gradient-{seed}.png")
        img.save(path)
        manifest.append({"id": f"syn-gradient-{seed}", "kind": "synthetic",
                         "subkind": "gradient", "path": relp(path),
                         "gt": {"objects": []}, "size": [w, h]})

    docs = []
    doc_cases = [
        (101, 6, 30, None, "Order confirmation"),
        (102, 3, 44, None, "INVOICE 8842"),
        (103, 8, 22, None, "Statement of account"),
        (104, 2, 60, None, "Receipt"),
        (105, 5, 26, ["total 137.50", "due date 2026-12-01", "thanks for your order"], "Order summary"),
        (106, 4, 34, ["item 3 x 12.00", "vat 20 percent", "net 36.00"], "Packing slip"),
        (107, 7, 20, None, "Shipping label"),
        (108, 3, 48, None, "Warranty card"),
        (109, 6, 24, None, "Bank statement"),
        (110, 5, 32, None, "Purchase order 5512"),
        (111, 4, 28, None, "Delivery note"),
        (112, 8, 18, None, "Fine print"),
    ]
    for seed, n, fs, lines, title in doc_cases:
        img, gt = make_doc(seed, n, fs, lines, title)
        path = os.path.join(OUT, f"syn-doc-{seed}.png")
        img.save(path)
        docs.append({"id": f"syn-doc-{seed}", "kind": "synthetic",
                     "subkind": "doc", "path": relp(path), "gt": gt,
                     "size": list(img.size)})

    ui_cases = [
        (201, "window"), (202, "mobile"), (203, "dialog"), (204, "terminal"),
        (205, "chart"), (206, "settings"), (207, "hud"),
        (208, "window"), (209, "mobile"), (210, "dialog"),
        (211, "terminal"), (212, "chart"), (213, "settings"), (214, "hud"),
    ]
    for seed, kind in ui_cases:
        img, gt = make_ui(kind, seed)
        path = os.path.join(OUT, f"syn-ui-{kind}-{seed}.png")
        img.save(path)
        docs.append({"id": f"syn-ui-{kind}-{seed}", "kind": "synthetic",
                     "subkind": f"ui-{kind}", "path": relp(path), "gt": gt,
                     "size": list(img.size)})

    allm = manifest + docs
    with open(os.path.join(OUT, "synthetic-manifest.jsonl"), "w") as fh:
        for m in allm:
            fh.write(json.dumps(m) + "\n")
    print(f"wrote {len(allm)} synthetic images to {OUT}")


if __name__ == "__main__":
    main()

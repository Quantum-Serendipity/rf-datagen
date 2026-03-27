"""SSTV test image generation."""

import numpy as np


def random_image(width, height):
    """Generate a random test image for SSTV encoding."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    pattern = np.random.randint(0, 5)

    if pattern == 0:
        # Color bars (SMPTE-style)
        colors = [
            (192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0),
            (192, 0, 192), (192, 0, 0), (0, 0, 192),
        ]
        bar_w = width // len(colors)
        for i, c in enumerate(colors):
            draw.rectangle([i * bar_w, 0, (i + 1) * bar_w, height], fill=c)

    elif pattern == 1:
        # Horizontal gradient
        for y in range(height):
            r = int(255 * y / height)
            g = int(255 * (1 - y / height))
            b = np.random.randint(0, 256)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    elif pattern == 2:
        # Random noise
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

    elif pattern == 3:
        # Concentric circles
        cx, cy = width // 2, height // 2
        max_r = min(cx, cy)
        for r in range(max_r, 0, -10):
            color = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    else:
        # Checkerboard
        sq = max(8, width // 16)
        for y in range(0, height, sq):
            for x in range(0, width, sq):
                if ((x // sq) + (y // sq)) % 2 == 0:
                    c = (255, 255, 255)
                else:
                    c = (0, 0, 0)
                draw.rectangle([x, y, x + sq, y + sq], fill=c)

    # Overlay callsign text
    callsigns = ["W1AW", "K3LR", "VE3XYZ", "DL1ABC", "JA1ABC"]
    call = np.random.choice(callsigns)
    try:
        draw.text((10, 10), call, fill=(255, 255, 0))
    except Exception:
        pass

    return img

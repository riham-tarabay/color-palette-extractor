"""
Color Palette Extractor
=======================
Extract dominant colors from any image using K-Means and Median-Cut clustering.
Outputs a beautiful hex palette with color names.

Author: Portfolio Project 02
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")


# ─── Color name mapping (basic CSS colors) ───────────────────────────────────

BASIC_COLORS = {
    (255, 0, 0): "Red", (0, 255, 0): "Lime", (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow", (0, 255, 255): "Cyan", (255, 0, 255): "Magenta",
    (0, 0, 0): "Black", (255, 255, 255): "White", (128, 128, 128): "Gray",
    (128, 0, 0): "Maroon", (0, 128, 0): "Green", (0, 0, 128): "Navy",
    (255, 165, 0): "Orange", (255, 192, 203): "Pink", (128, 0, 128): "Purple",
    (165, 42, 42): "Brown", (64, 224, 208): "Turquoise", (255, 215, 0): "Gold",
    (192, 192, 192): "Silver", (240, 248, 255): "AliceBlue",
}

def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    """Find the closest named color using Euclidean distance."""
    r, g, b = rgb
    min_dist = float("inf")
    name = "Unknown"
    for (cr, cg, cb), cname in BASIC_COLORS.items():
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            name = cname
    return name

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def luminance(rgb: tuple[int, int, int]) -> float:
    """Perceived luminance for text contrast."""
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ─── Extraction Methods ───────────────────────────────────────────────────────

def extract_kmeans(
    image_path: str,
    n_colors: int = 6,
    resize: int = 200,
    use_minibatch: bool = True,
) -> list[dict]:
    """
    Extract dominant colors using K-Means clustering.
    Pixels are treated as points in 3D RGB space.
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((resize, resize), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(float)

    model = MiniBatchKMeans(n_clusters=n_colors, random_state=42) if use_minibatch \
            else KMeans(n_clusters=n_colors, random_state=42, n_init="auto")
    labels = model.fit_predict(pixels)

    centers = model.cluster_centers_.astype(int)
    counts = Counter(labels)
    total = len(labels)

    palette = []
    for idx in sorted(counts, key=counts.get, reverse=True):
        rgb = tuple(np.clip(centers[idx], 0, 255))
        palette.append({
            "rgb": rgb,
            "hex": rgb_to_hex(rgb),
            "name": nearest_color_name(rgb),
            "percentage": round(counts[idx] / total * 100, 1),
        })
    return palette


def median_cut(pixels: np.ndarray, depth: int) -> list[np.ndarray]:
    """Recursive median-cut color quantization."""
    if depth == 0 or len(pixels) == 0:
        return [pixels]
    # Find channel with greatest range
    ranges = pixels.max(axis=0) - pixels.min(axis=0)
    channel = np.argmax(ranges)
    pixels = pixels[pixels[:, channel].argsort()]
    mid = len(pixels) // 2
    return median_cut(pixels[:mid], depth - 1) + median_cut(pixels[mid:], depth - 1)


def extract_median_cut(image_path: str, n_colors: int = 6, resize: int = 200) -> list[dict]:
    """Extract dominant colors using Median-Cut algorithm."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((resize, resize), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(float)

    depth = int(np.ceil(np.log2(n_colors)))
    buckets = median_cut(pixels, depth)

    palette = []
    total = len(pixels)
    for bucket in buckets[:n_colors]:
        if len(bucket) == 0:
            continue
        avg = bucket.mean(axis=0).astype(int)
        rgb = tuple(np.clip(avg, 0, 255))
        palette.append({
            "rgb": rgb,
            "hex": rgb_to_hex(rgb),
            "name": nearest_color_name(rgb),
            "percentage": round(len(bucket) / total * 100, 1),
        })
    palette.sort(key=lambda x: x["percentage"], reverse=True)
    return palette


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_palette(
    image_path: str,
    palette: list[dict],
    method: str = "K-Means",
    output_path: str = "palette_output.png",
) -> None:
    """Create a professional palette visualization."""
    img = Image.open(image_path).convert("RGB")
    n = len(palette)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F8F8")

    # Left: original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold", pad=12)
    axes[0].axis("off")

    # Right: palette swatches
    ax = axes[1]
    ax.set_facecolor("#F8F8F8")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")
    ax.set_title(f"Color Palette — {method}", fontsize=14, fontweight="bold", pad=12)

    for i, color in enumerate(palette):
        y = n - i - 1
        rgb_norm = tuple(c / 255 for c in color["rgb"])
        text_color = "white" if luminance(color["rgb"]) < 0.5 else "#1a1a1a"

        rect = mpatches.FancyBboxPatch(
            (0.02, y + 0.08), 0.96, 0.84,
            boxstyle="round,pad=0.01",
            facecolor=rgb_norm, edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(0.08, y + 0.5, color["hex"],
                va="center", ha="left", fontsize=13, fontweight="bold",
                color=text_color, fontfamily="monospace")
        ax.text(0.55, y + 0.65, color["name"],
                va="center", ha="left", fontsize=10, color=text_color)
        ax.text(0.55, y + 0.35, f"{color['percentage']}%",
                va="center", ha="left", fontsize=10, color=text_color, alpha=0.85)

    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Palette saved to: {output_path}")


def save_palette_json(palette: list[dict], output_path: str = "palette.json") -> None:
    clean = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in palette]
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"✅ JSON saved to: {output_path}")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract dominant colors from an image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-n", "--n-colors", type=int, default=6, help="Number of colors (default: 6)")
    parser.add_argument("-m", "--method", choices=["kmeans", "mediancut", "both"], default="both")
    parser.add_argument("-o", "--output", default="palette_output.png", help="Output image path")
    args = parser.parse_args()

    print(f"\n🎨 Extracting {args.n_colors} colors from: {args.image}")
    print("─" * 50)

    if args.method in ("kmeans", "both"):
        palette = extract_kmeans(args.image, args.n_colors)
        print("\n📊 K-Means Palette:")
        for c in palette:
            print(f"  {c['hex']}  {c['name']:15s}  {c['percentage']:5.1f}%")
        visualize_palette(args.image, palette, "K-Means", args.output)
        save_palette_json(palette, args.output.replace(".png", "_kmeans.json"))

    if args.method in ("mediancut", "both"):
        palette = extract_median_cut(args.image, args.n_colors)
        print("\n📊 Median-Cut Palette:")
        for c in palette:
            print(f"  {c['hex']}  {c['name']:15s}  {c['percentage']:5.1f}%")
        out = args.output.replace(".png", "_mediancut.png")
        visualize_palette(args.image, palette, "Median-Cut", out)
        save_palette_json(palette, args.output.replace(".png", "_mediancut.json"))


if __name__ == "__main__":
    main()

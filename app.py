# 🎨 Color Palette Extractor

> Extract dominant colors from any image using **K-Means** and **Median-Cut** clustering. Outputs a beautiful hex palette with percentages and color names.

## Results

| Algorithm | Speed | Quality | Best For |
|---|---|---|---|
| K-Means | Fast | High | General use |
| Median-Cut | Very Fast | Good | Real-time apps |

## Quick Start
```bash
pip install -r requirements.txt

# CLI
python extractor.py my_image.jpg -n 6 -m both

# Web app
streamlit run app.py
```

## Architecture
```
Image Input
    │
    ▼
Resize (200×200) for speed
    │
    ├─── K-Means: pixels as 3D RGB points → cluster centers = dominant colors
    └─── Median-Cut: recursively split color space by widest channel
    │
    ▼
Hex + Name + Percentage → Palette visualization
```

## What I Learned
- K-Means in a color context: RGB pixels as 3D points in color space
- Median-cut as a deterministic alternative to k-means for color quantization
- Perceived luminance for accessible text contrast on swatches
- Building interactive Streamlit demos with real-time image uploads

## Tech Stack
`Python` · `scikit-learn` · `Pillow` · `Matplotlib` · `Streamlit`

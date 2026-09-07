import base64
import numpy as np
from PIL import Image

def process_logo(src_path="assets/logo_raw.png", out_png="assets/logo.png", out_svg="assets/logo.svg"):
    img = Image.open(src_path).convert('RGBA')
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Calculate distance from pure white (255, 255, 255)
    diff = 255.0 - np.minimum(np.minimum(r, g), b)

    # Smooth alpha transition to prevent any white edge fringing
    alpha = np.clip((diff - 8.0) / 35.0, 0.0, 1.0) * 255.0

    # Color de-blending from white background
    alpha_norm = np.clip(alpha / 255.0, 0.001, 1.0)[:, :, np.newaxis]
    restored = (arr[:, :, :3] - 255.0 * (1.0 - alpha_norm)) / alpha_norm
    restored = np.clip(restored, 0.0, 255.0)

    out_arr = np.dstack([restored, alpha]).astype(np.uint8)
    out_img = Image.fromarray(out_arr, 'RGBA')

    # Tight crop around non-transparent logo pixels
    bbox = out_img.getbbox()
    cropped = out_img.crop(bbox)

    # Add gentle aesthetic padding
    w, h = cropped.size
    pad = int(max(w, h) * 0.06)
    padded_size = (w + pad * 2, h + pad * 2)
    final_img = Image.new('RGBA', padded_size, (0, 0, 0, 0))
    final_img.paste(cropped, (pad, pad))
    final_img.save(out_png, format='PNG')
    print(f"Saved transparent logo to {out_png} ({final_img.size})")

    # Generate animated SVG wrapper
    with open(out_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {padded_size[0]} {padded_size[1]}" width="100%" height="100%">
  <defs>
    <style>
      @keyframes burnPulse {{
        0%, 100% {{
          transform: scale(1) translateY(0);
          filter: drop-shadow(0 0 8px rgba(255, 87, 34, 0.35));
        }}
        50% {{
          transform: scale(1.025) translateY(-4px);
          filter: drop-shadow(0 0 24px rgba(255, 152, 0, 0.7));
        }}
      }}
      .torch-flame {{
        animation: burnPulse 3.2s ease-in-out infinite;
        transform-origin: {padded_size[0]//2}px {padded_size[1]//2}px;
      }}
    </style>
  </defs>
  <g class="torch-flame">
    <image href="data:image/png;base64,{b64}" width="{padded_size[0]}" height="{padded_size[1]}" />
  </g>
</svg>
"""
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved animated SVG to {out_svg}")

if __name__ == "__main__":
    import os
    src = r"C:\Users\Pikazu\.gemini\antigravity-ide\brain\15dbe98f-1b4a-4b39-b494-ddf75ed2eed1\.user_uploaded\media_1787889903478.png"
    if os.path.exists(src):
        process_logo(src)

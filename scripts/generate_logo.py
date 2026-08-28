import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

svg_code = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <!-- Color Gradients -->
    <linearGradient id="torchGradPrimary" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#E64A19" />
      <stop offset="35%" stop-color="#FF5722" />
      <stop offset="70%" stop-color="#FF9800" />
      <stop offset="100%" stop-color="#FFC107" />
    </linearGradient>
    <linearGradient id="torchGradCore" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#BF360C" />
      <stop offset="50%" stop-color="#FF7043" />
      <stop offset="100%" stop-color="#FFE082" />
    </linearGradient>
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFAB00" stop-opacity="0.8" />
      <stop offset="50%" stop-color="#FF5722" stop-opacity="0.4" />
      <stop offset="100%" stop-color="#DD2C00" stop-opacity="0.8" />
    </linearGradient>
    
    <!-- Smooth Native Animations -->
    <style>
      @keyframes flamePulse {
        0%, 100% { transform: scale(1) translateY(0); filter: drop-shadow(0 0 16px rgba(255, 87, 34, 0.45)); }
        50% { transform: scale(1.025) translateY(-5px); filter: drop-shadow(0 0 30px rgba(255, 152, 0, 0.75)); }
      }
      @keyframes coreGlow {
        0%, 100% { transform: scale(1) translateY(0); opacity: 0.92; }
        50% { transform: scale(0.98) translateY(-3px); opacity: 1.0; }
      }
      @keyframes nodePulse {
        0%, 100% { transform: scale(1); opacity: 0.85; }
        50% { transform: scale(1.2); opacity: 1.0; }
      }
      .flame-outer {
        animation: flamePulse 3.2s ease-in-out infinite;
        transform-origin: 256px 360px;
      }
      .flame-core {
        animation: coreGlow 3.2s ease-in-out infinite;
        transform-origin: 256px 360px;
      }
      .node-n { animation: nodePulse 2.8s ease-in-out infinite 0s; transform-origin: 256px 115px; }
      .node-e { animation: nodePulse 2.8s ease-in-out infinite 0.7s; transform-origin: 405px 256px; }
      .node-s { animation: nodePulse 2.8s ease-in-out infinite 1.4s; transform-origin: 256px 397px; }
      .node-w { animation: nodePulse 2.8s ease-in-out infinite 2.1s; transform-origin: 107px 256px; }
    </style>
  </defs>

  <!-- Modern Flat Canvas Background (Crisp Dark Charcoal) -->
  <rect width="512" height="512" rx="112" fill="#0E1117" />
  
  <!-- Subtle Outer Tensor Constellation Grid -->
  <circle cx="256" cy="256" r="172" fill="none" stroke="url(#ringGrad)" stroke-width="3" stroke-dasharray="8 6" opacity="0.45" />
  <polygon points="256,92 420,256 256,420 92,256" fill="none" stroke="#FF5722" stroke-width="2" stroke-dasharray="6 6" opacity="0.3" />

  <!-- Outer Geometric Flame (Flat modern vector contours) -->
  <g class="flame-outer">
    <path d="M 256,88
             C 290,155 358,205 358,292
             C 358,358 312,402 256,402
             C 200,402 154,358 154,292
             C 154,228 206,168 256,88 Z"
          fill="url(#torchGradPrimary)" />
  </g>

  <!-- Inner Dynamic Flame Core -->
  <g class="flame-core">
    <path d="M 256,165
             C 278,210 318,252 318,312
             C 318,354 290,384 256,384
             C 222,384 194,354 194,312
             C 194,262 226,224 256,165 Z"
          fill="url(#torchGradCore)" />

    <!-- Pure Radiance Center Spark -->
    <path d="M 256,232
             C 268,265 284,288 284,324
             C 284,345 272,364 256,364
             C 240,364 228,345 228,324
             C 228,292 242,265 256,232 Z"
          fill="#FFF9C4" opacity="0.95" />
  </g>

  <!-- Neural Tensor Graph Nodes & Links -->
  <g class="node-n">
    <circle cx="256" cy="115" r="14" fill="#FFD54F" stroke="#0E1117" stroke-width="3" />
    <circle cx="256" cy="115" r="6" fill="#FFFFFF" />
  </g>
  <g class="node-e">
    <circle cx="405" cy="256" r="14" fill="#FF5722" stroke="#0E1117" stroke-width="3" />
    <circle cx="405" cy="256" r="6" fill="#FFFFFF" />
  </g>
  <g class="node-s">
    <circle cx="256" cy="397" r="14" fill="#E64A19" stroke="#0E1117" stroke-width="3" />
    <circle cx="256" cy="397" r="6" fill="#FFFFFF" />
  </g>
  <g class="node-w">
    <circle cx="107" cy="256" r="14" fill="#FFA000" stroke="#0E1117" stroke-width="3" />
    <circle cx="107" cy="256" r="6" fill="#FFFFFF" />
  </g>
</svg>"""

with open("assets/logo.svg", "w", encoding="utf-8") as f:
    f.write(svg_code)
print("Saved assets/logo.svg")

# Render 512x512 high-resolution PNG using matplotlib
fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')
ax.set_xlim(0, 512)
ax.set_ylim(512, 0)
ax.axis('off')

# Outer dashed ring
ring = plt.Circle((256, 256), 172, fill=False, edgecolor='#FF7043', linewidth=2.5, linestyle='--', alpha=0.5)
ax.add_patch(ring)

# Tensor polygon
diamond = patches.Polygon([[256, 92], [420, 256], [256, 420], [92, 256]], closed=True, fill=False, edgecolor='#FF5722', linewidth=1.5, linestyle=':', alpha=0.4)
ax.add_patch(diamond)

# Flame outer path
verts_outer = [
    (256, 88),
    (290, 155), (358, 205), (358, 292),
    (358, 358), (312, 402), (256, 402),
    (200, 402), (154, 358), (154, 292),
    (154, 228), (206, 168), (256, 88),
]
codes_outer = [
    Path.MOVETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
]
flame_outer_patch = patches.PathPatch(Path(verts_outer, codes_outer), facecolor='#FF5722', edgecolor='#FF8A65', linewidth=2)
ax.add_patch(flame_outer_patch)

# Flame core path
verts_core = [
    (256, 165),
    (278, 210), (318, 252), (318, 312),
    (318, 354), (290, 384), (256, 384),
    (222, 384), (194, 354), (194, 312),
    (194, 262), (226, 224), (256, 165),
]
codes_core = [
    Path.MOVETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
]
flame_core_patch = patches.PathPatch(Path(verts_core, codes_core), facecolor='#FFA726', edgecolor='#FFCC80', linewidth=1.5)
ax.add_patch(flame_core_patch)

# Flame center spark
verts_spark = [
    (256, 232),
    (268, 265), (284, 288), (284, 324),
    (284, 345), (272, 364), (256, 364),
    (240, 364), (228, 345), (228, 324),
    (228, 292), (242, 265), (256, 232),
]
spark_patch = patches.PathPatch(Path(verts_spark, codes_core), facecolor='#FFF9C4', edgecolor='none')
ax.add_patch(spark_patch)

# Nodes
for (x, y, col) in [(256, 115, '#FFD54F'), (405, 256, '#FF5722'), (256, 397, '#E64A19'), (107, 256, '#FFA000')]:
    ax.add_patch(plt.Circle((x, y), 14, facecolor=col, edgecolor='#0E1117', linewidth=3))
    ax.add_patch(plt.Circle((x, y), 5, facecolor='#FFFFFF', edgecolor='none'))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('assets/logo.png', dpi=100, facecolor='#0E1117')
plt.close()
print("Saved assets/logo.png")

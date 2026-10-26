# DESIGN.md — Singapore Atmospheric & Haze Intelligence (Apple Daylight Edition)

## Aesthetic Direction & Apple HIG Principles
- **Visual World:** Apple Human Interface Guidelines (macOS / iPadOS Daylight Instrumentation & Apple Weather Precision).
- **Use Scene:** Effortless daytime and indoor readability on high-DPI Retina displays. High-contrast ink-on-paper clarity (`#1D1D1F` on `#FFFFFF` and `#F5F5F7`), subtle translucent frosted headers (`backdrop-filter: saturate(180%) blur(20px)`), and fluid spring-physics morphing across SVG paths, spatial contours, and numerical readouts.

## Color & Surface Tokens (Daylight Palette)
- **Canvas & Surfaces:**
  - `--bg-canvas`: `#F5F5F7` (Apple warm daylight alabaster background)
  - `--bg-surface`: `#FFFFFF` (Pure crisp card surface)
  - `--bg-elevated`: `#F0F2F5` (Segmented control & subtle well surface)
  - `--bg-inset`: `#F8F9FB` (Map & chart well canvas)
  - `--border-subtle`: `#E5E5EA` (1px Apple system separator hairline)
  - `--border-strong`: `#D1D1D6` (Interactive control border)
- **Typography Hierarchy:**
  - `--text-primary`: `#1D1D1F` (Apple primary obsidian ink, contrast > 16:1)
  - `--text-secondary`: `#515154` (Apple secondary slate label, contrast > 7.5:1)
  - `--text-muted`: `#6E6E73` (Axis ticks, coordinates, unit annotations, contrast > 4.8:1)
- **NEA Air Quality Semantic Scale (Tuned for White Daylight Surfaces):**
  - `--nea-good`: `#059669` (0–50 Good: Deep Emerald, contrast > 4.5:1 on white/tint)
  - `--nea-moderate`: `#0284C7` (51–100 Moderate: Apple System Cerulean)
  - `--nea-unhealthy`: `#D97706` (101–200 Unhealthy: Rich Amber Ochre)
  - `--nea-very-unhealthy`: `#EA580C` (201–300 Very Unhealthy: Deep Tangelo)
  - `--nea-hazardous`: `#DC2626` (>300 Hazardous: Crimson Alert)
- **Accent & Forecast Horizon:**
  - `--accent-live`: `#0071E3` (Apple System Blue)
  - `--accent-forecast`: `#7C3AED` (Deep Violet Predictive Trace)
  - `--accent-pm25`: `#0D9488` (Teal Particulate Trace)

## Motion & Morphing Architecture
- **Spring Physics Curve:** `cubic-bezier(0.22, 1, 0.36, 1)` across all layout, fill, stroke, and transform transitions.
- **SVG Path & Radial Plume Morphing:** Smoothly interpolate (`requestAnimationFrame`) polyline coordinates, wind-rose radar vertices, and numerical counters when scrubbing across the 72-hour horizon or perturbing wind vectors.

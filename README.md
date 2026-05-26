# 🎨 Gesture Drawing Board
Real-time hand gesture drawing using Python + OpenCV + MediaPipe

## ⚡ Quick Setup

### Step 1 — Install Python 3.11
Download from: https://www.python.org/downloads/release/python-3119/
✅ CHECK "Add Python to PATH" during install

### Step 2 — Open terminal in project folder
```
cd gesture-drawing-board
```

### Step 3 — Create virtual environment
```
py -3.11 -m venv venv
venv\Scripts\activate
```

### Step 4 — Install dependencies
```
pip install -r requirements.txt
```

### Step 5 — Run!
```
python main.py
```

---

## 🖐 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝ Index finger only | ✏ Draw |
| ✌ Index + Middle fingers | 🖱 Move / Select |
| 👊 Fist (all fingers down) | 🧼 Erase |
| 🖐 All 5 fingers open | 🗑 Clear canvas |

## ⌨ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| S | 💾 Save drawing as PNG |
| C | 🗑 Clear canvas |
| B | 🖌 Cycle brush (pen→neon→spray→glow) |
| + / - | 🔼🔽 Brush size |
| Q / ESC | 🚪 Quit |

## 🎨 Brush Types
- **Pen** — Clean solid strokes
- **Neon** — Glowing layered strokes with white core
- **Spray** — Airbrush scatter effect
- **Glow** — Multi-layer radial bloom

## 📁 Project Files
```
gesture-drawing-board/
├── main.py              ← Entry point — run this
├── hand_tracker.py      ← MediaPipe hand detection
├── drawing_engine.py    ← Canvas + brush effects
├── ui_overlay.py        ← HUD, color palette, slider
├── effects.py           ← Particles, flash, bloom
├── gesture_recognizer.py← Gesture → mode logic
├── utils.py             ← Smoothing + helpers
├── requirements.txt     ← pip dependencies
└── saves/               ← Your saved drawings go here
```

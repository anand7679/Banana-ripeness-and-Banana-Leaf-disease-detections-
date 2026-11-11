import html
import importlib.util
import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src_Banana_ripeness"))

ripeness_module_spec = importlib.util.spec_from_file_location(
    "banana_ripeness_predict",
    PROJECT_DIR / "src_Banana_ripeness" / "predict.py",
)
if ripeness_module_spec is None or ripeness_module_spec.loader is None:
    raise ImportError("Could not load the banana ripeness prediction module.")
ripeness_module = importlib.util.module_from_spec(ripeness_module_spec)
ripeness_module_spec.loader.exec_module(ripeness_module)
predict_ripeness = ripeness_module.predict
from banana_leaf_disease.predict import predict as predict_leaf_disease

RIPENESS_CHECKPOINT = PROJECT_DIR / "checkpoints" / "best_vit.pth"
LEAF_CHECKPOINT = PROJECT_DIR / "checkpoints" / "banana_leaf_disease_vit.pth"

st.set_page_config(
    page_title="Banana Leaf Health and Ripeness Tracking",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #08170F;
  --ink-2: #0F2A1C;
  --leaf: #8DC26F;
  --banana: #F2C230;
  --rust: #D9643A;
  --paper: #F4EBD6;
  --mute: rgba(244, 235, 214, 0.68);
  --line: rgba(244, 235, 214, 0.14);
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

html, body { background: #08170F; }

/* Lab-style background: measurement grid, topographic contours, gold scan line */
.stApp {
  background-color: #08170F;
  background-image:
    linear-gradient(rgba(242, 194, 48, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(242, 194, 48, 0.045) 1px, transparent 1px),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 720 720'%3E%3Cg fill='none' stroke='%23F2C230' stroke-opacity='0.08' stroke-width='1.2'%3E%3Cpath d='M60 360 C120 220 260 180 360 240 S600 300 660 380 S560 560 400 560 S120 520 60 360Z'/%3E%3Cpath d='M120 360 C170 270 280 240 360 285 S540 330 590 390 S520 500 400 500 S160 470 120 360Z'/%3E%3Cpath d='M190 360 C220 310 300 295 360 325 S480 360 520 395 S460 440 390 440 S220 420 190 360Z'/%3E%3Cpath d='M270 365 C290 340 330 335 360 350 S420 375 440 390 S400 400 360 398 S280 390 270 365Z'/%3E%3C/g%3E%3C/svg%3E");
  background-size: 56px 56px, 56px 56px, 720px 720px;
  color: var(--paper);
  font-family: 'Instrument Sans', system-ui, sans-serif;
  overflow-x: hidden;
}
.stApp::before {
  content: "";
  position: fixed;
  inset: -30%;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(closest-side at 50% 0%, rgba(60, 170, 130, 0.12), transparent 70%),
    radial-gradient(closest-side at 14% 18%, rgba(60, 170, 130, 0.10), transparent 70%),
    radial-gradient(closest-side at 86% 38%, rgba(60, 170, 130, 0.08), transparent 70%);
}
.stApp::after {
  content: "";
  position: fixed;
  left: 0;
  right: 0;
  top: 0;
  height: 2px;
  z-index: 0;
  pointer-events: none;
  background: linear-gradient(90deg, transparent, rgba(242, 194, 48, 0.55), transparent);
  box-shadow: 0 0 24px 4px rgba(242, 194, 48, 0.18);
  animation: scan-beam 9s linear infinite;
}
@keyframes scan-beam {
  0% { transform: translateY(0); opacity: 0; }
  8%, 92% { opacity: 0.8; }
  100% { transform: translateY(100vh); opacity: 0; }
}

[data-testid="stHeader"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  max-width: 100% !important;
  padding: 0 clamp(1.2rem, 4vw, 4.5rem) 4rem !important;
  position: relative;
  z-index: 1;
}

/* Hero */
.hero {
  position: relative;
  min-height: 66vh;
  display: flex;
  align-items: center;
  padding: 8vh 0 5vh;
  overflow: hidden;
}
.hero-inner { position: relative; z-index: 1; max-width: 1100px; }
.hero h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 800;
  font-size: clamp(2.5rem, 5.6vw, 5.4rem);
  line-height: 1.0;
  letter-spacing: -0.025em;
  margin: 0 0 1.4rem;
  max-width: 18ch;
  color: var(--paper);
  animation: rise 0.9s var(--ease) both;
}
.hero .lede {
  font-size: clamp(1.02rem, 1.4vw, 1.25rem);
  line-height: 1.65;
  color: rgba(244, 235, 214, 0.68);
  max-width: 56ch;
  margin: 0;
  animation: rise 0.9s 0.15s var(--ease) both;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-top: 1.8rem;
  animation: rise 0.9s 0.25s var(--ease) both;
}
.chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.45rem 1rem;
  font-size: 0.92rem;
  color: var(--paper);
  background: rgba(244, 235, 214, 0.04);
}
.chip.accent { border-color: var(--leaf); }

.spores { position: absolute; inset: 0; pointer-events: none; z-index: 0; }
.spore {
  position: absolute;
  bottom: -20px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(242, 194, 48, 0.95), rgba(242, 194, 48, 0) 70%);
  opacity: 0;
  animation-name: float;
  animation-timing-function: linear;
  animation-iteration-count: infinite;
}
.spore:nth-child(3n) {
  background: radial-gradient(circle, rgba(141, 194, 111, 0.95), rgba(141, 194, 111, 0) 70%);
}
@keyframes float {
  0%   { transform: translate3d(0, 0, 0); opacity: 0; }
  12%  { opacity: 0.75; }
  88%  { opacity: 0.5; }
  100% { transform: translate3d(28px, -110vh, 0); opacity: 0; }
}

.rule {
  height: 1px;
  background: linear-gradient(90deg, var(--banana), transparent 75%);
  margin: 0 0 2.6rem;
  opacity: 0.6;
  transform-origin: left;
  animation: grow-x 1.2s 0.3s var(--ease) both;
}
@keyframes rise {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: none; }
}
@keyframes grow-x {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

/* Panels */
.panel-title {
  font-family: 'Fraunces', serif;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 0.3rem;
  color: var(--paper);
}
.panel-sub { color: rgba(244, 235, 214, 0.68); margin: 0 0 1.2rem; line-height: 1.5; }

.track-card {
  margin-top: 0.9rem;
  padding: 1rem 1.2rem;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: rgba(244, 235, 214, 0.04);
  color: rgba(244, 235, 214, 0.68);
  line-height: 1.55;
  font-size: 0.97rem;
}
.track-card strong { color: var(--paper); font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.1rem; }

/* Radio as large pills */
[data-testid="stRadio"] label p {
  font-size: 1.02rem;
  color: var(--paper);
  font-weight: 500;
}
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.7rem; }
[data-testid="stRadio"] [role="radiogroup"] > label {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.6rem 1.3rem;
  background: rgba(244, 235, 214, 0.04);
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
  cursor: pointer;
}
[data-testid="stRadio"] [role="radiogroup"] > label:hover {
  border-color: rgba(242, 194, 48, 0.6);
  transform: translateY(-1px);
}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
  border-color: var(--banana);
  background: rgba(242, 194, 48, 0.14);
}

/* Text and date inputs */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
  background: rgba(244, 235, 214, 0.05);
  color: var(--paper);
  border-color: var(--line);
  border-radius: 12px;
}

/* Upload zone */
[data-testid="stFileUploader"] section {
  background: rgba(244, 235, 214, 0.04);
  border: 1.5px dashed rgba(242, 194, 48, 0.5);
  border-radius: 22px;
  min-height: 150px;
  transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--banana);
  background: rgba(242, 194, 48, 0.08);
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: rgba(244, 235, 214, 0.68) !important; }

[data-testid="stImage"] img {
  border-radius: 18px;
  border: 1px solid var(--line);
}

.tips {
  margin: 1rem 0 0;
  padding: 1.1rem 1.3rem;
  list-style: none;
  border-radius: 16px;
  background: rgba(244, 235, 214, 0.04);
  border: 1px solid var(--line);
  display: grid;
  gap: 0.55rem;
  color: rgba(244, 235, 214, 0.68);
  line-height: 1.5;
  font-size: 0.97rem;
}
.tips li::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--leaf);
  margin-right: 0.7rem;
  vertical-align: middle;
}

/* Empty state */
.empty {
  border: 1px dashed var(--line);
  border-radius: 24px;
  padding: 3rem 2rem;
  min-height: 420px;
  display: grid;
  place-items: center;
  text-align: center;
  color: rgba(244, 235, 214, 0.68);
  line-height: 1.65;
  animation: breathe 3.6s ease-in-out infinite;
}
.empty strong {
  display: block;
  font-family: 'Fraunces', serif;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--paper);
  margin-bottom: 0.6rem;
}
@keyframes breathe {
  0%, 100% { border-color: rgba(244, 235, 214, 0.12); }
  50%      { border-color: rgba(242, 194, 48, 0.45); }
}

/* Result readout */
.readout {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2rem;
  align-items: center;
  animation: rise 0.7s var(--ease) both;
}
.ring {
  width: clamp(140px, 15vw, 190px);
  height: clamp(140px, 15vw, 190px);
  border-radius: 50%;
  display: grid;
  place-items: center;
  position: relative;
  background: conic-gradient(var(--tone) calc(var(--p) * 1%), rgba(244, 235, 214, 0.1) 0);
  animation: sweep 1.2s var(--ease) both;
}
.ring::after {
  content: "";
  position: absolute;
  inset: 14px;
  border-radius: 50%;
  background: var(--ink-2);
}
.ring span {
  position: relative;
  z-index: 1;
  font-family: 'Fraunces', serif;
  font-size: clamp(1.9rem, 2.6vw, 2.6rem);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
@keyframes sweep {
  from { background: conic-gradient(var(--tone) 0%, rgba(244, 235, 214, 0.1) 0); }
}
.verdict-kicker { color: rgba(244, 235, 214, 0.68); font-size: 0.97rem; margin: 0; }
.verdict h2 {
  font-family: 'Fraunces', serif;
  font-size: clamp(2.2rem, 3.8vw, 3.6rem);
  font-weight: 600;
  line-height: 1.02;
  margin: 0.3rem 0 0.6rem;
  color: var(--paper);
}
.verdict p { margin: 0; color: rgba(244, 235, 214, 0.68); line-height: 1.55; }

.advice {
  margin-top: 1.8rem;
  padding: 1.1rem 1.3rem;
  border-left: 3px solid var(--tone);
  background: rgba(244, 235, 214, 0.05);
  border-radius: 0 16px 16px 0;
  line-height: 1.65;
  font-size: 1.02rem;
}

.bars { margin-top: 1.8rem; display: grid; gap: 1rem; }
.bar {
  display: grid;
  grid-template-columns: 9rem 1fr 4.2rem;
  gap: 1rem;
  align-items: center;
  font-size: 1rem;
  color: rgba(244, 235, 214, 0.68);
}
.bar.is-top { color: var(--paper); font-weight: 600; }
.bar-track {
  height: 12px;
  border-radius: 99px;
  background: rgba(244, 235, 214, 0.08);
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, rgba(242, 194, 48, 0.4), var(--banana));
  transform-origin: left;
  animation: grow 1s var(--ease) both;
}
.bar:not(.is-top) .bar-fill { background: rgba(244, 235, 214, 0.35); }
.bar-val { text-align: right; font-variant-numeric: tabular-nums; }
@keyframes grow {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

/* Leaf area affected */
.sev {
  margin-top: 1.4rem;
  padding: 1rem 1.2rem;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: rgba(244, 235, 214, 0.04);
}
.sev-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.6rem;
  color: rgba(244, 235, 214, 0.68);
}
.sev-head b {
  font-family: 'Fraunces', serif;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--rust);
}
.sev-track {
  height: 12px;
  border-radius: 99px;
  background: rgba(244, 235, 214, 0.08);
  overflow: hidden;
}
.sev-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, rgba(217, 100, 58, 0.5), var(--rust));
  transform-origin: left;
  animation: grow 1s var(--ease) both;
}
.sev p {
  margin: 0.7rem 0 0;
  color: rgba(244, 235, 214, 0.68);
  font-size: 0.88rem;
  line-height: 1.5;
}

.info-note { color: rgba(244, 235, 214, 0.68); font-size: 0.95rem; line-height: 1.6; margin-top: 1rem; }

/* Tracking log and timeline */
.section-title {
  font-family: 'Fraunces', serif;
  font-size: 1.7rem;
  font-weight: 600;
  margin: 4rem 0 0.3rem;
  color: var(--paper);
}
.section-sub { color: rgba(244, 235, 214, 0.68); margin: 0 0 1.4rem; line-height: 1.5; }
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.4rem;
}
.stat {
  padding: 1.1rem 1.2rem;
  border-radius: 18px;
  background: rgba(244, 235, 214, 0.04);
  border: 1px solid var(--line);
}
.stat b {
  display: block;
  font-family: 'Fraunces', serif;
  font-size: 2rem;
  font-weight: 600;
  color: var(--banana);
  font-variant-numeric: tabular-nums;
}
.stat span { color: rgba(244, 235, 214, 0.68); font-size: 0.92rem; }

.tl-result {
  margin-top: 1rem;
  padding: 1rem 1.2rem;
  border-radius: 16px;
  border-left: 3px solid var(--leaf);
  background: rgba(244, 235, 214, 0.05);
  line-height: 1.6;
}
.tl-result b {
  color: var(--leaf);
  font-family: 'Fraunces', serif;
  font-size: 1.2rem;
  font-weight: 600;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
}

.foot {
  margin-top: 4rem;
  padding-top: 1.6rem;
  border-top: 1px solid var(--line);
  color: rgba(244, 235, 214, 0.68);
  font-size: 0.9rem;
  line-height: 1.6;
  max-width: 80ch;
}

@media (max-width: 720px) {
  .hero { min-height: auto; padding: 6vh 0 4vh; }
  .readout { grid-template-columns: 1fr; justify-items: start; gap: 1.4rem; }
  .bar { grid-template-columns: 6.5rem 1fr 3.6rem; gap: 0.7rem; }
  .empty { min-height: 260px; }
  .stats { grid-template-columns: repeat(2, 1fr); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .spore { display: none; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def compact(markup: str) -> str:
    """Join HTML onto one line so Streamlit's markdown never reads it as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def pretty(name) -> str:
    return str(name).replace("_", " ").title()


def normalize_label(name) -> str:
    return str(name).lower().replace("_", "").replace(" ", "").replace("-", "")


TRACKS = {
    "Banana leaf": {
        "title": "Leaf Health",
        "desc": "Detect Cordana or Sigatoka leaf spot, or confirm a healthy leaf.",
    },
    "Banana fruit": {
        "title": "Ripeness Tracking",
        "desc": "Check whether the fruit is best to eat right now.",
    },
}

SUBJECT_PROMPTS = [
    "a photo of a banana leaf",
    "a photo of a banana fruit",
    "a photo of something that is not a banana",
]

# Photo quality thresholds: starting values, tune them with your own photos.
DARK_THRESHOLD = 60
BLUR_THRESHOLD = 60

@st.cache_resource(show_spinner="Loading the subject check...")
def load_subject_model():
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)


@st.cache_data(show_spinner=False)
def detect_subject(image_bytes: bytes) -> str:
    """Return 'leaf', 'fruit', 'other', or 'unclear' using CLIP zero-shot classification."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, processor = load_subject_model()
    inputs = processor(text=SUBJECT_PROMPTS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    best = int(probs.argmax())
    score = float(probs[best])
    if score < 0.5:
        return "unclear"
    return ["leaf", "fruit", "other"][best]


def photo_issues(image: Image.Image) -> list:
    """Check brightness and sharpness before running the model."""
    gray = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)
    issues = []
    if gray.mean() < DARK_THRESHOLD:
        issues.append("The photo is quite dark. Use brighter, even light for a more reliable result.")
    lap = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    if lap.var() < BLUR_THRESHOLD:
        issues.append("The photo looks blurry. Hold the camera steady and focus on the leaf or fruit.")
    return issues


@st.cache_data(show_spinner=False)
def run_model(image_bytes: bytes, mode: str) -> dict:
    """Run the selected model. Cached, so the same photo is not re-processed on every rerun."""
    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.jpg"
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(image_path)
        if mode == "Banana fruit":
            return predict_ripeness(str(image_path), checkpoint=str(RIPENESS_CHECKPOINT))
        return predict_leaf_disease(str(image_path), checkpoint_path=str(LEAF_CHECKPOINT))


@st.cache_data(show_spinner=False)
def leaf_severity(image_bytes: bytes):
    """
    Rough estimate of the share of leaf area covered by dark lesions, in percent.
    Colour-based, so it is approximate. Dark shadows can be counted as lesions.
    Returns None if the leaf area is too small to measure.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * 360 / 255
    sat = hsv[..., 1] / 255
    val = hsv[..., 2] / 255

    healthy_leaf = (sat > 0.2) & (val > 0.15) & (hue >= 35) & (hue <= 170)
    lesion = (val < 0.35) & (hue >= 5) & (hue <= 50)

    leaf_total = int((healthy_leaf | lesion).sum())
    if leaf_total < 2000:
        return None
    return float(lesion.sum() / leaf_total * 100)


SPORES = [
    (6, 10, 14, 0), (14, 6, 18, 3), (22, 14, 16, 7), (31, 8, 20, 1),
    (39, 12, 15, 5), (47, 7, 22, 9), (56, 11, 17, 2), (63, 9, 19, 6),
    (71, 13, 16, 4), (79, 6, 21, 8), (86, 10, 18, 11), (93, 8, 15, 12),
]
SPORE_HTML = "".join(
    f'<span class="spore" style="left:{left}%;width:{size}px;height:{size}px;'
    f'animation-duration:{dur}s;animation-delay:-{delay}s;"></span>'
    for left, size, dur, delay in SPORES
)

LEAF_GUIDANCE = {
    "Healthy": (
        "No strong disease pattern was detected. Keep monitoring the plant "
        "and keep leaves well ventilated."
    ),
    "Cordana": (
        "Possible Cordana leaf spot. Remove severely affected leaves, avoid "
        "overhead watering, and consult a local expert about approved treatment."
    ),
    "Sigatoka": (
        "Possible Sigatoka leaf spot. Inspect nearby leaves, remove badly "
        "affected material, improve airflow, and consult a local expert about "
        "approved fungicide options."
    ),
}


def render_hero():
    st.markdown(
        compact(
            f"""
            <section class="hero">
              <div class="spores" aria-hidden="true">{SPORE_HTML}</div>
              <div class="hero-inner">
                <h1>Edge-Optimized Vision Transformers for Integrated Banana Leaf Health and Banana Ripeness Tracking</h1>
                <p class="lede">Check a banana leaf for disease or a banana fruit for ripeness.
                A compact vision transformer, built for edge devices, gives the result,
                how confident it is, and logs every scan.</p>
                <div class="chips">
                  <span class="chip accent">Leaf Health</span>
                  <span class="chip">Ripeness Tracking</span>
                  <span class="chip">Edge-ready</span>
                </div>
              </div>
            </section>
            <div class="rule"></div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        compact(
            """
            <div class="empty">
              <div>
                <strong>Your result will appear here</strong>
                Upload a photo on the left and the class, confidence and
                full breakdown will show up in this space.
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_result(result, mode_name):
    """Render the result and return the headline and confidence."""
    label_raw = str(result["label"])
    label = pretty(label_raw)
    confidence = float(result["confidence"]) * 100

    raw_probabilities = result.get("probabilities")
    if raw_probabilities is None:
        raw_probabilities = {label_raw: float(result["confidence"])}
        st.markdown(
            compact(
                """
                <p class="info-note">This prediction did not include scores for every
                class, so the breakdown shows only the top prediction.</p>
                """
            ),
            unsafe_allow_html=True,
        )
    probabilities = {pretty(name): float(value) * 100 for name, value in raw_probabilities.items()}

    if confidence >= 80:
        tone = "var(--leaf)"
        tone_note = "The model is highly confident in this result."
    elif confidence >= 60:
        tone = "var(--banana)"
        tone_note = (
            "Moderate confidence. Use a clear, well-lit image and consider "
            "checking another sample."
        )
    else:
        tone = "var(--rust)"
        tone_note = (
            "The model is uncertain. Treat this as a screening result, not a "
            "final agricultural diagnosis."
        )

    if mode_name == "Banana leaf":
        kicker = "Leaf screening result"
        headline = label
        advice = LEAF_GUIDANCE.get(label, "Inspect the plant and consult an expert.")
    else:
        kicker = "Ripeness result"
        if normalize_label(label_raw) == "besttoeat":
            headline = "Best to eat"
            advice = "The model estimates that this fruit is in the best-to-eat class."
        else:
            headline = "Not yet best to eat"
            advice = "The model estimates that this fruit is not in the best-to-eat class."

    bars = "".join(
        compact(
            f"""
            <div class="bar{' is-top' if name == label else ''}">
              <span>{html.escape(name)}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{max(0.0, min(100.0, value)):.1f}%"></div></div>
              <span class="bar-val">{value:.1f}%</span>
            </div>
            """
        )
        for name, value in sorted(probabilities.items(), key=lambda item: -item[1])
    )

    st.markdown(
        compact(
            f"""
            <div class="readout" style="--tone:{tone}; --p:{confidence:.1f};">
              <div class="ring"><span>{confidence:.0f}%</span></div>
              <div class="verdict">
                <p class="verdict-kicker">{kicker}</p>
                <h2>{html.escape(headline)}</h2>
                <p>{tone_note}</p>
              </div>
            </div>
            <div class="advice" style="--tone:{tone};">{html.escape(advice)}</div>
            <div class="bars">{bars}</div>
            """
        ),
        unsafe_allow_html=True,
    )
    return headline, confidence


def render_severity(percent):
    if percent is None:
        st.markdown(
            compact(
                """
                <div class="sev">
                  <p>The leaf area could not be measured in this photo. Use a photo
                  where the whole leaf is visible.</p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        return
    width = max(0.0, min(100.0, percent))
    st.markdown(
        compact(
            f"""
            <div class="sev">
              <div class="sev-head">
                <span>Estimated leaf area affected</span>
                <b>{percent:.1f}%</b>
              </div>
              <div class="sev-track"><div class="sev-fill" style="width:{width:.1f}%"></div></div>
              <p>Rough estimate from lesion colour. Treat it as a guide until a
              segmentation model is trained on labelled leaf masks.</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_tracking_log():
    st.markdown(
        compact(
            """
            <h2 class="section-title">Tracking log</h2>
            <p class="section-sub">Every scan in this session, so you can follow leaf health and ripeness over time.</p>
            """
        ),
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.markdown(
            compact(
                """
                <p class="info-note">No scans yet. Your results will be logged here.</p>
                """
            ),
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(history)
    leaf_checks = int((df["Track"] == "Leaf Health").sum())
    fruit_checks = len(df) - leaf_checks
    avg_conf = df["Confidence"].mean()

    st.markdown(
        compact(
            f"""
            <div class="stats">
              <div class="stat"><b>{len(df)}</b><span>Total scans</span></div>
              <div class="stat"><b>{leaf_checks}</b><span>Leaf checks</span></div>
              <div class="stat"><b>{fruit_checks}</b><span>Ripeness checks</span></div>
              <div class="stat"><b>{avg_conf:.1f}%</b><span>Average confidence</span></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("**Confidence across scans**")
    st.line_chart(df["Confidence"], height=220)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col_csv, col_clear, _ = st.columns([1.3, 1, 3])
    with col_csv:
        st.download_button(
            "Download log as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="banana_tracking_log.csv",
            mime="text/csv",
        )
    with col_clear:
        if st.button("Clear tracking log"):
            st.session_state.history = []
            st.session_state.last_logged = None
            st.session_state.image_uploader_version += 1
            st.rerun()


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_logged" not in st.session_state:
    st.session_state.last_logged = None
if "image_uploader_version" not in st.session_state:
    st.session_state.image_uploader_version = 0
render_hero()

left, right = st.columns([1, 1.15], gap="large")

image = None
image_bytes = None
detected = "unclear"

with left:
    st.markdown(
        compact(
            """
            <h2 class="panel-title">Choose a check</h2>
            <p class="panel-sub">Pick the track, then upload one clear photo.</p>
            """
        ),
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Which check do you want to run?",
        list(TRACKS.keys()),
        format_func=lambda key: TRACKS[key]["title"],
        horizontal=True,
    )
    st.markdown(
        compact(
            f"""
            <div class="track-card"><strong>{TRACKS[mode]["title"]}</strong><br>{TRACKS[mode]["desc"]}</div>
            """
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"image_uploader_{st.session_state.image_uploader_version}",
    )

    st.markdown(
        compact(
            """
            <ul class="tips">
              <li>Keep one banana or one leaf in the frame.</li>
              <li>Use daylight or bright, even lighting.</li>
              <li>Avoid heavy shadows and blurry photos.</li>
            </ul>
            """
        ),
        unsafe_allow_html=True,
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        detected = detect_subject(image_bytes)

with right:
    st.markdown(
        compact(
            """
            <h2 class="panel-title">Result</h2>
            <p class="panel-sub">The class, how confident the model is, and every score.</p>
            """
        ),
        unsafe_allow_html=True,
    )
    if image is None:
        render_empty_state()
    elif detected == "other":
        st.warning(
            "This photo does not look like a banana leaf or fruit. "
            "Upload a clear photo of one banana leaf or one banana fruit."
        )
    elif mode == "Banana fruit" and detected == "leaf":
        st.warning(
            "This photo looks like a banana leaf, but Ripeness Tracking is selected. "
            "Choose Leaf Health above to get a valid result."
        )
    elif mode == "Banana leaf" and detected == "fruit":
        st.warning(
            "This photo looks like a banana fruit, but Leaf Health is selected. "
            "Choose Ripeness Tracking above to get a valid result."
        )
    else:
        issues = photo_issues(image)
        for issue in issues:
            st.warning(issue)

        with st.spinner("Reading the image..."):
            result = run_model(image_bytes, mode)

        headline, confidence = render_result(result, mode)

        if mode == "Banana leaf":
            render_severity(leaf_severity(image_bytes))

        scan_key = f"{uploaded_file.name}-{uploaded_file.size}-{mode}"
        if st.session_state.last_logged != scan_key:
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Track": TRACKS[mode]["title"],
                "Result": headline,
                "Confidence": round(confidence, 1),
                "Photo": "Check photo" if issues else "Good",
            })
            st.session_state.last_logged = scan_key

render_tracking_log()

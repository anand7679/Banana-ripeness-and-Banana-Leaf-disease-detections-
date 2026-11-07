# import sys
# import tempfile
# from pathlib import Path

# import streamlit as st
# from PIL import Image

# PROJECT_DIR = Path(__file__).resolve().parent
# SRC_DIR = PROJECT_DIR / "src"
# sys.path.insert(0, str(SRC_DIR))

# from predict import predict as predict_ripeness
# from banana_leaf_disease.predict import predict as predict_leaf_disease


# RIPENESS_CHECKPOINT = PROJECT_DIR / "checkpoints" / "best_vit.pth"
# LEAF_DISEASE_CHECKPOINT = (
#     PROJECT_DIR / "checkpoints" / "banana_leaf_disease_vit.pth"
# )

# st.set_page_config(page_title="Banana Image Classifier", layout="wide")
# st.title("Banana Image Classifier")
# st.write(
#     "Upload a banana image to check both its ripeness and possible leaf disease."
# )

# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png"],
# )

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded image", use_container_width=True)

#     with tempfile.NamedTemporaryFile(
#         suffix=".jpg", prefix="banana_classifier_", delete=False
#     ) as temporary_file:
#         temporary_image = Path(temporary_file.name)
#         image.save(temporary_file, format="JPEG")

#     try:
#         ripeness_tab, disease_tab = st.tabs(
#             ["Banana ripeness", "Banana leaf disease"]
#         )

#         with ripeness_tab:
#             st.subheader("Ripeness result")
#             if not RIPENESS_CHECKPOINT.exists():
#                 st.error(f"Ripeness checkpoint not found: {RIPENESS_CHECKPOINT}")
#             else:
#                 result = predict_ripeness(
#                     str(temporary_image),
#                     checkpoint=str(RIPENESS_CHECKPOINT),
#                 )
#                 confidence = float(result["confidence"]) * 100
#                 if result["label"] == "BestToEat":
#                     st.success(f"Best to eat ({confidence:.2f}% confidence)")
#                 else:
#                     st.warning(
#                         f"Not best to eat ({confidence:.2f}% confidence)"
#                     )

#         with disease_tab:
#             st.subheader("Leaf disease result")
#             if not LEAF_DISEASE_CHECKPOINT.exists():
#                 st.info(
#                     "Leaf disease model is not trained yet. "
#                     f"Expected checkpoint: {LEAF_DISEASE_CHECKPOINT}"
#                 )
#             else:
#                 result = predict_leaf_disease(
#                     str(temporary_image),
#                     checkpoint_path=str(LEAF_DISEASE_CHECKPOINT),
#                 )
#                 disease = str(result["label"]).capitalize()
#                 confidence = float(result["confidence"]) * 100
#                 if disease.lower() == "healthy":
#                     st.success(
#                         f"Healthy leaf ({confidence:.2f}% confidence)"
#                     )
#                 else:
#                     st.warning(
#                         f"Possible {disease} disease "
#                         f"({confidence:.2f}% confidence)"
#                     )
#     finally:
#         temporary_image.unlink(missing_ok=True)


# import sys
# import tempfile
# import importlib.util
# from pathlib import Path

# import pandas as pd
# import streamlit as st
# from PIL import Image

# PROJECT_DIR = Path(__file__).resolve().parent
# sys.path.insert(0, str(PROJECT_DIR / "src_Banana_ripeness"))

# ripeness_module_spec = importlib.util.spec_from_file_location(
#     "banana_ripeness_predict",
#     PROJECT_DIR / "src_Banana_ripeness" / "predict.py",
# )
# if ripeness_module_spec is None or ripeness_module_spec.loader is None:
#     raise ImportError("Could not load the banana ripeness prediction module.")
# ripeness_module = importlib.util.module_from_spec(ripeness_module_spec)
# ripeness_module_spec.loader.exec_module(ripeness_module)
# predict_ripeness = ripeness_module.predict
# from banana_leaf_disease.predict import predict as predict_leaf_disease

# RIPENESS_CHECKPOINT = PROJECT_DIR / "checkpoints" / "best_vit.pth"
# LEAF_CHECKPOINT = PROJECT_DIR / "checkpoints" / "banana_leaf_disease_vit.pth"

# st.set_page_config(page_title="Banana Image Classifier", layout="wide")
# st.title("AI Banana Health Assistant")
# st.caption(
#     "Upload a banana fruit or leaf image. The trained vision model will "
#     "estimate the class and show how confident it is."
# )

# mode = st.radio("What are you uploading?", ["Banana fruit", "Banana leaf"], horizontal=True)
# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# def show_prediction(result, mode_name):
#     label = str(result["label"]).replace("_", " ").title()
#     confidence = float(result["confidence"]) * 100
#     raw_probabilities = result.get("probabilities")
#     if raw_probabilities is None:
#         raw_probabilities = {str(result["label"]): float(result["confidence"])}
#         st.info(
#             "This prediction did not include scores for every class, so the "
#             "chart shows the available prediction score only."
#         )
#     probabilities = {
#         str(name).replace("_", " ").title(): float(value) * 100
#         for name, value in raw_probabilities.items()
#     }

#     left, right = st.columns([1, 2])
#     with left:
#         st.metric("AI prediction", label)
#         st.metric("Confidence", f"{confidence:.1f}%")
#     with right:
#         chart_data = pd.DataFrame(
#             {"Confidence": probabilities.values()},
#             index=probabilities.keys(),
#         )
#         st.bar_chart(chart_data, horizontal=True, y_label="Class", x_label="Probability (%)")

#     if confidence >= 80:
#         st.success("The model is highly confident in this result.")
#     elif confidence >= 60:
#         st.warning(
#             "The model has moderate confidence. Use a clear, well-lit image "
#             "and consider checking another leaf or fruit."
#         )
#     else:
#         st.warning(
#             "The model is uncertain. Treat this as a screening result, not "
#             "a final agricultural diagnosis."
#         )

#     if mode_name == "Banana leaf":
#         guidance = {
#             "Healthy": (
#                 "No strong disease pattern was detected. Continue monitoring "
#                 "the plant and keep leaves well ventilated."
#             ),
#             "Cordana": (
#                 "Possible Cordana leaf spot. Remove severely affected leaves, "
#                 "avoid overhead watering, and consult a local expert about "
#                 "approved treatment."
#             ),
#             "Sigatoka": (
#                 "Possible Sigatoka leaf spot. Inspect nearby leaves, remove "
#                 "badly affected material, improve airflow, and consult a local "
#                 "expert about approved fungicide options."
#             ),
#         }
#         st.info(guidance.get(label, "Inspect the plant and consult an expert."))
#     else:
#         if label == "Besttoeat":
#             st.info("The model estimates that this fruit is in the best-to-eat class.")
#         else:
#             st.info("The model estimates that this fruit is not in the best-to-eat class.")

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded image", use_container_width=True)

#     with tempfile.TemporaryDirectory() as folder:
#         image_path = Path(folder) / "image.jpg"
#         image.save(image_path)

#         if mode == "Banana fruit":
#             result = predict_ripeness(str(image_path), checkpoint=str(RIPENESS_CHECKPOINT))
#             show_prediction(result, mode)

#         else:
#             result = predict_leaf_disease(str(image_path), checkpoint_path=str(LEAF_CHECKPOINT))
#             show_prediction(result, mode)


import html
import importlib.util
import sys
import tempfile
from pathlib import Path
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
    page_title="Banana Health & Ripeness Lab",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette: deep leaf-shadow ink, banana-peel gold, fresh leaf green, blight rust, warm paper text.
# Type: Fraunces (display serif) with Instrument Sans (body grotesque).
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #0E2620;
  --ink-2: #143229;
  --leaf: #8DC26F;
  --banana: #F2C230;
  --rust: #D9643A;
  --paper: #F4EBD6;
  --mute: rgba(244, 235, 214, 0.68);
  --line: rgba(244, 235, 214, 0.14);
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

html, body { background: var(--ink); }

/* Full-screen canvas: the app fills the entire viewport edge to edge. */
.stApp {
  background: var(--ink);
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
    radial-gradient(closest-side at 18% 28%, rgba(242, 194, 48, 0.20), transparent 70%),
    radial-gradient(closest-side at 82% 18%, rgba(141, 194, 111, 0.22), transparent 70%),
    radial-gradient(closest-side at 62% 88%, rgba(217, 100, 58, 0.16), transparent 70%);
  animation: aurora 26s ease-in-out infinite alternate;
}
@keyframes aurora {
  from { transform: translate3d(-4%, -2%, 0) rotate(0deg); }
  to   { transform: translate3d(5%, 3%, 0) rotate(8deg); }
}

[data-testid="stHeader"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  max-width: 100% !important;
  padding: 0 clamp(1.2rem, 4vw, 4.5rem) 4rem !important;
  position: relative;
  z-index: 1;
}

/* Hero: fills most of the first screen, with floating spores drifting upward. */
.hero {
  position: relative;
  min-height: 72vh;
  display: flex;
  align-items: center;
  padding: 8vh 0 6vh;
  overflow: hidden;
}
.hero-inner { position: relative; z-index: 1; max-width: 1100px; }
.hero h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 800;
  font-size: clamp(2.6rem, 6vw, 5.8rem);
  line-height: 1.0;
  letter-spacing: -0.025em;
  margin: 0 0 1.4rem;
  max-width: 17ch;
  color: var(--paper);
  animation: rise 0.9s var(--ease) both;
}
.hero .lede {
  font-size: clamp(1.02rem, 1.4vw, 1.25rem);
  line-height: 1.65;
  color: var(--mute);
  max-width: 56ch;
  margin: 0;
  animation: rise 0.9s 0.15s var(--ease) both;
}

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

/* Step guide: a clear three-part path through the app. */
.steps {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.8rem 1.2rem;
  margin-top: 2.4rem;
  animation: rise 0.9s 0.3s var(--ease) both;
}
.step {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  color: var(--paper);
  font-size: 1rem;
  font-weight: 500;
}
.step b {
  display: grid;
  place-items: center;
  width: 2.1rem;
  height: 2.1rem;
  border-radius: 50%;
  border: 1px solid var(--banana);
  color: var(--banana);
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 1rem;
}
.step-sep { width: 2.2rem; height: 1px; background: var(--line); }

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

/* Workspace panels. */
.panel-title {
  font-family: 'Fraunces', serif;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 0.3rem;
  color: var(--paper);
}
.panel-sub { color: var(--mute); margin: 0 0 1.4rem; line-height: 1.5; }

/* Mode switch as large pills. */
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

/* Large, inviting upload zone. */
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
[data-testid="stFileUploader"] span { color: var(--mute) !important; }

[data-testid="stImage"] img {
  border-radius: 18px;
  border: 1px solid var(--line);
}

/* Photo tips. */
.tips {
  margin: 1.2rem 0 0;
  padding: 1.1rem 1.3rem;
  list-style: none;
  border-radius: 16px;
  background: rgba(244, 235, 214, 0.04);
  border: 1px solid var(--line);
  display: grid;
  gap: 0.55rem;
  color: var(--mute);
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

/* Empty state: a breathing invitation. */
.empty {
  border: 1px dashed var(--line);
  border-radius: 24px;
  padding: 3rem 2rem;
  min-height: 420px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--mute);
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

/* Result readout. */
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
.verdict-kicker { color: var(--mute); font-size: 0.97rem; margin: 0; }
.verdict h2 {
  font-family: 'Fraunces', serif;
  font-size: clamp(2.2rem, 3.8vw, 3.6rem);
  font-weight: 600;
  line-height: 1.02;
  margin: 0.3rem 0 0.6rem;
  color: var(--paper);
}
.verdict p { margin: 0; color: var(--mute); line-height: 1.55; }

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
  color: var(--mute);
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

.info-note { color: var(--mute); font-size: 0.95rem; line-height: 1.6; margin-top: 1rem; }

.foot {
  margin-top: 4rem;
  padding-top: 1.6rem;
  border-top: 1px solid var(--line);
  color: var(--mute);
  font-size: 0.9rem;
  line-height: 1.6;
  max-width: 80ch;
}

@media (max-width: 720px) {
  .hero { min-height: auto; padding: 6vh 0 4vh; }
  .readout { grid-template-columns: 1fr; justify-items: start; gap: 1.4rem; }
  .bar { grid-template-columns: 6.5rem 1fr 3.6rem; gap: 0.7rem; }
  .empty { min-height: 260px; }
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


SUBJECT_PROMPTS = [
    "a photo of a banana leaf",
    "a photo of a banana fruit",
    "a photo of something that is not a banana",
]


@st.cache_resource(show_spinner="Loading the subject check...")
def load_subject_model():
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)


def detect_subject(image: Image.Image) -> str:
    """Return 'leaf', 'fruit', 'other', or 'unclear' using CLIP zero-shot classification."""
    model, processor = load_subject_model()
    inputs = processor(text=SUBJECT_PROMPTS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    best = int(probs.argmax())
    score = float(probs[best])
    if score < 0.5:
        return "unclear"
    return ["leaf", "fruit", "other"][best]


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
                <p class="lede">Show the model a banana fruit or a leaf. A compact
                vision transformer, built to run on edge devices, estimates the
                class and how sure it is, in a few moments.</p>
                <div class="steps">
                  <span class="step"><b>1</b> Choose fruit or leaf</span>
                  <span class="step-sep"></span>
                  <span class="step"><b>2</b> Upload a photo</span>
                  <span class="step-sep"></span>
                  <span class="step"><b>3</b> Read the result</span>
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
        normalized_label = label_raw.lower().replace("_", "").replace(" ", "").replace("-", "")
        if normalized_label == "besttoeat":
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


render_hero()

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown(
        compact(
            """
            <h2 class="panel-title">Upload a photo</h2>
            <p class="panel-sub">Pick what you are checking, then add an image.</p>
            """
        ),
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "What are you uploading?",
        ["Banana fruit", "Banana leaf"],
        horizontal=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
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

    image = None
    detected = "unclear"
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        detected = detect_subject(image)

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
            "This photo does not look like a banana  leaf. "
            "Upload a clear photo of one banana leaf."
        )
    elif mode == "Banana fruit" and detected == "leaf":
        st.warning(
            "This photo looks like a banana leaf, but Banana fruit is selected. "
            "Choose Banana leaf above to get a valid result."
        )
    elif mode == "Banana leaf" and detected == "fruit":
        st.warning(
            "This photo looks like a banana fruit, but Banana leaf is selected. "
            "Choose Banana fruit above to get a valid result."
        )
    else:
        with st.spinner("Reading the image..."):
            with tempfile.TemporaryDirectory() as folder:
                image_path = Path(folder) / "image.jpg"
                image.save(image_path)

                if mode == "Banana fruit":
                    result = predict_ripeness(str(image_path), checkpoint=str(RIPENESS_CHECKPOINT))
                else:
                    result = predict_leaf_disease(str(image_path), checkpoint_path=str(LEAF_CHECKPOINT))

        render_result(result, mode)

st.markdown(
    compact(
        """
        <p class="foot">A screening aid for education and field triage. It does
        not replace a professional agricultural diagnosis.</p>
        """
    ),
    unsafe_allow_html=True,
)
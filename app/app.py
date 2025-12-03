import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pandas as pd
import time
import random

# -----------------------
# Page config (must be near top)
# -----------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️", layout="wide")

# -----------------------
# Top-level style + 3D parallax wallpaper
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    :root{
        --bg:#0b0d0f;
        --card: rgba(255,255,255,0.04);
        --muted: #bfc7cf;
        --accent1: #6a5cff;
        --accent2: #00e0a7;
    }

    html, body, [class*="css"]  {
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        background: var(--bg) !important;
        color: #e6eef7;
    }

    /* Parallax layers (3 layers for depth) */
    .bg-layer {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 140%;
        height: 140%;
        background-repeat: no-repeat;
        background-position: center;
        background-size: cover;
        will-change: transform;
        z-index: -30;
        pointer-events: none;
        filter: saturate(0.9) contrast(0.95);
    }

    .bg-layer.layer1 {
        background-image: url('https://images.unsplash.com/photo-1532323544238-c714f0e17f6a?auto=format&fit=crop&w=1920&q=80');
        opacity: 0.36;
        transform: translate3d(-50%, 0, 0) scale(1.15);
        mix-blend-mode: normal;
        filter: brightness(0.28) blur(0.6px);
    }

    .bg-layer.layer2 {
        background-image: url('https://images.unsplash.com/photo-1523473827532-6c1d5b5d3be8?auto=format&fit=crop&w=1920&q=80');
        opacity: 0.22;
        transform: translate3d(-50%, 0, 0) scale(1.25);
        filter: brightness(0.22) blur(2px);
        mix-blend-mode: overlay;
    }

    .bg-layer.layer3 {
        background-image: linear-gradient(135deg, rgba(106,92,255,0.12), rgba(0,224,167,0.08));
        opacity: 1;
        transform: translate3d(-50%, 0, 0) scale(1.4);
        filter: blur(30px) brightness(0.9);
        mix-blend-mode: screen;
    }

    /* subtle vignette */
    .vignette {
        position: fixed;
        z-index: -10;
        top:0;left:0;width:100%;height:100%;
        background: radial-gradient(ellipse at center, rgba(0,0,0,0) 0%, rgba(0,0,0,0.55) 70%);
        pointer-events: none;
    }

    /* header */
    .header {
        padding: 32px 48px 6px 48px;
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .brand {
        font-weight: 800;
        font-size: 38px;
        letter-spacing: -0.6px;
        background: linear-gradient(90deg, #ffffff, #cbd7ff 40%, #9bcfdd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtle {
        color: var(--muted);
        margin-top: 6px;
        font-size: 14px;
    }

    /* creator & tech chips */
    .meta {
        display:flex;
        gap:10px;
        align-items:center;
        color: var(--muted);
        margin-left: auto;
        font-size: 13px;
    }
    .chip {
        padding:6px 10px;
        border-radius:999px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
        display:inline-flex;
        gap:8px;
        align-items:center;
    }

    /* main card layout */
    .card {
        background: var(--card);
        padding:22px;
        border-radius:18px;
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    .card:hover{ transform: translateY(-6px); box-shadow: 0 18px 50px rgba(2,6,23,0.75); }

    .section-title {
        font-size:20px;
        font-weight:700;
        margin-bottom:14px;
    }

    .predict-btn {
        border-radius:12px;
        padding:10px 18px;
        font-weight:700;
        background: linear-gradient(90deg, var(--accent1), #8b64ff);
        border: none;
        color: white;
        cursor:pointer;
        transition: transform 150ms ease;
    }
    .predict-btn:hover{ transform: translateY(-3px); box-shadow:0 12px 30px rgba(106,92,255,0.18); }

    .result {
        background: linear-gradient(180deg, rgba(0,0,0,0.12), rgba(255,255,255,0.02));
        padding: 14px;
        border-radius: 12px;
        border-left: 4px solid var(--accent2);
        font-weight:700;
    }

    .small {
        font-size:13px;
        color:var(--muted);
    }

    /* responsive canvas size */
    .canvas-wrap { display:flex; flex-direction:column; gap:12px; align-items:flex-start; }
    .tech-list { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; }

    /* probability bars */
    .prob-bars {
        margin-top:10px;
        display:flex;
        flex-direction:column;
        gap:8px;
    }
    .prob-row { display:flex; gap:12px; align-items:center; }
    .prob-bar {
        flex:1;
        height:12px;
        border-radius:8px;
        background: rgba(255,255,255,0.06);
        overflow:hidden;
    }
    .prob-fill {
        height:100%;
        border-radius:8px;
        background: linear-gradient(90deg, #6a5cff, #00e0a7);
    }

    /* small footer */
    .footer {
        color: var(--muted);
        padding: 28px 48px;
        font-size: 13px;
    }

    /* make sure Streamlit's top spacing reduced */
    .css-1d391kg { padding: 0 !important; }  /* (may vary by Streamlit version) */

    </style>

    <!-- 3 parallax layers -->
    <div class="bg-layer layer1"></div>
    <div class="bg-layer layer2"></div>
    <div class="bg-layer layer3"></div>
    <div class="vignette"></div>

    <script>
    // JS to drive parallax - different strength per layer
    (function(){
        const l1 = document.querySelector('.bg-layer.layer1');
        const l2 = document.querySelector('.bg-layer.layer2');
        const l3 = document.querySelector('.bg-layer.layer3');

        // mouse movement parallax
        document.addEventListener('mousemove', (e) => {
            const x = (e.clientX / window.innerWidth - 0.5);
            const y = (e.clientY / window.innerHeight - 0.5);
            if (l1) l1.style.transform = `translate3d(calc(-50% + ${x * 10}px), ${y * 8}px, 0) scale(1.15)`;
            if (l2) l2.style.transform = `translate3d(calc(-50% + ${x * 20}px), ${y * 14}px, 0) scale(1.25)`;
            if (l3) l3.style.transform = `translate3d(calc(-50% + ${x * 35}px), ${y * 28}px, 0) scale(1.45)`;
        });

        // scroll parallax
        window.addEventListener('scroll', () => {
            const s = window.scrollY;
            if (l1) l1.style.transform += ` translateY(${s*0.06}px)`;
            if (l2) l2.style.transform += ` translateY(${s*0.04}px)`;
            if (l3) l3.style.transform += ` translateY(${s*0.02}px)`;
        });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Header content (top)
# -----------------------
header_col1, header_col2 = st.columns([0.8, 0.2])
with header_col1:
    st.markdown(
        """
        <div class="header">
            <div>
                <div class="brand">✏️ MNIST Digit Classifier</div>
                <div class="subtle">Draw a 28×28 handwritten digit or upload an image — instant, beautiful predictions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    # Creator & quick meta
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:flex-end;">
            <div class="meta">
                <div class="chip">👨‍💻 <strong>Mayank Meena</strong></div>
                <div class="chip">🚀 Portfolio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------
# Main layout: canvas + uploader + controls
# -----------------------
left_col, right_col = st.columns([0.9, 1.1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎨 Draw Digit</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Use your mouse or touch to draw. Strong contrast works best.</div>', unsafe_allow_html=True)

    # drawing canvas
    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",  # transparent fill
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        width=340,
        height=340,
        drawing_mode="freedraw",
        key="canvas_v1"
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Predict and controls row
    controls_col1, controls_col2 = st.columns([0.35, 0.65])
    with controls_col1:
        # Using HTML styled button isn't directly clickable by streamlit, keep st.button for logic
        predict_clicked = st.button("Predict", key="predict", help="Predict digit from drawing or uploaded image")
    with controls_col2:
        st.markdown('<div class="small">Tip: Draw thick strokes and place the digit centrally for best accuracy.</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖼 Upload Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Upload a photo or PNG of a handwritten digit</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='margin-top:6px'>⚙️ Project Info</div>", unsafe_allow_html=True)
    # Tech stack chips
    st.markdown(
        """
        <div class="tech-list">
            <div class="chip">🐍 Python</div>
            <div class="chip">🔢 TensorFlow / Keras</div>
            <div class="chip">📊 NumPy / Pandas</div>
            <div class="chip">🖼 Pillow</div>
            <div class="chip">🎛 Streamlit</div>
            <div class="chip">🎨 streamlit-drawable-canvas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Prediction logic (placeholder model)
# -----------------------
def preprocess_img(img_pil):
    """Convert to 28x28 grayscale normalized"""
    img = img_pil.convert("L").resize((28, 28))
    arr = np.array(img)
    # if background is white and strokes black, invert to have white on black
    if np.mean(arr) > 127:
        arr = 255 - arr
    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1).astype(np.float32)
    return arr, img

def fake_model_predict(_arr):
    """Simulated model output for demo. Replace with real model.predict"""
    # produce realistic softmax-like probabilities
    probs = np.random.dirichlet(np.ones(10) * 2.2)
    pred = int(np.argmax(probs))
    return pred, probs

# Run prediction if button pressed
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

if predict_clicked:
    with st.spinner("⏳ Analyzing your digit..."):
        image_pil = None

        # prefer canvas
        if canvas and getattr(canvas, "image_data", None) is not None:
            try:
                np_img = canvas.image_data.astype("uint8")
                image_pil = Image.fromarray(np_img).convert("RGB")
            except Exception:
                image_pil = None

        # fallback to uploader
        if uploaded_file and image_pil is None:
            image_pil = Image.open(uploaded_file).convert("RGB")

        if image_pil is not None:
            arr, preview = preprocess_img(image_pil)
            # ---- Replace with your model inference call here ----
            pred_digit, probabilities = fake_model_predict(arr)
            # -----------------------------------------------------
            st.session_state['last_result'] = (pred_digit, probabilities.tolist(), preview)
            # small delay to show spinner nicely
            time.sleep(0.6)
        else:
            st.error("Please draw on the canvas or upload an image to predict.")
            st.session_state['last_result'] = None

# -----------------------
# Display results (if available)
# -----------------------
if st.session_state.get('last_result'):
    pred_digit, probabilities, preview = st.session_state['last_result']
    st.markdown("<br>", unsafe_allow_html=True)
    # two-column result display
    res_col1, res_col2 = st.columns([0.6, 1.4])
    with res_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">✅ Predicted Digit</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result" style="font-size:20px">Predicted: <span style="font-size:28px">{pred_digit}</span></div>', unsafe_allow_html=True)

        # show the processed 28x28 preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(preview.resize((140, 140)), width=140, caption="Model Input (28×28 preview)")

        st.markdown("</div>", unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Probabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Confidence per class (0-9)</div>', unsafe_allow_html=True)
        # visual bars
        probs = np.array(probabilities)
        for i, p in enumerate(probs):
            # color width scaled
            bar_html = f"""
            <div class="prob-row">
                <div style="width:36px; font-weight:700;">{i}</div>
                <div class="prob-bar"><div class="prob-fill" style="width:{p*100}%;"></div></div>
                <div style="width:64px; text-align:right; font-weight:700;">{p*100:4.1f}%</div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Footer / credits
# -----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                Made with ❤️ by <strong>Mayank Meena</strong> — Built using Python, TensorFlow/Keras, NumPy, Pillow, and Streamlit.
            </div>
            <div class="small">Tip: For best results, center the digit and use thick strokes.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
from PIL import Image
import torch
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import os
import io, base64
from streamlit_cropper import st_cropper

# (Removed drawable-canvas compatibility shim)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🖼️ Image Translator", layout="wide")

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

# Initialize Groq Vision client securely
groq_client = Groq(api_key=api_key)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PAGE UI ----------------
st.title("🖼️ Image Caption → Malayalam Translator")
st.caption("Select a region on the image manually, generate its English description using Groq Vision, then translate it into Malayalam using Hugging Face Transformers.")

# ---------------- Load Translation Model ----------------
@st.cache_resource
def load_translation_pipeline():
    """Load and cache the translation pipeline."""
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-ml",
        device=0 if DEVICE == "cuda" else -1,
    )
    return translator

translator = load_translation_pipeline()

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("✏️ Draw a rectangle to select region")

    # Use streamlit-cropper for ROI selection; return cropped PIL image directly
    cropped_preview = st_cropper(
        image,
        realtime_update=True,
        box_color="#ff4b4b",
        aspect_ratio=None,
        return_type="image",
        key="cropper",
    )

    cropped_img = None
    rect_info = None
    if cropped_preview is not None:
        # Treat any returned image as a valid selection
        cropped_img = cropped_preview

    if cropped_img is not None:
        st.image(cropped_img, caption="Cropped Region", use_container_width=False)
        st.caption(f"Selected bbox: {rect_info}")
        st.session_state["cropped_img"] = cropped_img
    else:
        st.info("Draw a rectangle on the image to select a region.")

    # Step 1: Generate English description using Groq
    if st.button("🧠 Generate English Description"):
        target_img = st.session_state.get("cropped_img")
        if target_img is None:
            st.warning("Please draw a rectangle to select a region first.")
        else:
            buffered = io.BytesIO()
            target_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            with st.spinner("🔍 Generating English description from Groq Vision model..."):
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert vision analyst. Provide an accurate, thorough, and objective description of the provided image region in 2-4 sentences, and don't add bullet point or numbers or anything else just give me the caption in simple english ",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please describe this selected image region in depth."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ],
                        }
                    ],
                )

                try:
                    content = response.choices[0].message.content
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                            elif isinstance(part, str):
                                parts.append(part)
                        description = " "+" ".join(parts).strip()
                    else:
                        description = str(content).strip()
                except Exception as e:
                    st.error(f"❌ Failed to parse Groq response: {e}")
                    with st.expander("Show raw Groq response"):
                        st.write(response)
                    st.stop()

            st.session_state["description"] = description
            st.success(f"**English Description:** {description}")

        # Step 2: Translate English → Malayalam
    if st.button("🌐 Translate to Malayalam"):
        description = st.session_state.get("description")
        if not description:
            st.warning("Generate the English description first.")
        else:
            with st.spinner("🌍 Translating to Malayalam using Hugging Face Transformers..."):
                translated = translator(description)
                final_translation = translated[0]["translation_text"]

            st.session_state["translation"] = final_translation
            st.success(f"**Malayalam Translation:** {final_translation}")

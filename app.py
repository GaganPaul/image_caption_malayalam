import streamlit as st
from PIL import Image
import torch
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import os
import io, base64
from streamlit_drawable_canvas import st_canvas

# --- Compatibility shim for streamlit-drawable-canvas with newer Streamlit ---
try:
    import streamlit.elements.image as _st_image_mod
    if not hasattr(_st_image_mod, "image_to_url"):
        # Streamlit moved/changed image_to_url signature; provide adapter for (image, width:int)
        from streamlit.elements.lib import image_utils as _st_image_utils
        from types import SimpleNamespace

        def _compat_image_to_url(*args, **kwargs):
            # Old usage from drawable-canvas: image_to_url(image, width:int, image_format=None, clamp=False, channels="RGB", output_format="PNG")
            # New API: image_to_url(image_data, layout_config, image_format=None, clamp=False, channels="RGB", output_format="PNG")
            if len(args) == 0:
                # Fallback to new API directly
                return _st_image_utils.image_to_url(*args, **kwargs)

            image_data = args[0]
            width_or_layout = args[1] if len(args) > 1 else None
            image_format = args[2] if len(args) > 2 else None
            clamp = args[3] if len(args) > 3 else False
            channels = args[4] if len(args) > 4 else "RGB"
            output_format = args[5] if len(args) > 5 else "PNG"

            if hasattr(width_or_layout, "width"):
                layout_config = width_or_layout
            else:
                layout_config = SimpleNamespace(width=width_or_layout)

            return _st_image_utils.image_to_url(
                image_data,
                layout_config,
                image_format,
                clamp,
                channels,
                output_format,
            )

        _st_image_mod.image_to_url = _compat_image_to_url
except Exception:
    # Best-effort; if anything goes wrong, st_canvas may still handle gracefully
    pass

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
        model="kurianbenoy/kde_en_ml_translation_model",
        device=0 if DEVICE == "cuda" else -1,
    )
    return translator

translator = load_translation_pipeline()

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("✏️ Draw a rectangle to select region")

    # Determine canvas display size (keep aspect ratio)
    max_canvas_width = 900
    canvas_width = min(max_canvas_width, image.width)
    canvas_height = int(image.height * (canvas_width / image.width))

    # Draw on canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.15)",
        stroke_width=2,
        stroke_color="#ff4b4b",
        background_image=image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
    )

    cropped_img = None
    rect_info = None

    if canvas_result and canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        rects = [obj for obj in objects if obj.get("type") == "rect"]
        if rects:
            last_rect = rects[-1]
            left = float(last_rect.get("left", 0))
            top = float(last_rect.get("top", 0))
            rect_w = float(last_rect.get("width", 0))
            rect_h = float(last_rect.get("height", 0))

            # Scale from canvas coords to original image coords
            scale_x = image.width / canvas_width
            scale_y = image.height / canvas_height

            x0 = max(0, int(left * scale_x))
            y0 = max(0, int(top * scale_y))
            x1 = min(image.width, int((left + rect_w) * scale_x))
            y1 = min(image.height, int((top + rect_h) * scale_y))

            if x1 > x0 and y1 > y0:
                cropped_img = image.crop((x0, y0, x1, y1))
                rect_info = (x0, y0, x1, y1)

    if cropped_img is not None:
        st.image(cropped_img, caption="Cropped Region", use_container_width=False)
        st.caption(f"Selected bbox: {rect_info}")
        st.session_state["cropped_img"] = cropped_img
    else:
        st.info("Draw a rectangle on the image to select a region.")

    # System prompt for detailed captioning
    default_system_prompt = (
        "You are an expert vision analyst. Provide an accurate, thorough, and objective "
        "description of the provided image region. Include salient objects, their attributes "
        "(colors, shapes, textures), spatial relationships, visible text (OCR-like), and any "
        "relevant context. Avoid speculation beyond visible evidence. Use clear prose in 2-4 sentences."
    )
    system_prompt = st.text_area(
        "✍️ System prompt (optional, used for in-depth descriptions)",
        value=st.session_state.get("system_prompt", default_system_prompt),
        help="Customize the behavior and level of detail for the description.",
    )
    st.session_state["system_prompt"] = system_prompt

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
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please describe the selected image region in-depth as instructed."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ],
                        }
                    ],
                )

                try:
                    # Groq SDK returns objects; message.content is usually a string
                    description = response.choices[0].message.content
                    if isinstance(description, list):
                        # Defensive: flatten any content parts with 'text'
                        parts = []
                        for part in description:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                            elif isinstance(part, str):
                                parts.append(part)
                        description = " ".join(parts)
                    description = (description or "").strip()
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

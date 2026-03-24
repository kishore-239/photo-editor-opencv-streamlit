import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Photo Editor", layout="wide")
st.title("📸 Photo Editor using OpenCV & Streamlit")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Helper Functions
# -------------------------------
def to_gray_safe(img):
    # Returns grayscale safely for both color and grayscale inputs
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_bgr_safe(img):
    # Ensures image is 3-channel BGR
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# -------------------------------
# Main App
# -------------------------------
if uploaded_file:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Convert RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Show original
    st.subheader("Original Image")
    st.image(image, width="stretch")

    # -------------------------------
    # Resize
    # -------------------------------
    st.sidebar.subheader("Resize")
    width = st.sidebar.slider("Width", 100, 1500, img.shape[1])
    height = st.sidebar.slider("Height", 100, 1500, img.shape[0])
    img = cv2.resize(img, (width, height))

    # -------------------------------
    # Rotation
    # -------------------------------
    st.sidebar.subheader("Rotation")
    angle = st.sidebar.slider("Rotate Image", -180, 180, 0)

    if angle != 0:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))

    # -------------------------------
    # Brightness & Contrast
    # -------------------------------
    st.sidebar.subheader("Adjustments")
    alpha = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
    beta = st.sidebar.slider("Brightness", -100, 100, 0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # -------------------------------
    # Filters
    # -------------------------------
    st.sidebar.subheader("Filters")
    filter_option = st.sidebar.selectbox(
        "Choose Filter",
        ["None", "Grayscale", "Blur", "Sharpen", "Warm", "Edge Detection"]
    )

    if filter_option == "Grayscale":
        img = to_gray_safe(img)

    elif filter_option == "Blur":
        k = st.sidebar.slider("Blur Intensity", 1, 25, 5)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    elif filter_option == "Sharpen":
        img_color = to_bgr_safe(img)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img_color, -1, kernel)

    elif filter_option == "Warm":
        img_color = to_bgr_safe(img)
        img_color[:, :, 2] = np.clip(img_color[:, :, 2] + 30, 0, 255)
        img = img_color

    elif filter_option == "Edge Detection":
        gray = to_gray_safe(img)
        img = cv2.Canny(gray, 100, 200)

    # -------------------------------
    # Portrait Mode (Background Blur)
    # -------------------------------
    st.sidebar.subheader("Portrait Mode")
    portrait_mode = st.sidebar.checkbox("Enable Background Blur")

    if portrait_mode:
        # Ensure color image for processing
        img_color = to_bgr_safe(img)

        gray = to_gray_safe(img_color)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

        mask = np.zeros_like(gray)

        for (x, y, w, h) in faces:
            mask[y:y+h, x:x+w] = 255

        blurred = cv2.GaussianBlur(img_color, (25, 25), 0)
        img = np.where(mask[:, :, None] == 255, img_color, blurred)

    # -------------------------------
    # Convert for display
    # -------------------------------
    if len(img.shape) == 2:
        display_img = img
    else:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # Show edited image
    # -------------------------------
    st.subheader("Edited Image")
    st.image(display_img, width="stretch")

    # -------------------------------
    # Download Button
    # -------------------------------
    result = Image.fromarray(display_img)

    buf = io.BytesIO()
    result.save(buf, format="PNG")

    st.download_button(
        label="📥 Download Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )

else:
    st.info("⬅️ Upload an image from the sidebar to start editing.")
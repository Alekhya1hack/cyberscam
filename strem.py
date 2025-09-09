from PIL import Image
from background_detector import load_models, predict_background

cnn, rf = load_models("models/cnn_model.h5", "models/rf_model.joblib")

uploaded_file = st.file_uploader("Upload poster", type=["png","jpg","jpeg"])
if uploaded_file:
    pil = Image.open(uploaded_file).convert("RGB")
    prob, label = predict_background(pil, templates="templates", cnn=cnn, rf=rf)
    if prob is None:
        st.info("No detection model available.")
    else:
        st.write(f"Background fake probability: {prob:.2f}")
        if label == "fake":
            st.warning("⚠️ Background likely manipulated")
        else:
            st.success("✅ Background likely genuine")

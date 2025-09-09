import os
import re
import csv
import cv2
import numpy as np
import imagehash
import streamlit as st
import Levenshtein
from datetime import datetime
from PIL import Image, ImageStat

# ---------------- CONFIG ----------------
BLACKLIST_FILE = "blacklist.txt"
LOG_FILE = "blacklist_log.csv"
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]
TEMPLATES_PATH = "templates/genuine"

# ---------------- HELPER FUNCTIONS ----------------
def load_blacklist():
    try:
        with open(BLACKLIST_FILE, "r") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def save_blacklist(blacklist):
    with open(BLACKLIST_FILE, "w") as f:
        for upi in sorted(set(blacklist)):
            f.write(f"{upi}\n")

def log_blacklist_addition(upi_id, added_by="web_app"):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "UPI_ID", "Added_By"])
        writer.writerow([datetime.now().isoformat(), upi_id, added_by])

def is_valid_format(upi_id: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9.\-_]+@[a-zA-Z]+$", upi_id))

def check_similarity(upi_id: str, threshold=0.85):
    for black in BLACKLIST:
        ratio = Levenshtein.ratio(upi_id, black)
        if ratio >= threshold:
            return black, ratio
    return None, 0

def check_upi_id(upi_id: str):
    upi_id = upi_id.lower().strip()
    if not is_valid_format(upi_id):
        return "invalid", f"❌ Invalid UPI format: {upi_id}"
    if upi_id in BLACKLIST:
        return "danger", f"🚫 Blacklisted UPI ID: {upi_id}"
    for kw in KEYWORDS:
        if kw in upi_id:
            return "danger", f"⚠ Suspicious keyword '{kw}' found in {upi_id}"
    similar, score = check_similarity(upi_id)
    if similar:
        return "danger", f"⚠ '{upi_id}' looks similar to '{similar}' (score: {score:.2f})"
    return "safe", f"✅ Safe UPI ID: {upi_id}"

# ----------- Payment Screenshot Authenticity Functions -----------
def is_blurry(image: Image.Image, threshold=100):
    open_cv_image = np.array(image.convert("L"))
    variance = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    return variance < threshold

def average_hash_similarity(img1, img2):
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    max_hash_bits = len(hash1.hash) ** 2
    diff = hash1 - hash2
    similarity = 1 - (diff / max_hash_bits)
    return similarity

def check_similarity_with_templates(uploaded_img: Image.Image, threshold=0.75):
    best_score = 0
    for filename in os.listdir(TEMPLATES_PATH):
        try:
            template_img = Image.open(os.path.join(TEMPLATES_PATH, filename)).convert("RGB")
            score = average_hash_similarity(uploaded_img, template_img)
            if score > best_score:
                best_score = score
        except Exception:
            continue
    return best_score >= threshold, best_score

# ---------------- LOAD DATA ----------------
BLACKLIST = load_blacklist()

# ---------------- STREAMLIT APP ----------------
st.sidebar.title("⚡ Scam Detector")
menu = st.sidebar.radio("Choose Feature", ["🔐 UPI Scam Detector", "🔍 Payment Screenshot Authenticity"])

if menu == "🔐 UPI Scam Detector":
    st.title("🔐 UPI Scam Detector")
    st.write("Check if a UPI ID looks suspicious, blacklisted, or safe")
    upi_input = st.text_input("Enter a UPI ID to check:")
    if st.button("Check UPI"):
        if upi_input:
            status, message = check_upi_id(upi_input)
            if status == "safe":
                st.success(message)
            elif status == "invalid":
                st.warning(message)
            else:
                st.error(message)
        else:
            st.warning("Please enter a UPI ID first.")
    st.subheader("➕ Add UPI ID to Blacklist")
    new_upi = st.text_input("Enter a UPI ID to blacklist:")
    if st.button("Add to Blacklist"):
        if new_upi:
            new_upi = new_upi.lower().strip()
            if new_upi not in BLACKLIST:
                BLACKLIST.append(new_upi)
                save_blacklist(BLACKLIST)
                log_blacklist_addition(new_upi, added_by="streamlit_user")
                st.success(f"✅ '{new_upi}' added to blacklist and logged!")
            else:
                st.info(f"ℹ '{new_upi}' is already in the blacklist.")
        else:
            st.warning("Please enter a UPI ID before adding.")
    st.subheader("📄 Current Blacklist")
    if BLACKLIST:
        st.write(f"Total blacklisted UPI IDs: {len(BLACKLIST)}")
        st.dataframe({"Blacklisted UPI IDs": BLACKLIST})
    else:
        st.write("✅ No blacklisted UPI IDs found.")
    if os.path.exists(LOG_FILE):
        st.subheader("🕒 Blacklist Addition Log")
        with open(LOG_FILE, "r") as csvfile:
            st.download_button("📥 Download Log", csvfile, file_name="blacklist_log.csv")

elif menu == "🔍 Payment Screenshot Authenticity":
    st.title("🔍 Payment Screenshot Authenticity Checker")
    uploaded_file = st.file_uploader("Upload payment screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        uploaded_img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_img, caption="Uploaded Screenshot", use_column_width=True)

        blurry = is_blurry(uploaded_img)
        similar, similarity_score = check_similarity_with_templates(uploaded_img)

        if blurry:
            st.warning("⚠ Uploaded image appears blurry. For expected accuracy, upload a clear screenshot.")
        if similar:
            st.success(f"✅ Screenshot closely matches genuine templates (similarity: {similarity_score:.2f}). Likely authentic.")
        else:
            st.error(f"⚠ Screenshot does NOT match genuine templates well (similarity: {similarity_score:.2f}). Possible fake or edited.")

        stat = ImageStat.Stat(uploaded_img)
        st.write(f"Image Brightness (mean): {stat.mean[0]:.2f}")
        st.write(f"Image Contrast (stddev): {stat.stddev[0]:.2f}")
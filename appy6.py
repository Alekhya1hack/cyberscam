import os
import re
import csv
import cv2
import numpy as np
import imagehash
import Levenshtein
from datetime import datetime
from PIL import Image
import streamlit as st

# ---------------- CONFIG ----------------
BLACKLIST_FILE = "blacklist.txt"
LOG_FILE = "blacklist_log.csv"
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

# ---------------- HELPER FUNCTIONS (UPI) ----------------
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
        return "danger", f"⚠ '{upi_id}' looks similar to blacklisted '{similar}' (score: {score:.2f})"
    return "safe", f"✅ Safe UPI ID: {upi_id}"

# ---------------- HELPER FUNCTIONS (QR) ----------------
def orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def phash_similarity(img1, img2):
    hash1 = imagehash.phash(Image.fromarray(img1))
    hash2 = imagehash.phash(Image.fromarray(img2))
    return 1 - (hash1 - hash2) / len(hash1.hash)**2

def check_background(uploaded_img, templates_path="templates/"):
    uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
    best_score, best_type = 0, "Unknown"
    for category in ["genuine", "fake"]:
        folder = os.path.join(templates_path, category)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            template = cv2.imread(os.path.join(folder, file))
            if template is None:
                continue
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            orb_score = orb_similarity(uploaded_gray, template_gray)
            phash_score = phash_similarity(uploaded_gray, template_gray)
            combined = orb_score * 0.7 + phash_score * 0.3
            if combined > best_score:
                best_score, best_type = combined, category
    return best_type, best_score

# ---------------- LOAD DATA ----------------
BLACKLIST = load_blacklist()

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Scam Detector", layout="wide")
st.title("🛡 Scam Detector Dashboard")

tab1, tab2 = st.tabs(["🔐 UPI Scam Detector", "🔍 QR Background Checker"])

# -------- TAB 1: UPI Scam Detector --------
with tab1:
    st.subheader("Check a UPI ID")
    upi_input = st.text_input("Enter a UPI ID to check:")
    if st.button("Check UPI ID"):
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

    st.subheader("➕ Add to Blacklist")
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
            st.warning("Please enter a UPI ID first.")

    st.subheader("📄 Current Blacklist")
    if BLACKLIST:
        st.dataframe({"Blacklisted UPI IDs": BLACKLIST})
    else:
        st.write("✅ No blacklisted UPI IDs found.")

    if os.path.exists(LOG_FILE):
        st.subheader("🕒 Blacklist Log")
        with open(LOG_FILE, "r") as csvfile:
            st.download_button("📥 Download Log", csvfile, file_name="blacklist_log.csv")

# -------- TAB 2: QR Scam Detector --------
with tab2:
    st.subheader("Check QR Code Background")
    uploaded_file = st.file_uploader("Upload QR code poster", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        st.image(img, caption="Uploaded QR", use_container_width=True)

        result, score = check_background(cv_img)

        if result == "genuine":
            st.success(f"✅ Looks Genuine (Score: {score:.2f})")
        elif result == "fake":
            st.error(f"⚠ Suspicious Background Detected! (Score: {score:.2f})")
        else:
            st.warning("Could not classify background.")

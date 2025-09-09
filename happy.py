import streamlit as st
import re
import Levenshtein
import csv
import os
from datetime import datetime
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import pytesseract

# ---------------- CONFIG ----------------
BLACKLIST_FILE = "blacklist.txt"
LOG_FILE = "scan_logs.csv"


# Suspicious keywords
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

# ---------------- HELPER FUNCTIONS ----------------
def load_blacklist():
    try:
        with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def save_blacklist(blacklist):
    with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
        for upi in sorted(set(blacklist)):
            f.write(f"{upi}\n")

def log_check(check_type, details, score, verdict):
    """Logs every scan with timestamp"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Type", "Details", "Score", "Verdict"])
        writer.writerow([datetime.now().isoformat(), check_type, details, score, verdict])

def is_valid_format(upi_id: str) -> bool:
    """Check if UPI matches standard format: name@bank"""
    return bool(re.match(r"^[a-zA-Z0-9.\-_]+@[a-zA-Z]+$", upi_id))

def check_similarity(upi_id: str, threshold=0.85):
    """Check if UPI ID is very similar to a known blacklist entry"""
    for black in BLACKLIST:
        ratio = Levenshtein.ratio(upi_id, black)
        if ratio >= threshold:
            return black, ratio
    return None, 0

def check_upi_id(upi_id: str):
    upi_id = upi_id.lower().strip()

    # 1. Format check
    if not is_valid_format(upi_id):
        return "invalid", f"❌ Invalid UPI format: {upi_id}"

    # 2. Direct blacklist check
    if upi_id in BLACKLIST:
        return "danger", f"🚫 Blacklisted UPI ID: {upi_id}"

    # 3. Keyword-based detection
    for kw in KEYWORDS:
        if kw in upi_id:
            return "danger", f"⚠ Suspicious keyword '{kw}' found in {upi_id}"

    # 4. Similarity check
    similar, score = check_similarity(upi_id)
    if similar:
        return "danger", f"⚠ UPI ID '{upi_id}' looks similar to blacklisted '{similar}' (score: {score:.2f})"

    return "safe", f"✅ Safe UPI ID: {upi_id}"

# ---------------- Screenshot Forensics ----------------
def error_level_analysis(img: Image.Image):
    """Basic ELA check"""
    ela_img = img.copy()
    ela_img.save("temp.jpg", quality=90)
    ela = Image.open("temp.jpg")
    diff = ImageChops.difference(img, ela)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return diff, max_diff

def ocr_extract_text(img: Image.Image):
    """Extract text from screenshot using OCR"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def validate_payment_text(text: str):
    """Check if payment related keywords exist"""
    expected_keywords = ["paid", "upi", "to", "success", "₹", "rs", "payment", "transaction"]
    found = [kw for kw in expected_keywords if kw.lower() in text.lower()]
    score = 0
    if len(found) < 3:  # too few matches → likely fake
        score = 40
    return found, score

def analyze_screenshot(img: Image.Image):
    ela, max_diff = error_level_analysis(img)
    results = {}
    risk_score = 0

    # ELA score
    ela_score = 30 if max_diff > 40 else 5
    results["ELA"] = f"Max diff = {max_diff}"
    risk_score += ela_score

    # OCR text validation
    text = ocr_extract_text(img)
    validations, text_score = validate_payment_text(text)
    results["OCR"] = validations
    risk_score += text_score

    # Final scoring
    if risk_score >= 50:
        verdict = "🚫 Fake / Edited Screenshot"
    elif 25 <= risk_score < 50:
        verdict = "⚠ Suspicious Screenshot"
    else:
        verdict = "✅ Genuine Screenshot"

    return ela, text, results, risk_score, verdict

# ---------------- LOAD DATA ----------------
BLACKLIST = load_blacklist()

# ---------------- STREAMLIT APP ----------------
menu = st.sidebar.radio("Choose Option", ["🔐 UPI Scam Detector", "🖼️ Screenshot Forensics", "📄 Blacklist Manager"])

if menu == "🔐 UPI Scam Detector":
    st.title("🔐 UPI Scam Detector")
    st.write("Check if a UPI ID looks suspicious, blacklisted, or safe")

    upi_input = st.text_input("Enter a UPI ID to check:")

    if st.button("Check"):
        if upi_input:
            status, message = check_upi_id(upi_input)
            score = 0 if status == "safe" else 70
            if status == "safe":
                st.success(message)
            elif status == "invalid":
                st.warning(message)
            else:
                st.error(message)
            log_check("UPI", upi_input, score, message)
        else:
            st.warning("Please enter a UPI ID first.")

elif menu == "🖼️ Screenshot Forensics":
    st.title("🖼️ Screenshot Forensics")
    uploaded_ss = st.file_uploader("Upload payment/review screenshot", type=["png", "jpg", "jpeg"])

    if uploaded_ss:
        img = Image.open(uploaded_ss).convert("RGB")
        st.image(img, caption="Uploaded Screenshot", width="stretch")

        ela, text, results, risk, verdict = analyze_screenshot(img)
        st.image(ela, caption="ELA Analysis", width="stretch")
        st.text_area("Extracted Text", text, height=150)
        st.write("**Analysis Results:**")
        for key, val in results.items():
            st.write(f"{key}: {val}")
        st.progress(risk)
        st.metric("Risk Score", f"{risk}/100")
        st.write("**Verdict:**", verdict)
        log_check("Screenshot", uploaded_ss.name, risk, verdict)

elif menu == "📄 Blacklist Manager":
    st.title("📄 Blacklist Manager")

    st.subheader("➕ Add UPI ID to Blacklist")
    new_upi = st.text_input("Enter a UPI ID to blacklist:")
    if st.button("Add to Blacklist"):
        if new_upi:
            new_upi = new_upi.lower().strip()
            if new_upi not in BLACKLIST:
                BLACKLIST.append(new_upi)
                save_blacklist(BLACKLIST)
                st.success(f"✅ '{new_upi}' added to blacklist!")
            else:
                st.info(f"ℹ '{new_upi}' is already in the blacklist.")
        else:
            st.warning("Please enter a UPI ID before adding.")

    st.subheader("📌 Current Blacklist")
    if BLACKLIST:
        st.write(f"Total blacklisted UPI IDs: {len(BLACKLIST)}")
        st.dataframe({"Blacklisted UPI IDs": BLACKLIST})
        delete_upi = st.text_input("Enter a UPI ID to delete (Admin only):")
        if st.button("Delete from Blacklist"):
            if delete_upi in BLACKLIST:
                BLACKLIST.remove(delete_upi)
                save_blacklist(BLACKLIST)
                st.success(f"🗑️ '{delete_upi}' removed from blacklist!")
            else:
                st.error(f"'{delete_upi}' not found in blacklist.")
    else:
        st.write("✅ No blacklisted UPI IDs found.")

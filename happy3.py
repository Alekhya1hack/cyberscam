import re
import csv
import os
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import pytesseract

# ---------- Logging ----------
def log_check(check_type, details, score, verdict):
    file_exists = os.path.isfile("scan_log.csv")
    with open("scan_log.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Type", "Details", "Score", "Verdict"])
        writer.writerow([datetime.now().isoformat(), check_type, details, score, verdict])

# ---------- UPI Checker ----------
def is_valid_format(upi_id: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9.\-_]+@[a-zA-Z]+$", upi_id))

def check_upi(upi_id: str):
    score = 0
    reasons = []

    if not is_valid_format(upi_id):
        reasons.append("Invalid UPI format ❌")
        score += 50
    else:
        reasons.append("Valid UPI format ✅")

    risky = ["fraud", "scam", "test", "fake", "pmcaree", "pmcares"]
    if any(x in upi_id.lower() for x in risky):
        reasons.append("Suspicious keyword in UPI ID ❌")
        score += 50

    verdict = "Fake ⚠️" if score >= 50 else "Likely Genuine ✅"
    return score, verdict, reasons

# ---------- QR Scanner ----------
def analyze_qr(img: Image.Image):
    qr = cv2.QRCodeDetector()
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    data, points, _ = qr.detectAndDecode(img_cv)

    if not data:
        return None, 50, "Unscannable / Risky ❌"
    
    risk = 0
    if not is_valid_format(data):
        risk += 50
    verdict = "Fake ⚠️" if risk >= 50 else "Likely Genuine ✅"
    return data, risk, verdict

# ---------- Screenshot Forensics ----------

def ocr_extract_text(img: Image.Image):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def validate_payment_text(text: str):
    suspicious = ["gift", "reward", "lottery", "urgent", "donate"]
    score = 0
    findings = []
    for word in suspicious:
        if word.lower() in text.lower():
            findings.append(f"Suspicious keyword: {word} ❌")
            score += 20
    return findings, score

def validate_transaction_id(text: str):
    pattern = r"\b[a-zA-Z0-9]{8,35}\b"
    matches = re.findall(pattern, text)
    findings = []
    score = 0

    if matches:
        findings.append("Transaction ID format found ✅")
    else:
        findings.append("No valid transaction ID found ❌")
        score += 50
    return findings, score, matches

def ela_check(img: Image.Image):
    # For demonstration, we'll simulate ELA by slightly altering image brightness and comparing
    enhanced = ImageEnhance.Brightness(img).enhance(1.1)
    diff = ImageChops.difference(img, enhanced)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema])
    # If difference is noticeable, treat as edited
    if max_diff > 10:
        return "Edited ❌", 20
    else:
        return "Not edited ✅", 0

def analyze_screenshot(img: Image.Image):
    results = {}
    risk_score = 0

    # ELA Check
    ela_result, ela_score = ela_check(img)
    results["ELA Check"] = ela_result
    risk_score += ela_score

    # OCR Text Extraction
    text = ocr_extract_text(img)

    # Transaction ID Check
    tx_findings, tx_score, tx_matches = validate_transaction_id(text)
    results["Transaction ID Check"] = tx_findings
    risk_score += tx_score

    # Suspicious Keywords Check
    keyword_findings, keyword_score = validate_payment_text(text)
    results["Suspicious Keywords"] = keyword_findings
    risk_score += keyword_score

    verdict = "Fake ⚠️" if risk_score >= 50 else "Likely Genuine ✅"
    return ela_result, text, results, risk_score, verdict

# ---------- Streamlit UI ----------
st.sidebar.title("🛡️ Scam Detector")
menu = st.sidebar.radio("Choose Module", ["Optional", "UPI Checker", "QR Scanner", "Screenshot Forensics"])

if menu == "Optional":
    st.write("👉 Select a module from the sidebar to check UPI, QR, or Screenshot.")

elif menu == "UPI Checker":
    st.title("🔎 UPI ID Scam Detector")
    upi_input = st.text_input("Enter UPI ID:")
    if st.button("Check UPI"):
        score, verdict, reasons = check_upi(upi_input)
        st.metric("Risk Score", f"{score}/100")
        st.write("**Verdict:**", verdict)
        for r in reasons:
            st.write("-", r)
        log_check("UPI", upi_input, score, verdict)

elif menu == "QR Scanner":
    st.title("🔍 QR Scam Detector")
    uploaded_qr = st.file_uploader("Upload QR Code", type=["png", "jpg", "jpeg"])
    if uploaded_qr:
        img = Image.open(uploaded_qr).convert("RGB")
        st.image(img, caption="Uploaded QR", width=300)
        result, risk, verdict = analyze_qr(img)
        if result:
            st.write("**Decoded UPI:**", result)
        else:
            st.write("**Decoded UPI:** Not found ❌")
        st.metric("Risk Score", f"{risk}/100")
        st.write("**Verdict:**", verdict)
        log_check("QR", result or "Not found", risk, verdict)

elif menu == "Screenshot Forensics":
    st.title("🖼️ Screenshot Forensics")
    uploaded_ss = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_ss:
        img = Image.open(uploaded_ss).convert("RGB")
        st.image(img, caption="Uploaded Screenshot", width=500)
        
        ela_result, text, results, score, verdict = analyze_screenshot(img)
        
        st.write("**ELA Result:**", ela_result)
        
        st.text_area("Extracted Text (for review)", text, height=150)
        
        for key, val in results.items():
            st.write(f"**{key}:**")
            if isinstance(val, list):
                for item in val:
                    st.write("-", item)
            else:
                st.write(val)
        
        st.metric("Risk Score", f"{score}/100")
        st.write("**Verdict:**", verdict)
        
        log_check("Screenshot", "screenshot.png", score, verdict)

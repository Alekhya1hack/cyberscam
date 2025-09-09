# appy_final.py
import os
import re
import csv
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import imagehash
import streamlit as st
from datetime import datetime

# try to import Levenshtein, fallback to difflib
try:
    import Levenshtein
    def ratio_func(a, b): return Levenshtein.ratio(a, b)
except Exception:
    import difflib
    def ratio_func(a, b): return difflib.SequenceMatcher(None, a, b).ratio()

# try pytesseract (OCR). Tesseract engine must be installed separately.
try:
    import pytesseract
    def tesseract_available_check():
        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    TESSERACT_AVAILABLE = tesseract_available_check()
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# ---------------- CONFIG ----------------
BLACKLIST_FILE = "blacklist.txt"
CHECKS_LOG = "scam_checks_log.csv"
TEMPLATES_DIR = "templates"                # for QR background checks (folders: templates/genuine, templates/fake)
REFERENCE_DIR = "reference_screenshots"    # for screenshot phash similarity

KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

# ---------------- HELPERS: storage / logging ----------------
def load_blacklist():
    if not os.path.exists(BLACKLIST_FILE):
        return []
    with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

def save_blacklist(lst):
    with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
        for item in sorted(set(lst)):
            f.write(item + "\n")

def log_check(check_type, details, score, verdict):
    # details must be a string
    details_str = details if isinstance(details, str) else str(details)
    header = ["Timestamp", "Type", "Details", "Score", "Verdict"]
    new_row = [datetime.now().isoformat(), check_type, details_str, score, verdict]
    exists = os.path.exists(CHECKS_LOG)
    with open(CHECKS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(new_row)

# ---------------- UPI CHECKER ----------------
BLACKLIST = load_blacklist()

def is_valid_upi_format(upi_id: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9.\-_]+@[a-zA-Z]+$", upi_id))

def check_upi_id_score(upi_id: str):
    """Return (score:int 0-100, verdict:str, reasons:list[str])"""
    u = upi_id.lower().strip()
    score = 0
    reasons = []

    if not is_valid_upi_format(u):
        score += 40
        reasons.append("Invalid UPI format")

    if u in BLACKLIST:
        score += 80
        reasons.append("UPI is in blacklist")

    for kw in KEYWORDS:
        if kw in u:
            score += 50
            reasons.append(f"Contains suspicious keyword '{kw}'")

    # similarity check with blacklist (if any)
    for b in BLACKLIST:
        r = ratio_func(u, b)
        if r >= 0.85:
            score += 60
            reasons.append(f"Very similar to blacklisted '{b}' (ratio={r:.2f})")
            break

    score = min(100, score)
    if score < 30:
        verdict = "Genuine"
    elif score < 70:
        verdict = "Suspicious"
    else:
        verdict = "Fake"
    return score, verdict, reasons

# ---------------- QR BACKGROUND CHECK (template-based) ----------------
def _orb_matches(img1_gray, img2_gray):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except:
        return 0
    return len(matches)

def _phash_similarity_pil(pil1, pil2):
    h1 = imagehash.phash(pil1)
    h2 = imagehash.phash(pil2)
    # normalized similarity in [0,1]
    diff = abs(h1 - h2)
    max_bits = len(h1.hash) ** 2
    sim = 1 - (diff / max_bits)
    return sim

def check_qr_background(uploaded_cv_img, templates_path=TEMPLATES_DIR):
    """
    Compare uploaded image to templates in templates/genuine and templates/fake.
    Returns (best_label:str in {'genuine','fake','unknown'}, best_score:float)
    """
    uploaded_gray = cv2.cvtColor(uploaded_cv_img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_label = "unknown"

    for label in ("genuine","fake"):
        folder = os.path.join(templates_path, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                tpl = cv2.imread(fpath)
                if tpl is None:
                    continue
                tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                orb_score = _orb_matches(uploaded_gray, tpl_gray)
                # phash using PIL conversions
                pil_up = Image.fromarray(cv2.cvtColor(uploaded_cv_img, cv2.COLOR_BGR2RGB))
                pil_tpl = Image.fromarray(cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB))
                phash_sim = _phash_similarity_pil(pil_up, pil_tpl)
                combined = orb_score * 0.7 + phash_sim * 0.3
                if combined > best_score:
                    best_score = combined
                    best_label = label
            except Exception:
                continue

    if best_score < 0:
        return "unknown", 0.0
    return best_label, float(best_score)

# ---------------- SCREENSHOT FORENSICS ----------------
def ela_image_and_score(pil_img: Image.Image, quality=90):
    # produce ELA image and a numeric max_diff
    tmp = "temp_ela.jpg"
    pil_img.save(tmp, "JPEG", quality=quality)
    recompressed = Image.open(tmp).convert("RGB")
    ela = ImageChops.difference(pil_img, recompressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_vis = ImageEnhance.Brightness(ela).enhance(scale)
    return ela_vis, max_diff

def decode_qr_opencv(pil_img: Image.Image):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, bbox, rectified = detector.detectAndDecode(cv_img)
    return data, bbox

def phash_compare_with_reference(pil_img: Image.Image, ref_folder=REFERENCE_DIR, threshold=5):
    """Return list of matching filenames (phash diff < threshold)"""
    if not os.path.isdir(ref_folder):
        return []
    img_hash = imagehash.phash(pil_img)
    matches = []
    for fn in os.listdir(ref_folder):
        fpath = os.path.join(ref_folder, fn)
        try:
            ref = Image.open(fpath).convert("RGB")
            diff = abs(img_hash - imagehash.phash(ref))
            if diff < threshold:
                matches.append(fn)
        except Exception:
            continue
    return matches

def ocr_safe_extract(pil_img: Image.Image):
    if not TESSERACT_AVAILABLE:
        return ""  # OCR not available
    # perform OCR on grayscale numpy array
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    try:
        text = pytesseract.image_to_string(gray)
    except Exception:
        text = ""
    return text

def analyze_screenshot_full(pil_img: Image.Image):
    """
    Returns:
      ela_vis (PIL), max_diff (int),
      ocr_text (str),
      qr_result (str),
      phash_matches (list),
      metadata_info (str),
      final_score (0-100 int),
      verdict (str)
    """
    results = {}
    # ELA
    ela_vis, max_diff = ela_image_and_score(pil_img)
    results['ela_max_diff'] = max_diff

    # OCR
    ocr_text = ocr_safe_extract(pil_img)
    results['ocr_text'] = ocr_text

    # QR presence and content
    qr_data, qr_bbox = decode_qr_opencv(pil_img)
    if qr_bbox is None:
        qr_msg = "No QR detected"
    else:
        qr_msg = f"QR detected: {qr_data[:200] if qr_data else '(found, unreadable)'}"
    results['qr_msg'] = qr_msg

    # phash similarity against reference screenshots
    phash_matches = phash_compare_with_reference(pil_img)
    results['phash_matches'] = phash_matches

    # metadata / exif
    try:
        exif = pil_img.getexif()
        if exif and len(exif) > 0:
            meta_msg = f"EXIF tags: {len(exif)}"
        else:
            meta_msg = "No EXIF metadata"
    except Exception:
        meta_msg = "Could not read EXIF"
    results['meta'] = meta_msg

    # risk scoring heuristics (tuneable)
    score = 0
    # ELA threshold
    if max_diff > 50:
        score += 30
    elif max_diff > 20:
        score += 10

    # OCR: presence of payment-like keywords reduces suspicion; absence increases
    text_lower = ocr_text.lower()
    pay_keywords = ["upi", "paid", "success", "credited", "transaction", "txn", "amount", "ref"]
    found_pay_kw = sum(1 for k in pay_keywords if k in text_lower)
    if found_pay_kw >= 2:
        score += 0
    elif found_pay_kw == 1:
        score += 10
    else:
        score += 25

    # QR missing increases suspicion
    if qr_bbox is None:
        score += 25

    # phash matches with known scam refs adds risk
    if phash_matches:
        score += 30

    # missing EXIF adds minor risk
    if results['meta'] == "No EXIF metadata":
        score += 5

    score = min(100, score)
    if score < 30:
        verdict = "Genuine"
    elif score < 70:
        verdict = "Suspicious"
    else:
        verdict = "Fake"

    return ela_vis, max_diff, ocr_text, qr_msg, phash_matches, results['meta'], score, verdict

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Scam Detector Suite", layout="wide")
st.title("🛡 Scam Detector Suite — UPI / QR / Screenshot Forensics")

menu = st.sidebar.selectbox("Choose module", ["UPI Checker", "QR Background Check", "Screenshot Forensics", "Admin Panel"])

# ----- UPI Checker -----
if menu == "UPI Checker":
    st.header("🔐 UPI ID Checker")
    st.write("Check format, blacklist and similarity for a UPI ID.")
    upi = st.text_input("Enter UPI ID (example: name@bank)", key="upi_input")

    if st.button("Check UPI", key="btn_check_upi"):
        if not upi:
            st.warning("Enter a UPI ID first.")
        else:
            score, verdict, reasons = check_upi_id_score(upi)
            # show progress and metric
            st.progress(score)
            st.metric("Risk Score", f"{score}/100")
            st.write("**Verdict:**", verdict)
            if reasons:
                st.write("**Reasons:**")
                for r in reasons:
                    st.write("-", r)
            else:
                st.write("No major issues found.")

            # log (store plain strings without emojis)
            log_check("UPI", upi, score, verdict)

    st.markdown("---")
    st.write("You can also upload a text file containing UPI IDs (one per line) and add them to blacklist via Admin Panel.")

# ----- QR Background Check -----
elif menu == "QR Background Check":
    st.header("🔍 QR Background Detector (template-based)")
    st.write("Compare uploaded poster against templates in `templates/genuine` and `templates/fake`.")
    st.caption("Create folders: templates/genuine and templates/fake and put sample images to improve matching.")

    uploaded = st.file_uploader("Upload poster image", type=["png","jpg","jpeg"], key="qr_bg_upload")
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        st.image(pil, caption="Uploaded Image", width="stretch")

        label, score = check_qr_background(cv_img)
        # compute a simple risk score from result
        risk = 20 if label == "genuine" else 80 if label == "fake" else 50
        st.metric("Detected Label", label)
        st.metric("Background Match Score (internal)", f"{score:.2f}")
        st.progress(risk)
        verdict = "Genuine" if label == "genuine" else "Fake" if label == "fake" else "Suspicious"
        st.write("**Verdict:**", verdict)
        log_check("QR Background", uploaded.name, int(risk), verdict)

# ----- Screenshot Forensics -----
elif menu == "Screenshot Forensics":
    st.header("🖼 Screenshot Forensics (ELA + OCR + QR + pHash)")
    st.write("Upload payment or review screenshots - the app will run multiple heuristics and give a risk score + verdict.")
    if not TESSERACT_AVAILABLE:
        st.warning("Tesseract OCR not found. OCR checks will be skipped. (Install Tesseract and set path in Admin panel if you want OCR.)")

    uploaded = st.file_uploader("Upload screenshot image", type=["png","jpg","jpeg"], key="ss_upload")
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, caption="Uploaded Screenshot", width="stretch")

        ela_vis, max_diff, ocr_text, qr_msg, phash_matches, meta_msg, score, verdict = analyze_screenshot_full(pil)

        st.subheader("Results")
        st.write("**Verdict:**", verdict)
        st.metric("Risk Score", f"{score}/100")
        st.write("**QR check:**", qr_msg)
        st.write("**EXIF / metadata:**", meta_msg)
        st.write("**pHash matches with references:**", phash_matches if phash_matches else "None")
        st.write("**OCR extracted (if available):**")
        if TESSERACT_AVAILABLE:
            st.text_area("OCR text", ocr_text, height=150)
        else:
            st.write("(OCR not available)")

        st.write("**ELA visualization (bright spots may indicate edits):**")
        st.image(ela_vis, caption="ELA image", width="stretch")

        log_check("Screenshot", uploaded.name, int(score), verdict)

# ----- Admin Panel -----
elif menu == "Admin Panel":
    st.header("⚙️ Admin Panel")
    st.write("Manage blacklist, download logs, set Tesseract path (Windows).")

    st.subheader("Blacklist")
    st.write("Total blacklisted:", len(BLACKLIST))
    if BLACKLIST:
        st.dataframe({"blacklist": BLACKLIST})

    # add UPI
    with st.form("form_add_upi"):
        new_upi = st.text_input("Add UPI ID to blacklist (one)", key="add_upi")
        submitted_add = st.form_submit_button("Add UPI")
        if submitted_add:
            if new_upi:
                u = new_upi.lower().strip()
                if u in BLACKLIST:
                    st.info("Already in blacklist.")
                else:
                    BLACKLIST.append(u)
                    save_blacklist(BLACKLIST)
                    st.success(f"Added {u} to blacklist.")
            else:
                st.warning("Enter a UPI ID to add.")

    # delete UPI
    with st.form("form_delete_upi"):
        del_upi = st.text_input("Delete UPI ID from blacklist (one)", key="del_upi")
        submitted_del = st.form_submit_button("Delete UPI")
        if submitted_del:
            if del_upi:
                u = del_upi.lower().strip()
                if u in BLACKLIST:
                    BLACKLIST.remove(u)
                    save_blacklist(BLACKLIST)
                    st.success(f"Removed {u} from blacklist.")
                else:
                    st.warning("UPI not found.")
            else:
                st.warning("Enter a UPI ID to delete.")

    st.markdown("---")
    st.subheader("Logs")
    if os.path.exists(CHECKS_LOG):
        with open(CHECKS_LOG, "r", encoding="utf-8") as f:
            st.download_button("Download checks log (CSV)", data=f, file_name=CHECKS_LOG)
    else:
        st.write("No logs yet.")

    st.markdown("---")
    st.subheader("Tesseract OCR (optional)")
    st.write("If OCR is required (recommended), install Tesseract separately and provide the path below.")
    tpath = st.text_input("Tesseract exe path (Windows) or leave blank if installed & in PATH", value="", key="tpath")
    if st.button("Set Tesseract Path"):
        if not tpath:
            st.warning("Enter a valid path or install Tesseract and ensure it is in PATH.")
        else:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tpath
                ok = tesseract_available_check()
                if ok:
                    TESSERACT_AVAILABLE = True
                    st.success("Tesseract path set and available.")
                else:
                    st.error("Tesseract not available at that path.")
            except Exception as e:
                st.error(f"Could not set Tesseract path: {e}")

    st.markdown("---")
    st.write("Make sure folders exist (if you want to use background / reference matching):")
    st.write(f"- QR templates folder: `{TEMPLATES_DIR}` (subfolders: `genuine`, `fake`)")
    st.write(f"- Reference screenshots folder: `{REFERENCE_DIR}`")

# End of app


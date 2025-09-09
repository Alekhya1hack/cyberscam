import streamlit as st
import re
import Levenshtein
import csv
import os
from datetime import datetime

# ---------------- CONFIG ----------------
BLACKLIST_FILE = "blacklist.txt"
LOG_FILE = "blacklist_log.csv"

# Suspicious keywords
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

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
    """Logs every addition with timestamp and source"""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "UPI_ID", "Added_By"])
        writer.writerow([datetime.now().isoformat(), upi_id, added_by])

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

# ---------------- LOAD DATA ----------------
BLACKLIST = load_blacklist()

# ---------------- STREAMLIT APP ----------------
st.title("🔐 UPI Scam Detector")
st.write("Check if a UPI ID looks suspicious, blacklisted, or safe")

# --- Check UPI Section ---
upi_input = st.text_input("Enter a UPI ID to check:")

if st.button("Check"):
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

# --- Add to Blacklist Section ---
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

# --- Show Blacklist Section ---
st.subheader("📄 Current Blacklist")
if BLACKLIST:
    st.write(f"Total blacklisted UPI IDs: {len(BLACKLIST)}")
    st.dataframe({"Blacklisted UPI IDs": BLACKLIST})
else:
    st.write("✅ No blacklisted UPI IDs found.")

# --- Show Log Section ---
if os.path.exists(LOG_FILE):
    st.subheader("🕒 Blacklist Addition Log")
    with open(LOG_FILE, "r") as csvfile:
        st.download_button("📥 Download Log", csvfile, file_name="blacklist_log.csv")
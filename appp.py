import streamlit as st
import re
import Levenshtein

# Suspicious keywords
KEYWORDS = ["pmcare", "pmcares", "lottery", "winmoney", "prize", "donation"]

# Load blacklist
def load_blacklist(filepath="blacklist.txt"):
    try:
        with open(filepath, "r") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        return []

BLACKLIST = load_blacklist()

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

def check_upi_id(upi_id: str) -> str:
    upi_id = upi_id.lower().strip()

    # 1. Format check
    if not is_valid_format(upi_id):
        return f"❌ Invalid UPI format: {upi_id}"

    # 2. Direct blacklist check
    if upi_id in BLACKLIST:
        return f"⚠️ Blacklisted UPI ID: {upi_id}"

    # 3. Keyword-based detection
    for kw in KEYWORDS:
        if kw in upi_id:
            return f"⚠️ Suspicious keyword '{kw}' found in {upi_id}"

    # 4. Similarity check
    similar, score = check_similarity(upi_id)
    if similar:
        return f"⚠️ UPI ID '{upi_id}' looks similar to blacklisted '{similar}' (score: {score:.2f})"

    return f"✅ Safe UPI ID: {upi_id}"


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="UPI Scam Detector", page_icon="🔒")

st.title("🔒 UPI Scam Detector")
st.write("Check if a UPI ID looks suspicious or safe")

upi = st.text_input("Enter a UPI ID:")

if st.button("Check Now"):
    if upi.strip():   # ✅ check after input
        result = check_upi_id(upi.strip())

        if "✅" in result:
            st.success(result)      # Green box
        elif "⚠️" in result:
            st.warning(result)      # Yellow box
        else:
            st.error(result)        # Red box
    else:
        st.warning("⚠️ Please enter a UPI ID to check.")

"""
Prescription Reader
--------------------
1. User uploads a photo of a handwritten/printed prescription.
2. Gemini 2.5 Flash reads the image and extracts medicine names (+ dosage/frequency if visible).
3. Each extracted name is fuzzy-matched against medicine_dataset.csv using fuzzywuzzy.
4. Full details (composition, price, manufacturer, uses, side effects, substitutes/interactions)
   are displayed for every matched medicine.

Run with:
    streamlit run app.py
"""

import json
import os
import re

import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz, process
from PIL import Image

import google.generativeai as genai

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Prescription Reader", page_icon="💊", layout="wide")

st.markdown(
    """
    <style>
    .med-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .med-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .med-sub {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 Prescription Reader")
st.caption("Upload a prescription photo → Gemini extracts the medicines → fuzzy-matched against your dataset.")

# --------------------------------------------------------------------------------------
# Sidebar - API key, dataset, settings
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here",
        help="Get a free key from https://aistudio.google.com/apikey",
    )

    st.divider()

    st.subheader("🎯 Matching")
    match_threshold = st.slider("Fuzzy match confidence threshold", 40, 100, 60, 1)
    st.caption("Matches against both product name and composition — lower this if genuine matches show as 'Low confidence'.")

    st.divider()
    st.caption("Your API key is only used in this session and is never stored or logged.")

# --------------------------------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------------------------------
DATASET_PATH = "medicine_data.csv"  # <-- keep this file in the same folder as app.py


@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


df = None
if not os.path.exists(DATASET_PATH):
    st.error(
        f"Could not find `{DATASET_PATH}` next to app.py. "
        f"Place your medicine dataset CSV in the same folder and name it `{DATASET_PATH}` "
        f"(or edit DATASET_PATH at the top of the script)."
    )
else:
    try:
        df = load_dataset(DATASET_PATH)
    except Exception as e:
        st.error(f"Could not read `{DATASET_PATH}`: {e}")

if df is not None:
    with st.sidebar:
        st.divider()
        st.subheader("🧩 Column mapping")
        st.caption("Tell the app which column in your CSV holds each field.")

        cols = list(df.columns)

        def guess(*keywords):
            for c in cols:
                lc = c.lower()
                if any(k in lc for k in keywords):
                    return c
            return cols[0]

        col_name = st.selectbox("Medicine name column", cols, index=cols.index(guess("name", "medicine")))
        col_composition = st.selectbox(
            "Composition / salt column", cols, index=cols.index(guess("composition", "salt", "insulin"))
        )
        col_price = st.selectbox("Price column", cols, index=cols.index(guess("price", "₹", "cost")))
        col_manufacturer = st.selectbox(
            "Manufacturer column", cols, index=cols.index(guess("manufactur", "company"))
        )
        col_description = st.selectbox(
            "Uses / description column", cols, index=cols.index(guess("description", "use", "about"))
        )
        col_side_effects = st.selectbox(
            "Side effects column", cols, index=cols.index(guess("side_effect", "side effect"))
        )
        col_substitutes = st.selectbox(
            "Substitutes / interactions column (JSON)",
            ["(none)"] + cols,
            index=(["(none)"] + cols).index(guess("substitute", "interact", "drug")) if any(
                k in " ".join(cols).lower() for k in ["substitute", "interact"]
            ) else 0,
        )

# --------------------------------------------------------------------------------------
# Gemini extraction
# --------------------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are an expert pharmacist reading a medical prescription image.

Carefully read the prescription (it may be handwritten, so use medical context to make your
best guess for unclear words) and extract every medicine mentioned.

Return ONLY a valid JSON array, no markdown fences, no extra commentary, in this exact shape:

[
  {
    "medicine_name": "string - the medicine name as written/best guess",
    "dosage": "string - strength/dose if visible, else empty string",
    "frequency": "string - how often to take it if visible, else empty string",
    "duration": "string - how long to take it if visible, else empty string",
    "notes": "string - any other instruction like 'after food', else empty string"
  }
]

If no medicines can be identified, return an empty array [].
"""


def extract_medicines_from_image(image: Image.Image, key: str):
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([EXTRACTION_PROMPT, image])
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    return json.loads(text)


# --------------------------------------------------------------------------------------
# Fuzzy matching
# --------------------------------------------------------------------------------------
def find_best_match(query, df, col_name, col_composition):
    """Match query against both the product-name column and the composition column,
    since prescriptions often use the generic/salt name (e.g. 'Paracetamol') rather
    than the brand name (e.g. 'Dolo 650')."""
    name_choices = df[col_name].astype(str).tolist()
    best_name, best_name_score = process.extractOne(query, name_choices, scorer=fuzz.token_set_ratio)

    best_comp_score = -1
    best_comp_idx = None
    if col_composition in df.columns:
        comp_choices = df[col_composition].astype(str).tolist()
        best_comp, best_comp_score = process.extractOne(query, comp_choices, scorer=fuzz.token_set_ratio)
        if best_comp_score >= 0:
            best_comp_idx = df.index[df[col_composition].astype(str) == best_comp][0]

    if best_comp_score > best_name_score:
        row = df.loc[best_comp_idx]
        return row, best_comp_score, "composition"
    else:
        row = df[df[col_name].astype(str) == best_name].iloc[0]
        return row, best_name_score, "name"


def parse_substitutes(raw):
    if raw is None or (isinstance(raw, float)) or str(raw).strip() in ("", "nan", "(none)"):
        return None
    try:
        data = json.loads(raw)
        return data
    except Exception:
        try:
            data = json.loads(raw.replace("'", '"'))
            return data
        except Exception:
            return None


def score_color(score):
    if score >= 85:
        return "#1e8e3e"
    if score >= 70:
        return "#e8710a"
    return "#d93025"


# --------------------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------------------
uploaded_image = st.file_uploader(
    "📷 Upload a prescription image", type=["png", "jpg", "jpeg", "webp"]
)

col_a, col_b = st.columns([1, 1])
if uploaded_image is not None:
    with col_a:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded prescription", use_container_width=True)

analyze = st.button("🔍 Extract & Match Medicines", type="primary", use_container_width=False)

if analyze:
    if not api_key:
        st.error("Please paste your Gemini API key in the sidebar first.")
    elif uploaded_image is None:
        st.error("Please upload a prescription image first.")
    elif df is None:
        st.error(f"`{DATASET_PATH}` is missing or unreadable. Check the sidebar error above.")
    else:
        with st.spinner("Reading prescription with Gemini 2.5 Flash..."):
            try:
                image = Image.open(uploaded_image)
                extracted = extract_medicines_from_image(image, api_key)
            except json.JSONDecodeError:
                st.error("Gemini did not return valid JSON. Try a clearer image, or try again.")
                extracted = []
            except Exception as e:
                st.error(f"Gemini extraction failed: {e}")
                extracted = []

        if not extracted:
            st.warning("No medicines could be identified in this image.")
        else:
            st.success(f"Found {len(extracted)} medicine(s) in the prescription.")

            results = []
            for item in extracted:
                qname = item.get("medicine_name", "").strip()
                if not qname:
                    continue
                row, score, matched_on = find_best_match(qname, df, col_name, col_composition)
                results.append({"item": item, "qname": qname, "row": row, "score": score, "matched_on": matched_on})

            # ---------------- Summary table (always shown) ----------------
            st.markdown("### 📊 Summary")
            summary_df = pd.DataFrame(
                [
                    {
                        "Extracted (Rx)": r["qname"],
                        "Dosage": r["item"].get("dosage", ""),
                        "Frequency": r["item"].get("frequency", ""),
                        "Duration": r["item"].get("duration", ""),
                        "Matched Product": r["row"].get(col_name, ""),
                        "Composition": r["row"].get(col_composition, ""),
                        "Manufacturer": r["row"].get(col_manufacturer, ""),
                        "Price": r["row"].get(col_price, ""),
                        "Match %": r["score"],
                        "Confidence": "✅ High" if r["score"] >= match_threshold else "⚠️ Low - verify",
                    }
                    for r in results
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ---------------- Detail cards ----------------
            st.markdown("### 🗂️ Full Details")
            for r in results:
                item, qname, row, score = r["item"], r["qname"], r["row"], r["score"]

                st.markdown("---")
                st.markdown(f"#### 🩺 Extracted: *{qname}*")
                meta_bits = []
                if item.get("dosage"):
                    meta_bits.append(f"**Dosage:** {item['dosage']}")
                if item.get("frequency"):
                    meta_bits.append(f"**Frequency:** {item['frequency']}")
                if item.get("duration"):
                    meta_bits.append(f"**Duration:** {item['duration']}")
                if item.get("notes"):
                    meta_bits.append(f"**Notes:** {item['notes']}")
                if meta_bits:
                    st.markdown(" &nbsp;|&nbsp; ".join(meta_bits))

                if score < match_threshold:
                    st.warning(
                        f"Low-confidence match ({score}%) — showing closest dataset entry below, please verify."
                    )

                st.markdown(
                    f"""
                    <div class="med-card">
                        <div class="med-title">{row.get(col_name, '')}
                            <span class="score-badge" style="background-color:{score_color(score)}">
                                {score}% match ({r['matched_on']})
                            </span>
                        </div>
                        <div class="med-sub">{row.get(col_composition, '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**💰 Price:** {row.get(col_price, 'N/A')}")
                with c2:
                    st.markdown(f"**🏭 Manufacturer:** {row.get(col_manufacturer, 'N/A')}")

                with st.expander("📋 Uses / Description", expanded=True):
                    st.write(row.get(col_description, "No description available."))

                with st.expander("⚠️ Side Effects"):
                    se = row.get(col_side_effects, "")
                    if pd.isna(se) or not str(se).strip():
                        st.write("No side effect data available.")
                    else:
                        for s in str(se).split(","):
                            if s.strip():
                                st.markdown(f"- {s.strip()}")

                if col_substitutes != "(none)":
                    sub_data = parse_substitutes(row.get(col_substitutes))
                    if sub_data:
                        with st.expander("🔄 Substitutes / Interacting Drugs"):
                            try:
                                sub_df = pd.DataFrame(sub_data)
                                st.dataframe(sub_df, use_container_width=True, hide_index=True)
                            except Exception:
                                st.json(sub_data)

st.divider()
with st.expander("ℹ️ How to use this app"):
    st.markdown(
        f"""
        1. Paste your **Gemini API key** in the sidebar (get one free at
           [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
        2. Make sure **`{DATASET_PATH}`** is in the same folder as `app.py`
           (check the column mapping in the sidebar).
        3. Upload a **photo of a prescription**.
        4. Click **Extract & Match Medicines**.

        The app never stores your API key or images — everything runs in your local
        Streamlit session.
        """
    )
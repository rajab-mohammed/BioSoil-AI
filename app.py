import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# BioSoil-AI
# AI-Assisted Soil Diagnosis and Bacterial Recommendation
# =========================================================

st.set_page_config(
    page_title="BioSoil-AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 BioSoil-AI")
st.subheader("AI-Assisted Soil Diagnosis and Bacterial Bioremediation Recommendation System")


# =========================================================
# 1. LOAD KNOWLEDGE BASE
# =========================================================

@st.cache_data
def load_knowledge_base():

    return pd.read_excel(
        "BioSoil-AI_Knowledge_Base_FINAL_Evidence_Updated.xlsx",
        sheet_name="Bacteria_KB"
    )


try:
    kb = load_knowledge_base()

except Exception as e:

    st.error(
        "Could not load the knowledge base. "
        "Make sure BioSoil-AI_Knowledge_Base_FINAL_Evidence_Updated.xlsx "
        "is in the same folder/repository as app.py."
    )

    st.stop()


# =========================================================
# 2. NORMALIZATION
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


# =========================================================
# 3. SALINITY CLASSIFICATION
#    ECe = saturated soil-paste extract
# =========================================================

def classify_salinity(ec):

    if ec < 2:
        return "Non-saline"

    elif ec < 4:
        return "Slightly saline"

    elif ec < 8:
        return "Moderately saline"

    elif ec < 16:
        return "Strongly saline"

    else:
        return "Very strongly saline"


# =========================================================
# 4. SOIL DIAGNOSIS
# =========================================================

def diagnose_soil(sample):

    problems = []

    # pH
    if sample["pH"] < 6.0:
        problems.append("Acidic pH")

    elif sample["pH"] > 7.8:
        problems.append("Alkaline pH")

    # ECe
    salinity_class = classify_salinity(sample["EC"])

    if sample["EC"] >= 2:
        problems.append("High salinity")

    # Organic matter
    if sample["Organic_Matter"] < 1.5:
        problems.append("Low organic matter")

    # Carbon
    if sample["Carbon"] < 1.0:
        problems.append("Low carbon")

    # Nutrients
    if sample["N"] < 250:
        problems.append("Low N")

    if sample["P"] < 10:
        problems.append("Low P")

    if sample["K"] < 100:
        problems.append("Low K")

    if not problems:
        problems.append("No major limitation")

    return problems, salinity_class


# =========================================================
# 5. REQUIRED BACTERIAL TRAITS
# =========================================================

def get_required_traits(problems):

    traits = []

    if "Low N" in problems:
        traits.append("N Fixation")

    if "Low P" in problems:
        traits.append("P Solubilization")

    if "Low K" in problems:
        traits.append("K Mobilization")

    if "High salinity" in problems:
        traits.append("Salt Tolerance")

    if "Low organic matter" in problems or "Low carbon" in problems:
        traits.append("OM Decomposition")

    return traits


# =========================================================
# 6. EVIDENCE LEVEL
#
# Knowledge-base terminology:
# Strong
# Moderate
# Limited
# No
# =========================================================

def evidence_level(value):

    value = normalize_text(value)

    if value == "":
        return "no"

    if value == "no":
        return "no"

    if "strong" in value:
        return "strong"

    if "moderate" in value:
        return "moderate"

    if "limited" in value:
        return "limited"

    return "limited"


# =========================================================
# 7. TRAIT SCORE
# =========================================================

def trait_score(value, trait_name):

    level = evidence_level(value)

    if trait_name in ["N Fixation", "P Solubilization"]:

        if level == "strong":
            return 4

        elif level == "moderate":
            return 2.5

        elif level == "limited":
            return 1

        return 0


    if trait_name == "K Mobilization":

        if level == "strong":
            return 3

        elif level == "moderate":
            return 2

        elif level == "limited":
            return 1

        return 0


    if trait_name == "OM Decomposition":

        if level == "strong":
            return 3

        elif level == "moderate":
            return 2

        elif level == "limited":
            return 1

        return 0


    return 0


# =========================================================
# 8. SALT TOLERANCE SCORE
#
# The knowledge base uses:
# High / Moderate / Limited / No
#
# Salinity classification uses ECe.
# =========================================================

def salt_tolerance_score(level, salinity_class):

    level = evidence_level(level)

    if salinity_class == "Non-saline":
        return 0, "salinity is not a major limitation"


    if salinity_class == "Slightly saline":

        if level == "strong":
            return 3, "strong salt-tolerance evidence"

        if level == "moderate":
            return 2, "moderate salt-tolerance evidence"

        if level == "limited":
            return 1, "limited salt-tolerance evidence"

        return 0, "no documented salt-tolerance support"


    if salinity_class == "Moderately saline":

        if level == "strong":
            return 4, "strong salt-tolerance evidence"

        if level == "moderate":
            return 3, "moderate salt-tolerance evidence"

        if level == "limited":
            return 1, "limited salt-tolerance evidence"

        return 0, "no documented salt-tolerance support"


    if salinity_class == "Strongly saline":

        if level == "strong":
            return 5, "strong salt-tolerance evidence"

        if level == "moderate":
            return 3, "moderate salt-tolerance evidence"

        if level == "limited":
            return 1, "limited salt-tolerance evidence"

        return 0, "no documented salt-tolerance support"


    if salinity_class == "Very strongly saline":

        if level == "strong":
            return 5, "strong salt-tolerance evidence"

        if level == "moderate":
            return 2, "moderate salt-tolerance evidence may be insufficient for very strong salinity"

        if level == "limited":
            return 0, "limited salt-tolerance evidence"

        return 0, "no documented salt-tolerance support"


    return 0, "salt-tolerance evidence not available"


# =========================================================
# 9. pH COMPATIBILITY
# =========================================================

def pH_score(row, soil_pH):

    try:

        pH_min = float(row["pH_min"])
        pH_max = float(row["pH_max"])

    except (ValueError, TypeError, KeyError):

        return 0, "documented pH range unavailable"


    if pH_min <= soil_pH <= pH_max:

        return 2, "soil pH is within the documented pH range"

    return -1, "soil pH is outside the documented pH range"


# =========================================================
# 10. BIOLOGICAL SCORE
# =========================================================

def calculate_biological_score(
    row,
    problems,
    salinity_class,
    soil_pH
):

    score = 0
    reasons = []
    covered_problems = []


    # -----------------------------------------------------
    # Low N
    # -----------------------------------------------------

    if "Low N" in problems:

        points = trait_score(
            row.get("N_Fixation", ""),
            "N Fixation"
        )

        if points > 0:

            score += points
            covered_problems.append("Low N")

            reasons.append(
                f"supports biological nitrogen fixation (+{points:g})"
            )

        else:

            reasons.append(
                "does not provide documented nitrogen-fixation support"
            )


    # -----------------------------------------------------
    # Low P
    # -----------------------------------------------------

    if "Low P" in problems:

        points = trait_score(
            row.get("P_Solubilization", ""),
            "P Solubilization"
        )

        if points > 0:

            score += points
            covered_problems.append("Low P")

            reasons.append(
                f"supports phosphate solubilization (+{points:g})"
            )

        else:

            reasons.append(
                "does not provide documented phosphate-solubilization support"
            )


    # -----------------------------------------------------
    # Low K
    # -----------------------------------------------------

    if "Low K" in problems:

        points = trait_score(
            row.get("K_Mobilization", ""),
            "K Mobilization"
        )

        if points > 0:

            score += points
            covered_problems.append("Low K")

            reasons.append(
                f"supports potassium mobilization (+{points:g})"
            )

        else:

            reasons.append(
                "does not provide documented potassium-mobilization support"
            )


    # -----------------------------------------------------
    # Salinity
    # -----------------------------------------------------

    if "High salinity" in problems:

        points, salt_reason = salt_tolerance_score(
            row.get("Salt_Tolerance_Level", row.get("Salt_Tolerance", "")),
            salinity_class
        )

        score += points

        reasons.append(
            f"{salt_reason} (+{points})"
        )

        if points > 0:
            covered_problems.append("High salinity")


    # -----------------------------------------------------
    # Organic matter / carbon
    # -----------------------------------------------------

    if (
        "Low organic matter" in problems
        or "Low carbon" in problems
    ):

        function = normalize_text(
            row.get("Main_Function", "")
        )

        target = normalize_text(
            row.get("Target_Problem", "")
        )

        if (
            "decomposition" in function
            or "organic matter" in function
            or "low om" in target
            or "low organic" in target
        ):

            score += 3

            covered_problems.append(
                "Low organic matter"
            )

            reasons.append(
                "supports organic-matter decomposition (+3)"
            )


    # -----------------------------------------------------
    # pH
    # -----------------------------------------------------

    pH_points, pH_reason = pH_score(
        row,
        soil_pH
    )

    score += pH_points

    reasons.append(
        f"{pH_reason} ({'+' if pH_points >= 0 else ''}{pH_points})"
    )


    # -----------------------------------------------------
    # PGPR
    # -----------------------------------------------------

    pgpr_level = evidence_level(
        row.get("PGPR", "")
    )

    if pgpr_level == "strong":

        score += 1.5

        reasons.append(
            "has strong plant-growth-promoting activity (+1.5)"
        )

    elif pgpr_level == "moderate":

        score += 1

        reasons.append(
            "has moderate plant-growth-promoting activity (+1)"
        )


    covered_problems = list(
        dict.fromkeys(covered_problems)
    )

    if covered_problems:

        reasons.append(
            f"covers {len(covered_problems)} "
            f"of {count_actionable_problems(problems)} "
            f"detected soil problems"
        )

    return score, covered_problems, "; ".join(reasons)


def count_actionable_problems(problems):

    excluded = {
        "No major limitation",
        "Acidic pH",
        "Alkaline pH"
    }

    return len([
        p for p in problems
        if p not in excluded
    ])


# =========================================================
# 11. AI SIMILARITY
#
# AI_Profile is used only as a secondary text-similarity
# component. It is NOT treated as scientific evidence.
# =========================================================

def calculate_ai_similarity(
    kb,
    problems,
    required_traits
):

    kb = kb.copy()

    if "AI_Profile" not in kb.columns:

        # Compatible fallback for the compact KB.
        kb["AI_Profile"] = (
            kb["Bacteria"].fillna("").astype(str)
            + " "
            + kb["Target_Problem"].fillna("").astype(str)
            + " "
            + kb["Main_Function"].fillna("").astype(str)
            + " "
            + kb["N_Fixation"].fillna("").astype(str)
            + " "
            + kb["P_Solubilization"].fillna("").astype(str)
            + " "
            + kb["K_Mobilization"].fillna("").astype(str)
            + " "
            + kb["Salt_Tolerance"].fillna("").astype(str)
            + " "
            + kb["PGPR"].fillna("").astype(str)
        )


    query = " ".join(
        problems + required_traits
    )


    texts = (
        [query]
        + kb["AI_Profile"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()


    kb["AI_Similarity"] = similarities

    return kb


# =========================================================
# 12. SUPPORT ACTIONS
# =========================================================

def support_actions(problems):

    actions = []

    if "High salinity" in problems:

        actions.append(
            "Consider appropriate salinity-management practices "
            "such as drainage and irrigation management."
        )


    if "Alkaline pH" in problems:

        actions.append(
            "Evaluate appropriate management practices "
            "for alkaline soil conditions."
        )


    if "Low organic matter" in problems:

        actions.append(
            "Consider suitable organic-matter management "
            "practices such as compost or organic residues."
        )


    if "Low N" in problems:

        actions.append(
            "Evaluate nitrogen-management practices in addition "
            "to biological nitrogen-fixation candidates."
        )


    if "Low P" in problems:

        actions.append(
            "Evaluate phosphorus-management practices and "
            "phosphate-solubilizing bacterial candidates."
        )


    if "Low K" in problems:

        actions.append(
            "Evaluate potassium-management practices and "
            "potassium-mobilizing candidates."
        )


    return actions


# =========================================================
# 13. SOIL INPUTS
# =========================================================

st.header("🧪 Soil Sample Data")

col1, col2, col3 = st.columns(3)


with col1:

    sample_id = st.text_input(
        "Sample ID",
        value="S01"
    )


    pH = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=8.2,
        step=0.1
    )


    EC = st.number_input(
        "ECe (dS/m)",
        min_value=0.0,
        value=20.3,
        step=0.1,
        help=(
            "Electrical Conductivity of saturated "
            "soil-paste extract (ECe)."
        )
    )


with col2:

    OM = st.number_input(
        "Organic Matter (%)",
        min_value=0.0,
        value=0.9,
        step=0.1
    )


    Carbon = OM / 1.724


    st.metric(
        "Estimated Organic Carbon (%)",
        f"{Carbon:.2f}"
    )


    N = st.number_input(
        "Total Nitrogen (mg/kg)",
        min_value=0.0,
        value=387.0,
        step=1.0
    )


with col3:

    P = st.number_input(
        "Available Phosphorus (mg/kg)",
        min_value=0.0,
        value=4.0,
        step=1.0
    )


    K = st.number_input(
        "Available Potassium (mg/kg)",
        min_value=0.0,
        value=236.0,
        step=1.0
    )


# =========================================================
# 14. ANALYSIS
# =========================================================

if st.button(
    "🔬 Analyze Soil Sample",
    type="primary"
):

    sample = {

        "Sample_ID": sample_id,

        "pH": pH,

        "EC": EC,

        "Organic_Matter": OM,

        "Carbon": Carbon,

        "N": N,

        "P": P,

        "K": K
    }


    problems, salinity_class = diagnose_soil(
        sample
    )


    required_traits = get_required_traits(
        problems
    )


    kb_ai = calculate_ai_similarity(
        kb,
        problems,
        required_traits
    )


    results = []


    for _, row in kb_ai.iterrows():

        biological_score, covered_problems, reason = (
            calculate_biological_score(
                row,
                problems,
                salinity_class,
                pH
            )
        )


        # Biological evidence is dominant.
        # AI similarity is a secondary ranking component.
        final_score = (
            biological_score
            + (row["AI_Similarity"] * 4)
        )


        results.append({

            "Bacteria":
                row["Bacteria"],

            "Final_Score":
                round(final_score, 2),

            "Biological_Score":
                round(biological_score, 2),

            "AI_Similarity":
                round(row["AI_Similarity"], 3),

            "Covered_Problems":
                ", ".join(covered_problems),

            "Salt_Tolerance_Level":
                row.get(
                    "Salt_Tolerance_Level",
                    row.get("Salt_Tolerance", "")
                ),

            "Main_Function":
                row.get("Main_Function", ""),

            "Reason":
                reason
        })


    result_df = pd.DataFrame(
        results
    ).sort_values(
        "Final_Score",
        ascending=False
    ).reset_index(drop=True)


    # =====================================================
    # SOIL DIAGNOSIS
    # =====================================================

    st.divider()

    st.subheader("🧪 Soil Diagnosis")

    st.write(
        f"**Sample:** {sample_id}"
    )

    st.write(
        f"**Salinity Class:** {salinity_class}"
    )


    for problem in problems:

        st.write(
            f"• {problem}"
        )


    # =====================================================
    # REQUIRED TRAITS
    # =====================================================

    if required_traits:

        st.subheader(
            "🦠 Required Bacterial Traits"
        )


        for trait in required_traits:

            st.write(
                f"• {trait}"
            )


    # =====================================================
    # SUPPORT ACTIONS
    # =====================================================

    actions = support_actions(
        problems
    )


    if actions:

        st.subheader(
            "🌱 General Support Actions"
        )


        for action in actions:

            st.write(
                f"• {action}"
            )


    # =====================================================
    # RANKING
    # =====================================================

    st.subheader(
        "🏆 Top Recommendations"
    )


    display_columns = [

        "Bacteria",

        "Final_Score",

        "Biological_Score",

        "AI_Similarity",

        "Covered_Problems",

        "Salt_Tolerance_Level",

        "Main_Function"
    ]


    st.dataframe(
        result_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TOP 3
    # =====================================================

    for i, row in result_df.head(3).iterrows():

        st.markdown(
            f"### {i + 1}. {row['Bacteria']}"
        )


        st.write(
            f"**Final Score:** "
            f"{row['Final_Score']}"
        )


        st.write(
            f"**Biological Score:** "
            f"{row['Biological_Score']}"
        )


        st.write(
            f"**AI Similarity:** "
            f"{row['AI_Similarity']}"
        )


        st.write(
            f"**Salt Tolerance:** "
            f"{row['Salt_Tolerance_Level']}"
        )


        st.write(
            f"**Reason:** "
            f"{row['Reason']}"
        )


    # =====================================================
    # BEST CANDIDATE
    # =====================================================

    best = result_df.iloc[0]


    st.success(
        f"Best recommended candidate: "
        f"{best['Bacteria']}"
    )


    st.write(
        "**Final Score:**",
        best["Final_Score"]
    )


    st.write(
        "**Reason:**",
        best["Reason"]
    )


    # =====================================================
    # SCIENTIFIC DISCLAIMER
    # =====================================================

    st.info(
        "This system provides a knowledge-based AI recommendation. "
        "The recommendation should be experimentally validated before "
        "field application. Bacterial strains should be confirmed for "
        "identity, safety, viability, and suitability for the target "
        "crop and soil."
    )


# =========================================================
# 15. KNOWLEDGE BASE INFORMATION
# =========================================================

with st.expander("ℹ️ About BioSoil-AI"):

    st.write(
        "The recommendation is based primarily on the curated "
        "bacterial knowledge base. Evidence-weighted biological "
        "traits form the main component of the score."
    )

    st.write(
        "TF-IDF and cosine similarity are used as a secondary "
        "AI text-similarity component through AI_Profile."
    )

    st.write(
        "Evidence_Register is maintained for scientific "
        "documentation and is not used as a scoring input."
    )

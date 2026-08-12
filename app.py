import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Soil Bioremediation System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AI-Assisted Soil Bioremediation Recommendation System")
st.caption(
    "Knowledge-based AI system for soil diagnosis and bacterial candidate ranking"
)


# =========================================================
# 2. LOAD KNOWLEDGE BASE
# =========================================================

@st.cache_data
def load_knowledge_base():
    return pd.read_excel(
        "BioSoil-AI",
        sheet_name="Bacteria_KB"
    )


try:
    kb = load_knowledge_base()
except Exception as e:
    st.error(
        "Could not load Soil_Bacteria_Knowledge_Base_Final.xlsx. "
        "Make sure the Excel file is in the same GitHub repository as app.py."
    )
    st.stop()


# =========================================================
# 3. TEXT NORMALIZATION
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
# 4. SOIL SALINITY CLASSIFICATION
#    ECe = Electrical Conductivity of saturated soil-paste extract
# =========================================================

def classify_salinity(ec):
    """
    Soil salinity classification based on ECe (dS/m).

    The classification used by the application:
        < 2       Non-saline
        2 - <4    Slightly saline
        4 - <8    Moderately saline
        8 - <16   Strongly saline
        >=16      Very strongly saline
    """

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
# 5. SOIL DIAGNOSIS
# =========================================================

def diagnose_soil(sample):

    problems = []

    # pH
    if sample["pH"] < 6.0:
        problems.append("Acidic pH")
    elif sample["pH"] > 7.8:
        problems.append("Alkaline pH")

    # Salinity
    salinity_class = classify_salinity(sample["EC"])

    if sample["EC"] >= 2:
        problems.append("High salinity")

    # Organic matter
    if sample["Organic_Matter"] < 1.5:
        problems.append("Low organic matter")

    # Organic carbon
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
# 6. REQUIRED BACTERIAL TRAITS
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
# 7. EVIDENCE LEVEL HELPERS
# =========================================================

def evidence_level(value):

    value = normalize_text(value)

    if value == "":
        return "none"

    if "very high" in value:
        return "very_high"

    if "strong" in value:
        return "strong"

    if "moderate" in value:
        return "moderate"

    if "limited" in value:
        return "limited"

    if "strain" in value:
        return "strain_dependent"

    if "group" in value:
        return "group_dependent"

    if "context" in value:
        return "context_dependent"

    if value in ["no", "none", "not established"]:
        return "none"

    return "other"


# =========================================================
# 8. SALT TOLERANCE SCORE
# =========================================================

def salt_tolerance_score(level, salinity_class):

    level = normalize_text(level)

    # No meaningful salinity pressure
    if salinity_class == "Non-saline":
        return 0, "salinity is not a major limitation"

    # Very strongly saline soil
    if salinity_class == "Very strongly saline":

        if "very high" in level:
            return 5, "very high documented salt-tolerance level"

        if "high" in level:
            return 5, "high documented salt-tolerance level"

        if "moderate" in level:
            return 2, "moderate salt-tolerance level may be insufficient for very strong salinity"

        if "limited" in level:
            return 0, "limited salt-tolerance evidence"

        return 0, "salt-tolerance evidence is not sufficiently established"

    # Strongly saline soil
    if salinity_class == "Strongly saline":

        if "very high" in level:
            return 5, "very high documented salt-tolerance level"

        if "high" in level:
            return 4, "high documented salt-tolerance level"

        if "moderate" in level:
            return 2, "moderate salt-tolerance level"

        if "limited" in level:
            return 0, "limited salt-tolerance evidence"

        return 0, "salt-tolerance evidence is not sufficiently established"

    # Moderately saline
    if salinity_class == "Moderately saline":

        if "very high" in level:
            return 5, "very high documented salt-tolerance level"

        if "high" in level:
            return 4, "high documented salt-tolerance level"

        if "moderate" in level:
            return 3, "moderate salt-tolerance level"

        if "limited" in level:
            return 1, "limited salt-tolerance evidence"

        return 0, "salt-tolerance evidence is not sufficiently established"

    # Slightly saline
    if salinity_class == "Slightly saline":

        if "very high" in level:
            return 4, "very high documented salt-tolerance level"

        if "high" in level:
            return 4, "high documented salt-tolerance level"

        if "moderate" in level:
            return 2, "moderate salt-tolerance level"

        if "limited" in level:
            return 1, "limited salt-tolerance evidence"

        return 0, "salt-tolerance evidence is not sufficiently established"

    return 0, "no salt-tolerance score"


# =========================================================
# 9. BIOLOGICAL TRAIT SCORE
# =========================================================

def trait_score(value, trait_name):

    level = evidence_level(value)

    if trait_name in ["N Fixation", "P Solubilization"]:

        if level == "strong":
            return 4

        if level == "moderate":
            return 3

        if level in ["limited", "strain_dependent"]:
            return 1

        return 0

    if trait_name == "K Mobilization":

        if level == "strong":
            return 3

        if level == "moderate":
            return 2

        if level in ["limited", "strain_dependent"]:
            return 1

        return 0

    if trait_name == "OM Decomposition":

        if level == "strong":
            return 3

        if level == "moderate":
            return 2

        if level in ["limited", "group_dependent"]:
            return 1

        return 0

    return 0


# =========================================================
# 10. BIOLOGICAL SCORE
# =========================================================

def calculate_biological_score(row, problems, salinity_class):

    score = 0
    reasons = []
    covered_problems = []

    # -----------------------------------------
    # N deficiency
    # -----------------------------------------

    if "Low N" in problems:

        value = row.get("N_Fixation", "")
        points = trait_score(value, "N Fixation")

        if points > 0:
            score += points
            covered_problems.append("Low N")
            reasons.append(
                f"supports biological nitrogen fixation (+{points})"
            )
        else:
            reasons.append(
                "does not have sufficient documented nitrogen-fixation support"
            )

    # -----------------------------------------
    # P deficiency
    # -----------------------------------------

    if "Low P" in problems:

        value = row.get("P_Solubilization", "")
        points = trait_score(value, "P Solubilization")

        if points > 0:
            score += points
            covered_problems.append("Low P")
            reasons.append(
                f"supports phosphate solubilization (+{points})"
            )
        else:
            reasons.append(
                "does not have sufficient documented phosphate-solubilization support"
            )

    # -----------------------------------------
    # K deficiency
    # -----------------------------------------

    if "Low K" in problems:

        value = row.get("K_Mobilization", "")
        points = trait_score(value, "K Mobilization")

        if points > 0:
            score += points
            covered_problems.append("Low K")
            reasons.append(
                f"supports potassium mobilization (+{points})"
            )
        else:
            reasons.append(
                "does not have sufficient documented potassium-mobilization support"
            )

    # -----------------------------------------
    # Salinity
    # -----------------------------------------

    if "High salinity" in problems:

        points, salt_reason = salt_tolerance_score(
            row.get("Salt_Tolerance_Level", ""),
            salinity_class
        )

        score += points
        reasons.append(salt_reason)

        if points > 0:
            covered_problems.append("High salinity")

    # -----------------------------------------
    # Organic matter / carbon
    # -----------------------------------------

    if "Low organic matter" in problems or "Low carbon" in problems:

        value = row.get("Main_Function", "")
        value_normalized = normalize_text(value)

        if (
            "decomposition" in value_normalized
            or "decomposer" in value_normalized
            or "organic matter" in value_normalized
        ):
            score += 3
            covered_problems.append("Low organic matter")
            covered_problems.append("Low carbon")
            reasons.append("supports organic-matter decomposition (+3)")

    # -----------------------------------------
    # pH compatibility
    # -----------------------------------------

    try:
        pH_min = float(row["pH_min"])
        pH_max = float(row["pH_max"])
        current_pH = st.session_state.get("current_pH", None)

        if current_pH is not None:

            if pH_min <= current_pH <= pH_max:
                score += 2
                reasons.append("soil pH is within the documented pH range")
            else:
                score -= 1
                reasons.append("soil pH is outside the documented pH range")

    except (ValueError, TypeError, KeyError):
        pass

    # -----------------------------------------
    # PGPR support
    # -----------------------------------------

    pgpr = evidence_level(row.get("PGPR", ""))

    if pgpr == "strong":
        score += 1.5
        reasons.append("has strong plant-growth-promoting activity")

    elif pgpr == "moderate":
        score += 1
        reasons.append("has moderate plant-growth-promoting activity")

    # -----------------------------------------
    # Number of covered problems
    # -----------------------------------------

    covered_problems = list(dict.fromkeys(covered_problems))

    if covered_problems:
        reasons.append(
            f"covers {len(covered_problems)} of {count_soil_problems(problems)} detected soil problems"
        )

    return score, covered_problems, "; ".join(reasons)


def count_soil_problems(problems):

    return len([
        p for p in problems
        if p != "No major limitation"
        and p not in ["Acidic pH", "Alkaline pH"]
    ])


# =========================================================
# 11. AI SIMILARITY
# =========================================================

def calculate_ai_similarity(kb, problems, required_traits):

    problem_text = " ".join(problems + required_traits)

    if "AI_Profile" not in kb.columns:

        kb["AI_Profile"] = (
            kb["Bacteria"].astype(str) + " " +
            kb["Target_Problem"].astype(str) + " " +
            kb["Main_Function"].astype(str) + " " +
            kb["AI_Tag"].astype(str)
        )

    texts = (
        [problem_text]
        + kb["AI_Profile"].fillna("").astype(str).tolist()
    )

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    kb = kb.copy()
    kb["AI_Similarity"] = similarities

    return kb


# =========================================================
# 12. SUPPORT ACTIONS
# =========================================================

def get_support_actions(problems):

    actions = []

    if "High salinity" in problems:
        actions.append(
            "Consider salinity-management practices such as appropriate drainage and irrigation management."
        )

    if "Alkaline pH" in problems:
        actions.append(
            "Evaluate appropriate management practices for alkaline soil conditions."
        )

    if "Low organic matter" in problems:
        actions.append(
            "Consider suitable organic-matter management practices such as compost or organic residues."
        )

    if "Low N" in problems:
        actions.append(
            "Evaluate nitrogen-management practices in addition to biological nitrogen-fixation candidates."
        )

    if "Low P" in problems:
        actions.append(
            "Evaluate phosphorus-management practices and phosphate-solubilizing bacterial candidates."
        )

    if "Low K" in problems:
        actions.append(
            "Evaluate potassium-management practices and potassium-mobilizing candidates."
        )

    return actions


# =========================================================
# 13. USER INPUT
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
        help="Electrical Conductivity of saturated soil-paste extract."
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
# 14. ANALYZE
# =========================================================

if st.button("🔬 Analyze Soil Sample", type="primary"):

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

    # Make pH available to scoring function
    st.session_state["current_pH"] = pH

    detected_problems, salinity_class = diagnose_soil(sample)

    required_traits = get_required_traits(
        detected_problems
    )

    kb_ai = calculate_ai_similarity(
        kb,
        detected_problems,
        required_traits
    )

    results = []

    for _, row in kb_ai.iterrows():

        biological_score, covered_problems, reason = (
            calculate_biological_score(
                row,
                detected_problems,
                salinity_class
            )
        )

        # AI similarity is deliberately a secondary component.
        final_score = (
            biological_score
            + (row["AI_Similarity"] * 4)
        )

        results.append({

            "Bacteria": row["Bacteria"],

            "Final_Score": round(
                final_score,
                2
            ),

            "Biological_Score": round(
                biological_score,
                2
            ),

            "AI_Similarity": round(
                row["AI_Similarity"],
                3
            ),

            "Covered_Problems": ", ".join(
                covered_problems
            ),

            "Salt_Tolerance": row[
                "Salt_Tolerance_Level"
            ],

            "Main_Function": row[
                "Main_Function"
            ],

            "Reason": reason
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "Final_Score",
        ascending=False
    ).reset_index(drop=True)


    # =====================================================
    # DISPLAY DIAGNOSIS
    # =====================================================

    st.divider()

    st.subheader("🧪 Soil Diagnosis")

    st.write(
        f"**Sample:** {sample_id}"
    )

    st.write(
        f"**Salinity Class:** {salinity_class}"
    )

    for problem in detected_problems:
        st.write(f"• {problem}")


    # =====================================================
    # REQUIRED TRAITS
    # =====================================================

    if required_traits:

        st.subheader("🦠 Required Bacterial Traits")

        for trait in required_traits:
            st.write(f"• {trait}")


    # =====================================================
    # SUPPORT ACTIONS
    # =====================================================

    support_actions = get_support_actions(
        detected_problems
    )

    if support_actions:

        st.subheader("🌱 General Support Actions")

        for action in support_actions:
            st.write(f"• {action}")


    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader("🏆 Top Recommendations")

    display_columns = [
        "Bacteria",
        "Final_Score",
        "Biological_Score",
        "AI_Similarity",
        "Covered_Problems",
        "Salt_Tolerance",
        "Main_Function"
    ]

    st.dataframe(
        result_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TOP 3 EXPLANATIONS
    # =====================================================

    for i, row in result_df.head(3).iterrows():

        st.markdown(
            f"### {i + 1}. {row['Bacteria']}"
        )

        st.write(
            f"**Final Score:** {row['Final_Score']}"
        )

        st.write(
            f"**Biological Score:** {row['Biological_Score']}"
        )

        st.write(
            f"**AI Similarity:** {row['AI_Similarity']}"
        )

        st.write(
            f"**Reason:** {row['Reason']}"
        )


    # =====================================================
    # BEST CANDIDATE
    # =====================================================

    best = result_df.iloc[0]

    st.success(
        f"Best recommended candidate: {best['Bacteria']}"
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
        "identity, safety, viability, and suitability for the target crop "
        "and soil."
    )


# =========================================================
# 15. KNOWLEDGE BASE INFORMATION
# =========================================================

with st.expander("ℹ️ About the Knowledge Base"):

    st.write(
        "The system uses a curated bacterial knowledge base containing "
        "documented biological traits, salinity-tolerance levels, pH "
        "ranges, AI tags, and functional information."
    )

    st.write(
        "AI Similarity uses TF-IDF and cosine similarity as a secondary "
        "text-similarity component. Biological evidence remains the main "
        "basis for bacterial ranking."
    )

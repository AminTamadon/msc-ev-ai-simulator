import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1. Load data
# =========================

@st.cache_data
def load_data():
    file_path = "MSC_EV_Master_AI_FINAL.xlsx"  # فایل باید کنار app.py باشد
    main_df = pd.read_excel(file_path, sheet_name="Main_Data")
    return main_df

main_df = load_data()

# =========================
# 2. Define feature columns
# =========================

feature_cols = [
    "Cell_Line",
    "Subtype",
    "MSC_Source",
    "MSC_Source_Group",
    "MSC_Preconditioning",
    "Preconditioning_Type",
    "EV_Isolation_Method",
    "Isolation_Group",
    "EV_Loading_Type",
    "Cargo_Family",
    "EV_Loading_Target",
    "Surface_Modification",
    "Engineering_Type",
    "Outcome_Type",
    "Outcome_Family",
    "EV_Dose_Unit",
    "Dose_Band",
    "RiskOfBias_Level",
    "MISEV_Compliance_Level",
    "Country",
]

feature_cols = [c for c in feature_cols if c in main_df.columns]


# =========================
# 3. Build labels
# =========================

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Therapy label
    if "Direction_num" in df.columns:
        df["Label_therapy"] = (df["Direction_num"] == 1).astype(int)
    else:
        df["Label_therapy"] = 1

    # Pro-tumor / dormancy
    if "ProTumor_num" in df.columns:
        df["Label_pro_dormant"] = (df["ProTumor_num"] > 0).astype(int)
    else:
        df["Label_pro_dormant"] = 0

    return df

main_df = build_labels(main_df)


# =========================
# 4. Train models
# =========================

from sklearn.compose import ColumnTransformer

def get_prep_step_name(pipeline):
    for name, step in pipeline.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return name
    raise KeyError("No ColumnTransformer step found.")

@st.cache_resource
def train_models(df: pd.DataFrame):
    df = df.dropna(subset=feature_cols).copy()

    # Therapy
    X_ther = df[feature_cols]
    y_ther = df["Label_therapy"].astype(int)

    preprocess_t = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
    )
    model_therapy = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    pipe_therapy = Pipeline([("prep", preprocess_t), ("model", model_therapy)])
    pipe_therapy.fit(X_ther, y_ther)

    # Pro-tumor
    X_risk = df[feature_cols]
    y_risk = df["Label_pro_dormant"].astype(int)

    preprocess_r = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
    )
    model_pro_risk = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    pipe_pro_risk = Pipeline([("prep", preprocess_r), ("model", model_pro_risk)])
    pipe_pro_risk.fit(X_risk, y_risk)

    return pipe_therapy, pipe_pro_risk

pipe_therapy, pipe_pro_risk = train_models(main_df)


# =========================
# 5. Manufacturability score
# =========================

def manufacturability_score(row):
    score = 0.0
    iso_group = str(row.get("Isolation_Group", "")).lower()
    eng_type = str(row.get("Engineering_Type", "")).lower()
    misev = str(row.get("MISEV_Compliance_Level", "")).lower()

    if any(k in iso_group for k in ["sec", "tff", "tff/aec", "uc"]):
        score += 0.5
    elif "kit" in iso_group:
        score += 0.1
    elif "secretome" in iso_group:
        score -= 0.1

    if "full" in misev:
        score += 0.2
    elif "partial" in misev:
        score += 0.1

    if eng_type in ["none", "natural"]:
        score += 0.2
    elif "surface" in eng_type:
        score += 0.1
    elif "genetic" in eng_type:
        score -= 0.1

    return score


def risk_label_color(p):
    if p < 0.02:
        return "LOW", "green"
    elif p < 0.07:
        return "MODERATE", "orange"
    else:
        return "HIGH", "red"


# =========================
# 6. Build AI candidate grid (for recommendations)
# =========================

@st.cache_data
def build_candidates(df: pd.DataFrame, n_samples=200, risk_threshold=0.02, random_state=321):
    rng = np.random.default_rng(random_state)

    design_space = {}
    for col in feature_cols:
        vals = df[col].dropna().value_counts().index[:5].tolist()
        if len(vals) == 0:
            vals = [None]
        design_space[col] = vals

    def is_valid_protocol(row):
        src = str(row["MSC_Source"]).lower()
        grp = str(row["MSC_Source_Group"]).lower()
        cl  = str(row["Cell_Line"]).lower()
        surf = str(row["Surface_Modification"]).lower()
        eng  = str(row["Engineering_Type"]).lower()

        if not any(k in cl for k in ["mda-mb", "mcf", "4t1", "t47d", "skbr"]):
            return False
        if eng in ["none", "natural"] and ("lamp2b" in surf or "crgd" in surf):
            return False
        return True

    rows = []
    attempts = 0
    max_attempts = 5000
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        row = {col: rng.choice(vals) for col, vals in design_space.items()}
        if is_valid_protocol(row):
            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    grid_df = pd.DataFrame(rows)

    # Predict P_therapy
    prep_t = get_prep_step_name(pipe_therapy)
    Xg_t = grid_df[feature_cols]
    Xg_enc_t = pipe_therapy.named_steps[prep_t].transform(Xg_t)
    if hasattr(Xg_enc_t, "toarray"):
        Xg_enc_t = Xg_enc_t.toarray().astype(float)
    grid_df["P_therapy"] = pipe_therapy.named_steps["model"].predict_proba(Xg_enc_t)[:, 1]

    # Predict P_pro_tumor
    prep_r = get_prep_step_name(pipe_pro_risk)
    Xg_r = grid_df[feature_cols]
    Xg_enc_r = pipe_pro_risk.named_steps[prep_r].transform(Xg_r)
    if hasattr(Xg_enc_r, "toarray"):
        Xg_enc_r = Xg_enc_r.toarray().astype(float)
    grid_df["P_pro_tumor"] = pipe_pro_risk.named_steps["model"].predict_proba(Xg_enc_r)[:, 1]

    # Manufacturability + Translation index
    manuf_scores = [manufacturability_score(r) for _, r in grid_df.iterrows()]
    grid_df["Manufacturability_score"] = manuf_scores
    alpha, beta, gamma = 1.0, 1.5, 1.0
    grid_df["Translation_index"] = (
        alpha * grid_df["P_therapy"]
        - beta * grid_df["P_pro_tumor"]
        + grid_df["Manufacturability_score"]
    )

    candidates = grid_df[grid_df["P_pro_tumor"] < risk_threshold].copy()
    candidates = candidates.sort_values(by="Translation_index", ascending=False)
    return candidates

candidates2 = build_candidates(main_df)


# =========================
# 7. Streamlit UI
# =========================

st.title("MSC-EV AI Simulator for Breast Cancer")

st.markdown("""
Select four key design parameters and let the AI:
1) Predict anti-tumor efficacy and dormancy/pro-tumor risk,  
2) If your design is **not recommended**, suggest the closest **AI-optimized protocol**.
""")

cell_line = st.selectbox(
    "Cell line",
    sorted(main_df["Cell_Line"].dropna().unique().tolist())
)

msc_source = st.selectbox(
    "EV / MSC source",
    sorted(main_df["MSC_Source"].dropna().unique().tolist())
)

iso_method = st.selectbox(
    "EV isolation method",
    sorted(main_df["EV_Isolation_Method"].dropna().unique().tolist())
)

dose_band = st.selectbox(
    "Dose band",
    sorted(main_df["Dose_Band"].dropna().unique().tolist())
)

run = st.button("Run AI simulation")

# defaults برای بقیه‌ی فیچرها
default_values = {}
for col in feature_cols:
    vals = main_df[col].dropna()
    default_values[col] = vals.mode().iloc[0] if len(vals) > 0 else None


if run:
    # --- user design ---
    row_user = {}
    for col in feature_cols:
        if col == "Cell_Line":
            row_user[col] = cell_line
        elif col == "MSC_Source":
            row_user[col] = msc_source
        elif col == "EV_Isolation_Method":
            row_user[col] = iso_method
        elif col == "Dose_Band":
            row_user[col] = dose_band
        else:
            row_user[col] = default_values[col]

    df_user = pd.DataFrame([row_user])

    # Predict
    try:
        prep_t = get_prep_step_name(pipe_therapy)
        Xu_t = pipe_therapy.named_steps[prep_t].transform(df_user)
        if hasattr(Xu_t, "toarray"):
            Xu_t = Xu_t.toarray().astype(float)
        p_th_user = pipe_therapy.named_steps["model"].predict_proba(Xu_t)[0][1]
    except Exception as e:
        st.error(f"Therapy model error: {e}")
        p_th_user = np.nan

    try:
        prep_r = get_prep_step_name(pipe_pro_risk)
        Xu_r = pipe_pro_risk.named_steps[prep_r].transform(df_user)
        if hasattr(Xu_r, "toarray"):
            Xu_r = Xu_r.toarray().astype(float)
        p_pro_user = pipe_pro_risk.named_steps["model"].predict_proba(Xu_r)[0][1]
    except Exception as e:
        st.error(f"Pro-tumor model error: {e}")
        p_pro_user = np.nan

    manuf_user = manufacturability_score(row_user)
    alpha, beta, gamma = 1.0, 1.5, 1.0
    TI_user = alpha * p_th_user - beta * p_pro_user + manuf_user

    risk_txt_user, risk_col_user = risk_label_color(p_pro_user)
    rec_user = (p_pro_user < 0.02) and (p_th_user > 0.85)

    st.subheader("1) Your design – AI prediction")
    st.write(f"**Cell line:** {row_user['Cell_Line']}")
    st.write(f"**EV source:** {row_user['MSC_Source']}")
    st.write(f"**Isolation method:** {row_user['EV_Isolation_Method']}")
    st.write(f"**Dose band:** {row_user['Dose_Band']}")

    st.write(f"**P(anti-tumor):** `{p_th_user:.3f}`")
    st.markdown(
        f"**P(pro-tumor/dormancy):** "
        f"<span style='color:{risk_col_user}; font-weight:bold'>{p_pro_user:.3f} ({risk_txt_user} risk)</span>",
        unsafe_allow_html=True
    )
    st.write(f"**Manufacturability score:** `{manuf_user:.2f}`")
    st.write(f"**Translation index:** `{TI_user:.3f}`")

    if rec_user:
        st.markdown("**AI Recommendation:** ✅ **RECOMMENDED for in vivo testing**", unsafe_allow_html=True)
    else:
        st.markdown("**AI Recommendation:** ❌ **NOT recommended**", unsafe_allow_html=True)

        # ---- AI suggestion ----
        st.subheader("2) AI-suggested protocol based on your choices")

        subset = candidates2.copy()

        sub1 = subset[subset["Cell_Line"] == cell_line]
        if not sub1.empty:
            subset = sub1

        sub2 = subset[subset["MSC_Source"] == msc_source]
        if not sub2.empty:
            subset = sub2

        sub3 = subset[subset["EV_Isolation_Method"] == iso_method]
        if not sub3.empty:
            subset = sub3

        sub4 = subset[subset["Dose_Band"] == dose_band]
        if not sub4.empty:
            subset = sub4

        if subset.empty:
            subset = candidates2.copy()

        best_idx = subset["Translation_index"].idxmax()
        row_ai = subset.loc[best_idx]

        df_ai = pd.DataFrame([row_ai[feature_cols]])

        try:
            prep_t2 = get_prep_step_name(pipe_therapy)
            Xa_t = pipe_therapy.named_steps[prep_t2].transform(df_ai)
            if hasattr(Xa_t, "toarray"):
                Xa_t = Xa_t.toarray().astype(float)
            p_th_ai = pipe_therapy.named_steps["model"].predict_proba(Xa_t)[0][1]
        except:
            p_th_ai = row_ai["P_therapy"]

        try:
            prep_r2 = get_prep_step_name(pipe_pro_risk)
            Xa_r = pipe_pro_risk.named_steps[prep_r2].transform(df_ai)
            if hasattr(Xa_r, "toarray"):
                Xa_r = Xa_r.toarray().astype(float)
            p_pro_ai = pipe_pro_risk.named_steps["model"].predict_proba(Xa_r)[0][1]
        except:
            p_pro_ai = row_ai["P_pro_tumor"]

        manuf_ai = manufacturability_score(row_ai)
        TI_ai = alpha * p_th_ai - beta * p_pro_ai + manuf_ai
        risk_txt_ai, risk_col_ai = risk_label_color(p_pro_ai)
        rec_ai = (p_pro_ai < 0.02) and (p_th_ai > 0.85)

        st.write(f"**Cell line:** {row_ai['Cell_Line']}")
        st.write(f"**EV source:** {row_ai['MSC_Source']}")
        st.write(f"**Isolation method:** {row_ai['EV_Isolation_Method']}")
        st.write(f"**Dose band:** {row_ai['Dose_Band']}")

        st.write(f"**P(anti-tumor):** `{p_th_ai:.3f}`")
        st.markdown(
            f"**P(pro-tumor/dormancy):** "
            f"<span style='color:{risk_col_ai}; font-weight:bold'>{p_pro_ai:.3f} ({risk_txt_ai} risk)</span>",
            unsafe_allow_html=True
        )
        st.write(f"**Manufacturability score:** `{manuf_ai:.2f}`")
        st.write(f"**Translation index:** `{TI_ai:.3f}`")

        if rec_ai:
            st.markdown("**AI Recommendation:** ✅ **RECOMMENDED for in vivo testing**", unsafe_allow_html=True)
        else:
            st.markdown("**AI Recommendation:** ❌ **NOT strongly recommended**", unsafe_allow_html=True)

        st.markdown("**AI candidate full configuration:**")
        st.dataframe(row_ai[feature_cols].to_frame(name="Value"))

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroMind AI | Alzheimer's Assistant", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. RESET LOGIC ---
def reset_app():
    st.session_state.clear()
    st.rerun()

# --- 2. DATA LOADING & MODELING ---
@st.cache_resource
def load_and_train():
    with zipfile.ZipFile('Tadpole_data.zip', 'r') as z:
        with z.open('TADPOLE_D1_D2.csv') as f:
            df = pd.read_csv(f, low_memory=False)

    # Rename & Clean
    search_map = {'ABETA': 'ABETA', 'TAU': 'TAU', 'PTAU': 'PTAU'}
    rename_dict = {}
    for std, key in search_map.items():
        matches = [c for c in df.columns if key in c]
        if matches: rename_dict[matches[0]] = std
    df = df.rename(columns=rename_dict)

    # Select Features
    features = ['ABETA', 'TAU', 'PTAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 
                'Entorhinal', 'Fusiform', 'AGE', 'PTGENDER', 'APOE4', 'DX_bl']
    
    available_feats = [c for c in features if c in df.columns]
    df = df[available_feats].copy()

    # Clean Numerics
    numeric_cols = ['ABETA', 'TAU', 'PTAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'Entorhinal', 'Fusiform', 'AGE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[><]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # Feature Engineering
    df['Neurotoxic_Ratio'] = df['PTAU'] / df['ABETA']
    df['Hippocampus_Fraction'] = df['Hippocampus'] / df['WholeBrain']

    # Target Mapping
    target_map = {'CN': 0, 'SMC': 0, 'EMCI': 1, 'LMCI': 1, 'AD': 1}
    df['Target'] = df['DX_bl'].map(target_map)
    df = df.dropna(subset=['Target'])

    # Encode Gender
    if df['PTGENDER'].dtype == 'object':
        df['PTGENDER'] = df['PTGENDER'].astype('category').cat.codes

    # Train Model
    train_cols = ['ABETA', 'TAU', 'PTAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 
                  'Entorhinal', 'Fusiform', 'AGE', 'PTGENDER', 'APOE4', 
                  'Neurotoxic_Ratio', 'Hippocampus_Fraction']
    
    final_cols = [c for c in train_cols if c in df.columns]
    
    # Monotonic Constraints to ensure logical behavior (e.g., Age increases Risk)
    constraint_map = {
        'ABETA': -1, 'TAU': 1, 'PTAU': 1, 
        'Hippocampus': -1, 'Ventricles': 1, 'WholeBrain': -1, 
        'Entorhinal': -1, 'Fusiform': -1, 
        'AGE': 1, 'PTGENDER': 0, 'APOE4': 1, 
        'Neurotoxic_Ratio': 1, 'Hippocampus_Fraction': -1
    }
    constraints = tuple(constraint_map.get(c, 0) for c in final_cols)

    X = df[final_cols]
    y = df['Target']

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', monotone_constraints=constraints)
    model.fit(X, y)

    return model, df, final_cols

model, df_ref, feature_names = load_and_train()

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000569.png", width=80)
    st.title("NeuroMind AI")
    
    # RESET BUTTON
    if st.button("🔄 Reset Application", type="primary"):
        reset_app()
        
    st.markdown("---")
    
    # API KEY HANDLING
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ AI Doctor Connected (Auto)")
    else:
        st.warning("⚠️ No API Key in Secrets")
        api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 🧪 Patient Input")
    
    age = st.slider("Age", 50, 90, 72)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender == "Female" else 0
    apoe = st.selectbox("APOE4 Alleles (Genetics)", [0, 1, 2], index=0, help="Number of APOE4 alleles (0 = low risk, 1 = moderate risk, 2 = high risk)")
    
    st.markdown("### 🧬 CSF Proteins")
    abeta = st.slider("Amyloid Beta (pg/mL)", 200, 1700, 1000, help="Normal > 900. Low levels indicate plaque accumulation.")
    tau = st.slider("Total Tau (pg/mL)", 80, 1300, 200, help="Normal < 300. High levels indicate neuron damage.")
    ptau = st.slider("Phosphorylated Tau (pg/mL)", 8, 120, 20, help="Normal < 30. The toxic form of Tau.")
    
    st.markdown("### 🧠 MRI Volumetrics")
    hippocampus = st.slider("Hippocampus (mm3)", 4000, 9000, 7000, help="Memory center. Shrinks in AD.")
    ventricles = st.slider("Ventricles (mm3)", 5000, 120000, 25000, help="Fluid filled spaces. Expand in AD.")
    whole_brain = st.number_input("Whole Brain Volume", value=1000000, help="Overall brain volume. Lower values indicate increased global brain atrophy, which is common in Alzheimer’s disease.")
    entorhinal = st.number_input("Entorhinal Cortex", value=3500, help="One of the earliest regions affected by Alzheimer’s disease. Reduced volume here is a strong early indicator of AD-related neurodegeneration.")
    fusiform = st.number_input("Fusiform Gyrus", value=17000, help="Region involved in visual and facial recognition. Decreases in volume may reflect AD progression affecting higher-order visual processing.")
    
    st.markdown("---")
    st.info("**About:** NeuroMind AI fuses Multimodal Data (Proteins + MRI + Genetics) to predict Alzheimer's progression risk.")

# --- 4. MAIN HERO SECTION (INTRODUCTION) ---
st.title("🧠 NeuroMind AI | Alzheimer's Disease")
st.markdown("### Early Detection of Alzheimer's Disease via Multimodal Biomarker Fusion")

st.markdown("""
This system uses **XGBoost** and **Generative AI** to analyze the interaction between molecular pathology and structural atrophy.
It replicates the clinical decision-making process by fusing three diagnostic pillars:
""")

# 3-Column Introduction
c1, c2, c3 = st.columns(3)
with c1:
    st.info("**1. Proteomics (CSF)**")
    st.markdown("Analyzes **Amyloid Beta** (Plaques) and **Tau** (Tangles) levels to detect molecular Alzheimer's pathology. Elevated Amyloid-Beta indicates plaque buildup, while increased Tau reflects neuronal damage and tangle formation. Together, they signal underlying molecular Alzheimer’s pathology even before symptoms appear.")
with c2:
    st.warning("**2. Neuroimaging (MRI)**")
    st.markdown("Quantifies atrophy in the **Hippocampus** and **Ventricles** to measure actual brain tissue loss. Reduced hippocampal volume reflects memory-related degeneration, while enlarged ventricles indicate widespread atrophy—both key structural markers of Alzheimer’s progression.")
with c3:
    st.error("**3. Genetics (APOE)**")
    st.markdown("Factors in **APOE4** allele status, the strongest genetic risk factor for late-onset Alzheimer's. It is inherited from parents. 0, 1, or 2 copies can be present. Possessing one or two copies significantly increases lifetime risk and accelerates disease onset.")

st.markdown("---")

# --- 5. CALCULATION & PREDICTION ---
neurotoxic_ratio = ptau / abeta if abeta > 0 else 0
hippo_fraction = hippocampus / whole_brain if whole_brain > 0 else 0

input_data = pd.DataFrame([[
    abeta, tau, ptau, hippocampus, ventricles, whole_brain, 
    entorhinal, fusiform, age, gender_val, apoe, 
    neurotoxic_ratio, hippo_fraction
]], columns=feature_names)

prob = model.predict_proba(input_data)[0][1]

# --- 6. DIAGNOSIS DASHBOARD ---
st.header("📊 Diagnostic Analysis")

# Color Logic
if prob < 0.30:
    color, status = "green", "Low Risk (Healthy Profile)"
elif prob < 0.70:
    color, status = "orange", "Moderate Risk (MCI Profile)"
else:
    color, status = "red", "High Risk (Alzheimer's Profile)"

# Gauge Chart
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Alzheimer's Conversion Risk", 'font': {'size': 20}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1},
        'bar': {'color': "black"},
        'steps': [
            {'range': [0, 30], 'color': "#2ecc71"},
            {'range': [30, 70], 'color': "#f1c40f"},
            {'range': [70, 100], 'color': "#e74c3c"}
        ],
    }
))
fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))

col1, col2 = st.columns([1, 1.5])

with col1:
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("### 🔬 Critical Biomarkers")
    
    # Ratio Card
    r_color = "green" if neurotoxic_ratio < 0.05 else "red"
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {r_color};">
        <h4 style="margin:0; color: black;">Neurotoxic Ratio (pTau/Aβ)</h4>
        <h2 style="margin:0; color: {r_color};">{neurotoxic_ratio:.4f}</h2>
        <small style="color: black;">Normal < 0.04 | High Risk > 0.08</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Hippocampus Card
    h_color = "green" if hippocampus > 6500 else "red"
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {h_color};">
        <h4 style="margin:0; color: black;">Hippocampal Volume</h4>
        <h2 style="margin:0; color: {h_color};">{hippocampus} mm³</h2>
        <small style="color: black;">Normal > 6500 | Atrophy < 5500</small>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🧠 Pathology Visualizer")
    
    # Visual Logic
    amyloid_bad = abeta < 900 # Low CSF Abeta = High Brain Plaques
    tau_bad = ptau > 30       # High CSF pTau = High Brain Tangles
    
    # SVG Icons (Inline)
    svg_plaque = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="30" fill="#e74c3c" opacity="0.8"/><circle cx="30" cy="30" r="20" fill="#c0392b" opacity="0.8"/><circle cx="70" cy="70" r="25" fill="#c0392b" opacity="0.8"/></svg>'
    svg_clean = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="20" cy="20" r="5" fill="#2ecc71" opacity="0.6"/><circle cx="80" cy="80" r="5" fill="#2ecc71" opacity="0.6"/><circle cx="50" cy="50" r="5" fill="#2ecc71" opacity="0.6"/></svg>'
    
    svg_tangle = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M20,20 Q50,80 80,20" stroke="#e74c3c" stroke-width="5" fill="none"/><path d="M20,80 Q50,20 80,80" stroke="#c0392b" stroke-width="5" fill="none"/></svg>'
    svg_stable = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><line x1="10" y1="30" x2="90" y2="30" stroke="#2ecc71" stroke-width="5"/><line x1="10" y1="50" x2="90" y2="50" stroke="#2ecc71" stroke-width="5"/><line x1="10" y1="70" x2="90" y2="70" stroke="#2ecc71" stroke-width="5"/></svg>'

    a_icon = svg_plaque if amyloid_bad else svg_clean
    a_text = "High Plaque Load" if amyloid_bad else "Normal Clearance"
    a_color = "#e74c3c" if amyloid_bad else "#2ecc71"

    t_icon = svg_tangle if tau_bad else svg_stable
    t_text = "Neurofibrillary Tangles" if tau_bad else "Stable Microtubules"
    t_color = "#e74c3c" if tau_bad else "#2ecc71"

    st.markdown(f"""
    <div style="display: flex; gap: 10px;">
        <div style="flex: 1; background: white; padding: 10px; border-radius: 10px; border: 2px solid {a_color}; text-align: center;">
            <div style="height: 60px; width: 60px; margin: 0 auto;">{a_icon}</div>
            <div style="font-weight: bold; font-size: 14px; margin-top: 5px; color: black;">Amyloid Beta</div>
            <div style="font-size: 12px; color: {a_color};">{a_text}</div>
        </div>
        <div style="flex: 1; background: white; padding: 10px; border-radius: 10px; border: 2px solid {t_color}; text-align: center;">
            <div style="height: 60px; width: 60px; margin: 0 auto;">{t_icon}</div>
            <div style="font-weight: bold; font-size: 14px; margin-top: 5px; color: black;">Tau Protein</div>
            <div style="font-size: 12px; color: {t_color};">{t_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📍 Patient Positioning Map")
    st.info("The Blue Star represents the current patient relative to the ADNI population.")
    
    fig = px.scatter(df_ref, x='Neurotoxic_Ratio', y='Hippocampus', color='Target',
                     color_continuous_scale=['#2ecc71', '#e74c3c'], opacity=0.2,
                     labels={'Target': 'Diagnosis'}, height=400)
    fig.add_scatter(x=[neurotoxic_ratio], y=[hippocampus], mode='markers', 
                    marker=dict(size=25, color='blue', symbol='star', line=dict(width=2, color='white')), 
                    name='Current Patient')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 7. AI CONSULTANT ---
st.markdown("---")
st.subheader("🤖 AI Neurologist Consultation")

if st.button("📝 Generate Clinical Narrative"):
    if not api_key:
        st.error("❌ No API Key found in .streamlit/secrets.toml")
    else:
        with st.spinner("Analyzing multimodal data..."):
            try:
                genai.configure(api_key=api_key)
                # Using Gemini 2.0 Flash as confirmed working
                llm = genai.GenerativeModel('gemini-2.5-flash')
                
                # Calculate SHAP values for the prompt
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_data)
                
                # Get top contributing features
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values[0].values
                })
                shap_df['abs_shap'] = shap_df['shap_value'].abs()
                top_features = shap_df.sort_values('abs_shap', ascending=False).head(5)
                
                shap_summary = "\n".join([
                    f"- {row['feature']}: {row['shap_value']:.4f} (Impact: {'Increases Risk' if row['shap_value'] > 0 else 'Decreases Risk'})"
                    for _, row in top_features.iterrows()
                ])

                # Pathology Visual State for LLM
                amyloid_state = "High Plaque Load (Red Icon - Plaques Present)" if abeta < 900 else "Normal Clearance (Green Icon - Healthy)"
                tau_state = "Neurofibrillary Tangles (Red Icon - Tangles Present)" if ptau > 30 else "Stable Microtubules (Green Icon - Healthy)"

                prompt = f"""
                Act as a senior Neurologist. Analyze this patient data:
                - Risk Probability: {prob:.1%} ({status})
                - Amyloid Beta: {abeta} (Low is bad) -> Visual State: {amyloid_state}
                - Phosphorylated Tau: {ptau} (High is bad) -> Visual State: {tau_state}
                - Neurotoxic Ratio: {neurotoxic_ratio:.4f} (High is bad)
                - Hippocampus: {hippocampus} (Low is bad)
                - APOE4 Genes: {apoe}
                
                SHAP Analysis (Feature Importance for this specific prediction):
                {shap_summary}

                Format the response in Markdown:
                ### 📋 Clinical Summary
                [One concise paragraph highlighting key findings]
                
                ### 🔍 Key Pathology Drivers
                * **Protein:** [Analyze Ratio]
                * **Structure:** [Analyze Hippocampus]
                
                ### 📊 SHAP Analysis Explanation
                [Explain the SHAP figure and which features drove the prediction the most based on the provided SHAP values]

                ### 🧠 Pathology Visualizer Interpretation
                [Explain what the visual icons for Amyloid Beta and Tau Protein indicate for this specific patient. Mention if they show Plaques or Tangles based on the Visual State provided above.]

                ### 🛡️ Recommendation
                [Actionable advice with references to guidelines and studies]
                
                ### 📚 References
                [List of 2-3 recent studies or guidelines on Alzheimer's diagnosis and management]
                """
                
                response = llm.generate_content(prompt)
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"AI Error: {e}")

# --- 8. SHAP EXPLANATION ---
with st.expander("View Logic (SHAP Analysis)"):
    st.write("This waterfall chart shows exactly how much each feature contributed to the risk score.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    fig_shap, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], show=False, max_display=12)
    st.pyplot(fig_shap)

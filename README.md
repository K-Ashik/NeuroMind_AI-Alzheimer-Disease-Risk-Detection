# 🧠 NeuroMind AI: Multimodal Alzheimer's Risk Assessment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![GenAI](https://img.shields.io/badge/GenAI-Google%20Gemini-purple)

**NeuroMind AI** is a clinical decision support system designed to bridge the gap between complex biological data and actionable medical insights. It predicts the risk of Alzheimer's Disease (AD) by fusing three distinct data modalities:
1.  **Proteomics:** Cerebrospinal Fluid (CSF) biomarkers (Amyloid Beta, Tau).
2.  **Neuroimaging:** MRI volumetrics (Hippocampus, Ventricles).
3.  **Genetics:** APOE4 allele status.

It goes beyond simple prediction by integrating a **Generative AI Doctor** (Google Gemini) to explain the *biology* behind the risk score in plain English.

---
## 🚀 Live Demo
Click here: [https://neuromindai-alzheimer-disease-risk-detection-4pktih6hk5k2yzmge.streamlit.app/]
---

## 🚀 Key Features

### 1. The "Neurotoxic Ratio" Engine
Raw protein levels can be misleading. I engineered a custom feature, the **Neurotoxic Ratio** (`pTau / Abeta`), which quantifies the imbalance between neuronal damage (Tau) and plaque clearance (Amyloid). 
* **Result:** This engineered feature proved to be a stronger predictor than raw protein levels alone, increasing model accuracy.

### 2. Multimodal Fusion
The model doesn't look at just one factor. It learns non-linear interactions, such as:
* *"High Tau is bad, but High Tau + Shrinking Hippocampus + APOE4 Gene is catastrophic."*

### 3. The AI Neurologist (LLM Integration)
Instead of just giving a probability ("85% Risk"), the app uses **Google Gemini 2.5 Flash** to generate a clinical narrative:
> *"Patient shows a critical imbalance in the Neurotoxic Ratio (0.08), suggesting active pathology. However, Hippocampal volume is preserved, indicating early-stage disease with cognitive reserve."*

---

## 📊 Model Performance
Trained on the **TADPOLE (ADNI)** dataset, a gold-standard longitudinal study.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **80.65%** | Strong predictive power on unseen patients |
| **ROC-AUC** | **0.88** | Excellent discrimination between Healthy vs. At-Risk |
| **Recall (Sick)** | **0.86** | Misses very few actual cases (High Sensitivity) |

### **Explainability (SHAP)**
<img width="800" height="774" alt="image" src="https://github.com/user-attachments/assets/559d432b-2b2f-4d73-8a52-7bafdd071e4c" />

*The model confirms that **Neurotoxic Ratio** and **TAU** are the top drivers of risk, validating the biological hypothesis.*

---

## 🛠️ Tech Stack
* **Core Logic:** Python, Pandas, NumPy
* **Machine Learning:** XGBoost (Gradient Boosting)
* **Generative AI:** Google Gemini 2.0 Flash (via API)
* **Explainability:** SHAP (Shapley Values)
* **Visualization:** Plotly (Interactive Charts), Matplotlib
* **App Framework:** Streamlit

---

## 🚀 How to Run Locally

1. **Clone the Repo**
   ```bash
   git clone [https://github.com/K-Ashik/NeuroMind_AI-Alzheimer-Disease-Risk-Detection.git](https://github.com/K-Ashik/NeuroMind_AI-Alzheimer-Disease-Risk-Detection.git)
   cd NeuroMind_AI-Alzheimer-Disease-Risk-Detection

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   
3. Get a Free Gemini API Key
   Go to Google AI Studio and get a key.

   Add it to .streamlit/secrets.toml or paste it in the app sidebar.

4. Run the App
   ```bash
   streamlit run app.py

⚠️ Disclaimer
For Educational & Portfolio Purposes Only. This tool utilizes clinical trial data (ADNI) but has not been FDA-approved for clinical diagnosis. It is a proof-of-concept for AI-assisted screening.

📬 Contact
Created by [Khalid Md Ashik] - LinkedIn [https://www.linkedin.com/in/khalid-md-ashik/] | GitHub [https://github.com/K-Ashik]
   
   

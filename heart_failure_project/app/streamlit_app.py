"""
Heart Failure Patient Assessment & Care System
A clinical decision support platform for physiological risk analysis and evidence-based care guidance.
Features custom Dark & Light themes with high-contrast clinical styling.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="CardioClinical Assessment System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# THEME STATE MANAGEMENT (Default: Dark Mode)
# -----------------------------------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'

is_dark = (st.session_state['theme'] == 'dark')

# Theme Color Palettes
if is_dark:
    theme_colors = {
        'bg_app': '#0B0F19',
        'bg_card': '#111827',
        'bg_card_inner': '#1F2937',
        'border': '#374151',
        'text_primary': '#F9FAFB',
        'text_secondary': '#9CA3AF',
        'nav_bg': '#111827',
        'tab_bg': '#111827',
        'tab_active_bg': '#1F2937',
        'tab_active_text': '#38BDF8',
        'gauge_bg': '#1F2937',
        'gauge_axis': '#9CA3AF',
        'gauge_green': '#064E3B',
        'gauge_amber': '#78350F',
        'gauge_red': '#7F1D1D',
        'grid_color': '#374151',
        'alert_high_bg': '#450A0A',
        'alert_high_border': '#EF4444',
        'alert_high_text': '#FCA5A5',
        'alert_low_bg': '#064E3B',
        'alert_low_border': '#10B981',
        'alert_low_text': '#6EE7B7'
    }
else:
    theme_colors = {
        'bg_app': '#F8FAFC',
        'bg_card': '#FFFFFF',
        'bg_card_inner': '#F1F5F9',
        'border': '#CBD5E1',
        'text_primary': '#0F172A',
        'text_secondary': '#475569',
        'nav_bg': '#0F172A',
        'tab_bg': '#FFFFFF',
        'tab_active_bg': '#E2E8F0',
        'tab_active_text': '#0F172A',
        'gauge_bg': '#F1F5F9',
        'gauge_axis': '#475569',
        'gauge_green': '#ECFDF5',
        'gauge_amber': '#FEF3C7',
        'gauge_red': '#FEE2E2',
        'grid_color': '#E2E8F0',
        'alert_high_bg': '#FEF2F2',
        'alert_high_border': '#DC2626',
        'alert_high_text': '#7F1D1D',
        'alert_low_bg': '#ECFDF5',
        'alert_low_border': '#059669',
        'alert_low_text': '#064E3B'
    }

# Inject High-Contrast CSS matching active theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {theme_colors['bg_app']} !important;
        color: {theme_colors['text_primary']} !important;
    }}
    
    /* Hide default sidebar completely */
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    /* Global Label and Text Contrast */
    label, p, span, h1, h2, h3, h4, h5, h6, .stMarkdown {{
        color: {theme_colors['text_primary']} !important;
    }}
    
    /* Top Navigation Bar */
    .top-navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {theme_colors['nav_bg']};
        padding: 14px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid {theme_colors['border']};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }}
    .nav-brand-title {{
        font-size: 20px;
        font-weight: 700;
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .nav-meta-text {{
        font-size: 13px;
        color: #94A3B8 !important;
        font-weight: 500;
    }}
    .badge-status {{
        background: #059669;
        color: #FFFFFF !important;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Custom Styled Top Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {theme_colors['tab_bg']} !important;
        border-radius: 10px;
        padding: 6px 12px;
        border: 1px solid {theme_colors['border']};
        gap: 8px;
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        font-size: 14px;
        font-weight: 600;
        color: {theme_colors['text_secondary']} !important;
        border-radius: 8px;
        padding: 0 16px;
        border: none !important;
        background-color: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme_colors['tab_active_text']} !important;
        background-color: {theme_colors['tab_active_bg']} !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }}
    
    /* Custom Cards */
    .clinical-card {{
        background: {theme_colors['bg_card']};
        border: 1px solid {theme_colors['border']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }}
    .card-title {{
        font-size: 16px;
        font-weight: 700;
        color: {theme_colors['text_primary']} !important;
        margin-bottom: 4px;
    }}
    .card-subtitle {{
        font-size: 13px;
        color: {theme_colors['text_secondary']} !important;
        margin-bottom: 14px;
    }}
    
    /* Custom Indicator Pills */
    .pill-danger {{
        background: {theme_colors['gauge_red']};
        color: #FCA5A5 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
        border: 1px solid #EF4444;
    }}
    .pill-success {{
        background: {theme_colors['gauge_green']};
        color: #6EE7B7 !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
        border: 1px solid #10B981;
    }}
    .pill-neutral {{
        background: {theme_colors['bg_card_inner']};
        color: {theme_colors['text_secondary']} !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
        border: 1px solid {theme_colors['border']};
    }}
    /* Theme Toggle Icon Button */
    div[data-testid="stColumn"]:nth-of-type(2) .stButton button {{
        border-radius: 10px !important;
        height: 52px !important;
        font-size: 22px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid {theme_colors['border']} !important;
        background-color: {theme_colors['bg_card']} !important;
        color: {theme_colors['text_primary']} !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stColumn"]:nth-of-type(2) .stButton button:hover {{
        border-color: #38BDF8 !important;
        transform: scale(1.05);
    }}
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts quietly
@st.cache_resource
def load_system_core():
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    recommender = joblib.load(os.path.join(models_dir, 'recommender.pkl'))
    cluster_model = joblib.load(os.path.join(models_dir, 'cluster_model.pkl'))
    raw_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'heart_failure_clinical_records.csv'))
    
    meta = joblib.load(os.path.join(models_dir, 'best_model_meta.pkl'))
    if meta.get('type') == 'keras':
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(models_dir, meta['path']))
    else:
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        
    return {
        'model': model,
        'model_type': meta.get('type', 'sklearn'),
        'scaler': scaler,
        'feature_names': feature_names,
        'recommender': recommender,
        'cluster_model': cluster_model,
        'raw_df': raw_df
    }

try:
    core = load_system_core()
    system_ready = True
except Exception as e:
    system_ready = False
    load_error = str(e)

# -----------------------------------------------------------------------------
# TOP NAVBAR WITH THEME SWITCHER
# -----------------------------------------------------------------------------
col_header_left, col_header_right = st.columns([5.3, 0.7])

with col_header_left:
    st.markdown("""
    <div class="top-navbar" style="margin-bottom: 0;">
        <div class="nav-brand-title">
            <span style="font-size: 24px;">🫀</span>
            <span>CardioClinical Decision Support System</span>
        </div>
        <div class="nav-meta-text">
            <span class="badge-status">Active</span>
            &nbsp;Database: 299 Records &nbsp;|&nbsp; Reference: UCI Clinical Cardiology
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_header_right:
    # Compact icon-only theme toggle button
    theme_icon = "☀️" if is_dark else "🌙"
    theme_tooltip = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
    if st.button(theme_icon, help=theme_tooltip, key="theme_icon_toggle", use_container_width=True):
        st.session_state['theme'] = 'light' if is_dark else 'dark'
        st.rerun()

st.write("")

if not system_ready:
    st.error(f"System initialization error: {load_error}")
    st.stop()

# Top Navbar Navigation Tabs
tab_assess, tab_categories, tab_recommend = st.tabs([
    "🩺 Patient Risk Assessment & Vitals Analysis",
    "👥 Clinical Cohort Profiles & Population Distribution",
    "💊 Evidence-Based Care Pathways & Treatment Matching"
])

# -----------------------------------------------------------------------------
# TAB 1: PATIENT RISK ASSESSMENT & VITALS ANALYSIS
# -----------------------------------------------------------------------------
with tab_assess:
    col_preset_lbl, col_case1, col_case2, col_case3 = st.columns([1.2, 1, 1, 1])
    with col_preset_lbl:
        st.markdown(f"<div style='padding-top: 8px; font-weight: 600; color: {theme_colors['text_secondary']};'>Load Hospital Case Record:</div>", unsafe_allow_html=True)
    with col_case1:
        if st.button("🟢 Case 1: Ramesh Gupta (Stable)", use_container_width=True):
            st.session_state['v_name'] = "Ramesh Gupta"
            st.session_state['v_mrn'] = "MRN-2026-1042"
            st.session_state['v_sex'] = "Male"
            st.session_state['v_ward'] = "Outpatient Cardiology Clinic"
            st.session_state['v_age'] = 54
            st.session_state['v_ef'] = 50
            st.session_state['v_creat'] = 0.9
            st.session_state['v_sod'] = 140
            st.session_state['v_bp'] = False
            st.session_state['v_diab'] = False
            st.session_state['v_time'] = 180
    with col_case2:
        if st.button("🔴 Case 2: Robert Chen (Critical)", use_container_width=True):
            st.session_state['v_name'] = "Robert Chen"
            st.session_state['v_mrn'] = "MRN-2026-3819"
            st.session_state['v_sex'] = "Male"
            st.session_state['v_ward'] = "Cardiac Intensive Care (CCU)"
            st.session_state['v_age'] = 74
            st.session_state['v_ef'] = 20
            st.session_state['v_creat'] = 2.4
            st.session_state['v_sod'] = 130
            st.session_state['v_bp'] = True
            st.session_state['v_diab'] = True
            st.session_state['v_time'] = 30
    with col_case3:
        if st.button("🟡 Case 3: Eleanor Vance (Senior)", use_container_width=True):
            st.session_state['v_name'] = "Eleanor Vance"
            st.session_state['v_mrn'] = "MRN-2026-5521"
            st.session_state['v_sex'] = "Female"
            st.session_state['v_ward'] = "Inpatient Cardiology Ward"
            st.session_state['v_age'] = 68
            st.session_state['v_ef'] = 35
            st.session_state['v_creat'] = 1.3
            st.session_state['v_sod'] = 135
            st.session_state['v_bp'] = True
            st.session_state['v_diab'] = False
            st.session_state['v_time'] = 90

    # Defaults
    if 'v_name' not in st.session_state: st.session_state['v_name'] = "Ramesh Gupta"
    if 'v_mrn' not in st.session_state: st.session_state['v_mrn'] = "MRN-2026-1042"
    if 'v_sex' not in st.session_state: st.session_state['v_sex'] = "Male"
    if 'v_ward' not in st.session_state: st.session_state['v_ward'] = "Outpatient Cardiology Clinic"
    if 'v_age' not in st.session_state: st.session_state['v_age'] = 62
    if 'v_ef' not in st.session_state: st.session_state['v_ef'] = 38
    if 'v_creat' not in st.session_state: st.session_state['v_creat'] = 1.2
    if 'v_sod' not in st.session_state: st.session_state['v_sod'] = 136
    if 'v_bp' not in st.session_state: st.session_state['v_bp'] = True
    if 'v_diab' not in st.session_state: st.session_state['v_diab'] = False
    if 'v_time' not in st.session_state: st.session_state['v_time'] = 120

    st.write("")
    col_input, col_analytics = st.columns([1.5, 1.5])

    with col_input:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">Hospital Patient Admission & Intake Record</div>
            <div class="card-subtitle">General demographics, clinical identification, and medical records</div>
        """, unsafe_allow_html=True)
        
        # Section 1: Demographics
        st.markdown(f"<div style='font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 8px;'>SECTION 1: PATIENT DEMOGRAPHICS & IDENTIFICATION</div>", unsafe_allow_html=True)
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            p_name = st.text_input("Patient Full Name", value=st.session_state['v_name'])
            p_age = st.number_input("Age (Years)", min_value=18, max_value=105, value=st.session_state['v_age'], step=1)
            p_sex = st.selectbox("Biological Sex / Gender", ["Male", "Female"], index=0 if st.session_state['v_sex'] == "Male" else 1)
        with demo_col2:
            p_mrn = st.text_input("Hospital / Medical Record ID (MRN)", value=st.session_state['v_mrn'])
            ward_options = ["Outpatient Cardiology Clinic", "Inpatient Cardiology Ward", "Cardiac Intensive Care (CCU)", "Emergency Observation"]
            p_ward = st.selectbox("Clinical Department / Ward", ward_options, index=ward_options.index(st.session_state['v_ward']) if st.session_state['v_ward'] in ward_options else 0)
            p_followup = st.number_input("Observation / Follow-Up Window (Days)", min_value=1, max_value=300, value=st.session_state['v_time'], step=5)

        st.markdown(f"<hr style='border: none; border-top: 1px solid {theme_colors['border']}; margin: 16px 0;'>", unsafe_allow_html=True)

        # Section 2: Vitals & Hemodynamics
        st.markdown(f"<div style='font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 8px;'>SECTION 2: CARDIAC & RENAL BIOMARKER PANEL</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ef = st.slider("Left Ventricular Ejection Fraction (LVEF %)", min_value=10, max_value=80, value=st.session_state['v_ef'],
                           help="Healthy reference: 55-70%. Depressed EF (<40%) indicates systolic dysfunction.")
            creat = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=10.0, value=float(st.session_state['v_creat']), step=0.1,
                                    help="Healthy reference: 0.6-1.3 mg/dL. Values >1.4 denote renal impairment.")
        with c2:
            sod = st.number_input("Serum Sodium (mEq/L)", min_value=105, max_value=155, value=st.session_state['v_sod'], step=1,
                                  help="Healthy reference: 135-145 mEq/L.")
            cpk = st.number_input("Creatinine Phosphokinase - CPK (mcg/L)", min_value=20, max_value=10000, value=582, step=25)

        platelets = st.number_input("Platelets Count (kiloplatelets/mL)", min_value=20000.0, max_value=900000.0, value=263000.0, step=5000.0)

        st.markdown(f"<hr style='border: none; border-top: 1px solid {theme_colors['border']}; margin: 16px 0;'>", unsafe_allow_html=True)

        # Section 3: Comorbidities
        st.markdown(f"<div style='font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 8px;'>SECTION 3: CHRONIC COMORBIDITIES & MEDICAL HISTORY</div>", unsafe_allow_html=True)
        ch1, ch2, ch3, ch4 = st.columns(4)
        with ch1:
            high_bp = st.checkbox("Hypertension", value=st.session_state['v_bp'])
        with ch2:
            diabetes = st.checkbox("Diabetes", value=st.session_state['v_diab'])
        with ch3:
            anaemia = st.checkbox("Anaemia", value=False)
        with ch4:
            smoking = st.checkbox("Smoker", value=False)

        run_btn = st.button("Generate Hospital Clinical Evaluation Report", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_analytics:
        # Preprocessing calculations
        renal_ratio = creat / (ef + 1e-5)
        hypo_flag = 1 if sod < 135 else 0
        severe_ef = 1 if ef < 30 else 0
        age_risk = 1 if p_age >= 65 else 0
        high_creat = 1 if creat > 1.4 else 0
        comorbid_count = int(anaemia) + int(diabetes) + int(high_bp) + int(smoking)

        patient_record = {
            'age': float(p_age),
            'anaemia': 1 if anaemia else 0,
            'creatinine_phosphokinase': int(cpk),
            'diabetes': 1 if diabetes else 0,
            'ejection_fraction': int(ef),
            'high_blood_pressure': 1 if high_bp else 0,
            'platelets': float(platelets),
            'serum_creatinine': float(creat),
            'serum_sodium': int(sod),
            'sex': 1 if p_sex == "Male" else 0,
            'smoking': 1 if smoking else 0,
            'time': int(p_followup),
            'Renal_Cardiac_Ratio': float(renal_ratio),
            'Hyponatremia_Flag': hypo_flag,
            'Severe_EF_Flag': severe_ef,
            'Age_Risk_Group': age_risk,
            'High_Creatinine_Flag': high_creat,
            'Comorbidity_Count': comorbid_count
        }

        input_df = pd.DataFrame([patient_record])[core['feature_names']]
        scaled_input = core['scaler'].transform(input_df)

        if core['model_type'] == 'keras':
            prob_mortality = float(core['model'].predict(scaled_input, verbose=0)[0][0])
            predicted_class = int(prob_mortality >= 0.5)
        else:
            predicted_class = int(core['model'].predict(scaled_input)[0])
            prob_mortality = float(core['model'].predict_proba(scaled_input)[0][1])

        prob_survival = 1.0 - prob_mortality

        # Visual 1: High-Contrast Plotly Gauge Chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_mortality * 100, 1),
            number={'suffix': "%", 'font': {'size': 36, 'color': theme_colors['text_primary'], 'family': 'Inter'}},
            title={'text': "Calculated Mortality Risk Score", 'font': {'size': 14, 'color': theme_colors['text_secondary'], 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': theme_colors['gauge_axis']},
                'bar': {'color': "#EF4444" if prob_mortality >= 0.5 else "#10B981", 'thickness': 0.3},
                'bgcolor': theme_colors['gauge_bg'],
                'borderwidth': 1,
                'bordercolor': theme_colors['border'],
                'steps': [
                    {'range': [0, 30], 'color': theme_colors['gauge_green']},
                    {'range': [30, 60], 'color': theme_colors['gauge_amber']},
                    {'range': [60, 100], 'color': theme_colors['gauge_red']}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        gauge_fig.update_layout(
            height=210,
            margin=dict(l=20, r=20, t=35, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.markdown(f"""
        <div class="clinical-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid {theme_colors['border']}; padding-bottom: 10px; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 18px; font-weight: 700; color: {theme_colors['text_primary']};">{p_name}</div>
                    <div style="font-size: 12px; color: {theme_colors['text_secondary']}; margin-top: 2px;">
                        MRN: <b style="color: #38BDF8;">{p_mrn}</b> &nbsp;|&nbsp; {p_age} Yrs &nbsp;|&nbsp; Sex: <b>{p_sex}</b> &nbsp;|&nbsp; {p_ward}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="badge-status">{p_ward.split()[0]}</span>
                </div>
            </div>
            <div class="card-title">Clinical Risk Assessment & Vital Status</div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(gauge_fig, use_container_width=True)

        if predicted_class == 1 or prob_mortality >= 0.50:
            st.markdown(f"""
            <div style="background: {theme_colors['alert_high_bg']}; border-left: 4px solid {theme_colors['alert_high_border']}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: {theme_colors['alert_high_text']}; font-size: 15px;">⚠️ Elevated Cardiac Decompensation Risk</div>
                <div style="font-size: 13px; color: {theme_colors['alert_high_text']}; margin-top: 2px;">
                    Estimated Mortality Risk: <b>{prob_mortality * 100:.1f}%</b> (Survival Probability: {prob_survival * 100:.1f}%). 
                    Close hemodynamic monitoring and cardiology inpatient consult recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: {theme_colors['alert_low_bg']}; border-left: 4px solid {theme_colors['alert_low_border']}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: {theme_colors['alert_low_text']}; font-size: 15px;">✅ Stable Prognosis / Favorable Survival</div>
                <div style="font-size: 13px; color: {theme_colors['alert_low_text']}; margin-top: 2px;">
                    Estimated Survival Probability: <b>{prob_survival * 100:.1f}%</b> (Mortality Risk: {prob_mortality * 100:.1f}%). 
                    Clinical profile supports routine outpatient maintenance care.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual 2: Vitals Reference Comparison Chart
        st.markdown(f"<div style='font-size: 13px; font-weight: 600; color: {theme_colors['text_secondary']}; margin-bottom: 6px;'>Biomarker Values vs Standard Reference Levels:</div>", unsafe_allow_html=True)
        
        vitals_labels = ['Ejection Fraction (%)', 'Serum Creatinine (mg/dL)', 'Serum Sodium (mEq/L)']
        patient_vals = [ef, creat, sod]
        status_colors = [
            '#10B981' if ef >= 45 else '#EF4444',
            '#10B981' if creat <= 1.3 else '#EF4444',
            '#10B981' if sod >= 135 else '#EF4444'
        ]

        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            y=vitals_labels,
            x=patient_vals,
            orientation='h',
            marker_color=status_colors,
            text=[f"{v}" for v in patient_vals],
            textposition='auto',
            textfont=dict(color='#FFFFFF' if is_dark else '#000000'),
            name="Patient Value"
        ))
        fig_bars.update_layout(
            height=150,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor=theme_colors['grid_color'], tickfont=dict(color=theme_colors['text_secondary'])),
            yaxis=dict(autorange="reversed", tickfont=dict(color=theme_colors['text_primary']))
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # Active physiological tags
        chips = ""
        if severe_ef:
            chips += '<span class="pill-danger">⚠️ EF < 30% (Severe)</span>'
        else:
            chips += f'<span class="pill-success">✓ EF: {ef}%</span>'
            
        if high_creat:
            chips += f'<span class="pill-danger">⚠️ Creatinine: {creat} (Elevated)</span>'
        else:
            chips += f'<span class="pill-success">✓ Creatinine: {creat}</span>'
            
        if hypo_flag:
            chips += f'<span class="pill-danger">⚠️ Sodium: {sod} (Hyponatremia)</span>'
            
        chips += f'<span class="pill-neutral">Comorbidities: {comorbid_count}/4</span>'
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: CLINICAL COHORT PROFILES & POPULATION DISTRIBUTION
# -----------------------------------------------------------------------------
with tab_categories:
    raw_df = core['raw_df']
    
    col_dist_chart, col_pop_pie = st.columns([1.8, 1.2])
    
    with col_dist_chart:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">Cohort Physiological Distribution (Ejection Fraction vs Serum Creatinine)</div>
            <div class="card-subtitle">Patient mapping across the 299 clinical records database</div>
        """, unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            raw_df,
            x="ejection_fraction",
            y="serum_creatinine",
            color="DEATH_EVENT",
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            labels={"ejection_fraction": "Ejection Fraction (%)", "serum_creatinine": "Serum Creatinine (mg/dL)", "DEATH_EVENT": "Observed Outcome"},
            hover_data=["age", "serum_sodium", "time"],
            opacity=0.8
        )
        fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='#1F2937' if is_dark else 'white')))
        # Highlight current patient point
        fig_scatter.add_trace(go.Scatter(
            x=[ef],
            y=[creat],
            mode='markers+text',
            marker=dict(size=16, color='#38BDF8', symbol='star', line=dict(width=2, color='white')),
            name='Current Patient',
            text=["Current Patient"],
            textposition="top right"
        ))
        fig_scatter.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text_primary']),
            xaxis=dict(showgrid=True, gridcolor=theme_colors['grid_color'], tickfont=dict(color=theme_colors['text_secondary'])),
            yaxis=dict(showgrid=True, gridcolor=theme_colors['grid_color'], tickfont=dict(color=theme_colors['text_secondary'])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=theme_colors['text_primary']))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pop_pie:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">Clinical Cohort Breakdown</div>
            <div class="card-subtitle">Distribution across 3 patient risk phenotypes</div>
        """, unsafe_allow_html=True)
        
        cohort_labels = ['Stable Outpatients', 'Cardiorenal Risk', 'Elderly Comorbid']
        cohort_counts = [142, 81, 76]
        cohort_colors = ['#10B981', '#EF4444', '#F59E0B']

        fig_pie = go.Figure(data=[go.Pie(
            labels=cohort_labels,
            values=cohort_counts,
            hole=.45,
            marker_colors=cohort_colors,
            textinfo='label+percent',
            textfont=dict(color='#FFFFFF' if is_dark else '#000000')
        )])
        fig_pie.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text_primary']),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Clinical Category Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="clinical-card" style="border-top: 4px solid #10B981;">
            <div style="font-weight: 700; color: #10B981; font-size: 16px;">🟢 Category 1: Stable Patients</div>
            <div style="font-size: 12px; color: {theme_colors['text_secondary']}; font-weight: 600; margin-bottom: 8px;">Observed Mortality Rate: ~12%</div>
            <p style="font-size: 13px; color: {theme_colors['text_primary']}; line-height: 1.5; margin: 0;">
                <b>Vitals Profile:</b> Preserved ejection fraction (>45%), normal serum creatinine (<1.1 mg/dL), normal electrolytes.<br><br>
                <b>Clinical Pathway:</b> Routine outpatient follow-up, sodium restriction, annual echocardiogram.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="clinical-card" style="border-top: 4px solid #EF4444;">
            <div style="font-weight: 700; color: #EF4444; font-size: 16px;">🔴 Category 2: Cardiorenal Patients</div>
            <div style="font-size: 12px; color: {theme_colors['text_secondary']}; font-weight: 600; margin-bottom: 8px;">Observed Mortality Rate: ~58%</div>
            <p style="font-size: 13px; color: {theme_colors['text_primary']}; line-height: 1.5; margin: 0;">
                <b>Vitals Profile:</b> Severely depressed EF (<30%), elevated serum creatinine (>1.8 mg/dL), hyponatremia (<135 mEq/L).<br><br>
                <b>Clinical Pathway:</b> Cardiology-nephrology co-management, loop diuretic titration, continuous fluid balance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="clinical-card" style="border-top: 4px solid #F59E0B;">
            <div style="font-weight: 700; color: #F59E0B; font-size: 16px;">🟡 Category 3: Senior Comorbid</div>
            <div style="font-size: 12px; color: {theme_colors['text_secondary']}; font-weight: 600; margin-bottom: 8px;">Observed Mortality Rate: ~34%</div>
            <p style="font-size: 13px; color: {theme_colors['text_primary']}; line-height: 1.5; margin: 0;">
                <b>Vitals Profile:</b> Age > 70 years with multiple chronic risk factors (hypertension, diabetes, anaemia).<br><br>
                <b>Clinical Pathway:</b> Structured home telemonitoring, medication reconciliation, fall risk mitigation.
            </p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: EVIDENCE-BASED CARE PATHWAYS & TREATMENT MATCHING
# -----------------------------------------------------------------------------
with tab_recommend:
    recommender = core['recommender']
    
    col_p_select, col_rec_visual = st.columns([1.1, 1.9])
    
    with col_p_select:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">Select Patient Cohort Record</div>
            <div class="card-subtitle">Match against historical guideline-directed therapies</div>
        """, unsafe_allow_html=True)
        
        patient_ids = list(recommender.patient_intervention_matrix.index[:30])
        selected_pid = st.selectbox("Select Patient Record ID:", patient_ids, index=1)
        
        st.write(f"**Baseline Indication Scores for {selected_pid}:**")
        p_ratings = recommender.patient_intervention_matrix.loc[selected_pid]
        st.dataframe(p_ratings.to_frame(name="Indication Score (1-5)"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_rec_visual:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">Recommended Evidence-Based Clinical Interventions</div>
            <div class="card-subtitle">Therapies indicated by matching clinical cohort cases</div>
        """, unsafe_allow_html=True)
        
        suggestions = recommender.recommend_cohort_based(selected_pid, top_n=4)
        
        therapy_names = [s['intervention'] for s in suggestions]
        therapy_scores = [s['score'] for s in suggestions]
        
        fig_rec = go.Figure(go.Bar(
            x=therapy_scores,
            y=therapy_names,
            orientation='h',
            marker_color='#38BDF8' if is_dark else '#0284C7',
            text=[f"{score} / 5.0" for score in therapy_scores],
            textposition='auto',
            textfont=dict(color='#FFFFFF' if is_dark else '#000000')
        ))
        fig_rec.update_layout(
            height=180,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text_primary']),
            xaxis=dict(range=[0, 5.5], showgrid=True, gridcolor=theme_colors['grid_color'], tickfont=dict(color=theme_colors['text_secondary'])),
            yaxis=dict(autorange="reversed", tickfont=dict(color=theme_colors['text_primary']))
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        st.markdown("---")
        for i, s in enumerate(suggestions):
            st.markdown(f"""
            <div style="background: {theme_colors['bg_card_inner']}; border: 1px solid {theme_colors['border']}; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: {theme_colors['text_primary']}; font-size: 14px;">#{i+1}: {s['intervention']}</span>
                    <span style="background: {'#0369A1' if is_dark else '#DBEAFE'}; color: {'#E0F2FE' if is_dark else '#0369A1'}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Priority Match: {s['score']}/5.0</span>
                </div>
                <div style="font-size: 13px; color: {theme_colors['text_secondary']}; margin-top: 4px;">{s['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

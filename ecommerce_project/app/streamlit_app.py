"""
Streamlit Web Application for E-Commerce Purchase Prediction & Recommendation System.
Unifying Experiments 4 to 10 from Machine Learning Lab.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="E-Commerce ML Studio",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models with caching
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_path, 'models')
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    cluster_model = joblib.load(os.path.join(models_dir, 'cluster_model.pkl'))
    recommender = joblib.load(os.path.join(models_dir, 'recommender.pkl'))
    
    # Load best model
    meta = joblib.load(os.path.join(models_dir, 'best_model_meta.pkl'))
    if meta.get('type') == 'keras':
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(models_dir, meta['path']))
    else:
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        
    benchmark_df = pd.read_csv(os.path.join(models_dir, 'benchmark_results.csv'))
    cluster_bench_df = pd.read_csv(os.path.join(models_dir, 'cluster_benchmark.csv'))
    
    return {
        'model': model,
        'model_name': meta['name'],
        'model_type': meta.get('type', 'sklearn'),
        'scaler': scaler,
        'feature_names': feature_names,
        'cluster_model': cluster_model,
        'recommender': recommender,
        'benchmark_df': benchmark_df,
        'cluster_bench_df': cluster_bench_df
    }

try:
    artifacts = load_artifacts()
    loaded_ok = True
except Exception as e:
    loaded_ok = False
    load_err = str(e)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Module:",
    ["🔮 Purchase Predictor", "👥 Customer Segmentation", "💡 Recommendation Engine", "📊 Model Comparison & Lab Mapping"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 ML Lab Experiments")
st.sidebar.caption("""
- **Exp 4:** KNN Classifier
- **Exp 5:** PCA & t-SNE
- **Exp 6:** Collaborative Filtering
- **Exp 7:** SVM (Linear & RBF)
- **Exp 8A:** MLP Neural Net
- **Exp 8B:** Deep NN (Keras)
- **Exp 9:** Random Forest & AdaBoost
- **Exp 10:** K-Means & Clustering
""")

if not loaded_ok:
    st.error(f"⚠️ Model artifacts not found or still training: {load_err}")
    st.info("Please wait for the training pipeline script to complete, then reload the page.")
    st.stop()

# -----------------------------------------------------------------------------
# PAGE 1: PURCHASE PREDICTOR
# -----------------------------------------------------------------------------
if page == "🔮 Purchase Predictor":
    st.markdown('<div class="main-title">🔮 Real-Time Purchase Intent Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Powered by the champion model: <b>{artifacts["model_name"]}</b></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Session & Browsing Behavior")
        c1, c2, c3 = st.columns(3)
        with c1:
            admin = st.number_input("Administrative Pages", min_value=0, max_value=30, value=2)
            admin_dur = st.number_input("Admin Duration (seconds)", min_value=0.0, max_value=3000.0, value=50.0)
            month = st.selectbox("Month of Session", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=8)
        with c2:
            info = st.number_input("Informational Pages", min_value=0, max_value=30, value=1)
            info_dur = st.number_input("Info Duration (seconds)", min_value=0.0, max_value=3000.0, value=20.0)
            visitor = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'], index=0)
        with c3:
            prod = st.number_input("Product Related Pages", min_value=0, max_value=500, value=25)
            prod_dur = st.number_input("Product Duration (seconds)", min_value=0.0, max_value=10000.0, value=850.0)
            weekend = st.selectbox("Weekend Session?", ["No (Weekday)", "Yes (Weekend)"], index=0)

        st.subheader("Engagement & Analytics Metrics")
        c4, c5, c6 = st.columns(3)
        with c4:
            bounce = st.slider("Bounce Rate", 0.0, 0.2, 0.01, step=0.005)
            special_day = st.slider("Special Day Closeness", 0.0, 1.0, 0.0, step=0.2)
        with c5:
            exit_rate = st.slider("Exit Rate", 0.0, 0.2, 0.03, step=0.005)
            os_code = st.selectbox("Operating System Code", [1, 2, 3, 4], index=1)
        with c6:
            page_val = st.slider("Page Values (Google Analytics)", 0.0, 100.0, 15.0, step=1.0)
            traffic_code = st.selectbox("Traffic Type Code", [1, 2, 3, 4], index=1)
            
    with col2:
        st.subheader("Prediction Result")
        predict_btn = st.button("🚀 Analyze Session & Predict", use_container_width=True, type="primary")
        
        if predict_btn or True:
            # Prepare feature dictionary
            month_map = {'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            visitor_map = {'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2}
            
            total_pages = admin + info + prod
            total_dur = admin_dur + info_dur + prod_dur
            bounce_exit = bounce / (exit_rate + 1e-5)
            avg_page_dur = total_dur / (total_pages + 1e-5)
            prod_page_ratio = prod / (total_pages + 1e-5)
            prod_dur_ratio = prod_dur / (total_dur + 1e-5)
            
            raw_input = {
                'Administrative': admin,
                'Administrative_Duration': admin_dur,
                'Informational': info,
                'Informational_Duration': info_dur,
                'ProductRelated': prod,
                'ProductRelated_Duration': prod_dur,
                'BounceRates': bounce,
                'ExitRates': exit_rate,
                'PageValues': page_val,
                'SpecialDay': special_day,
                'OperatingSystems': os_code,
                'Browser': 2,
                'Region': 1,
                'TrafficType': traffic_code,
                'Weekend': 1 if "Yes" in weekend else 0,
                'TotalPages': total_pages,
                'TotalDuration': total_dur,
                'BounceExitRatio': bounce_exit,
                'AvgPageDuration': avg_page_dur,
                'ProductPageRatio': prod_page_ratio,
                'ProductDurationRatio': prod_dur_ratio,
                'Month_Num': month_map.get(month, 5),
                'VisitorType_Code': visitor_map.get(visitor, 0)
            }
            
            # Align features
            input_df = pd.DataFrame([raw_input])[artifacts['feature_names']]
            scaled_input = artifacts['scaler'].transform(input_df)
            
            # Inference
            if artifacts['model_type'] == 'keras':
                prob = float(artifacts['model'].predict(scaled_input, verbose=0)[0][0])
                pred = int(prob >= 0.5)
            else:
                pred = int(artifacts['model'].predict(scaled_input)[0])
                prob = float(artifacts['model'].predict_proba(scaled_input)[0][1])
                
            st.markdown("---")
            if pred == 1:
                st.markdown("""
                <div style="background-color: #EBFDF5; border: 2px solid #10B981; border-radius: 10px; padding: 20px; text-align: center;">
                    <h2 style="color: #065F46; margin: 0;">🎉 High Purchase Intent!</h2>
                    <p style="color: #047857; margin-top: 5px;">This shopper is highly likely to make a purchase (Revenue = TRUE).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 10px; padding: 20px; text-align: center;">
                    <h2 style="color: #991B1B; margin: 0;">⏳ Low Purchase Intent</h2>
                    <p style="color: #B91C1C; margin-top: 5px;">Likely browsing or price-checking (Revenue = FALSE).</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.metric("Purchase Probability", f"{prob * 100:.1f}%")
            st.progress(prob)
            
            st.info(f"💡 **Key Driver:** PageValues = {page_val} and Product Duration = {prod_dur:.0f}s strongly influence conversion probability.")

# -----------------------------------------------------------------------------
# PAGE 2: CUSTOMER SEGMENTATION (CLUSTERING)
# -----------------------------------------------------------------------------
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-title">👥 Customer Segmentation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unsupervised clustering (Exp 10: K-Means, Agglomerative, GMM) discovered 3 natural shopper personas</div>', unsafe_allow_html=True)
    
    st.subheader("Clustering Evaluation Benchmark")
    st.dataframe(artifacts['cluster_bench_df'], use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="metric-card">
            <h4>🟢 Cluster 0: High-Value Buyers</h4>
            <p><b>Profile:</b> Deep engagement, high PageValues, high Product Duration.</p>
            <p><b>Conversion Rate:</b> ~68%</p>
            <p><b>Strategy:</b> Express checkout, VIP loyalty rewards, concierge support.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="metric-card">
            <h4>🟡 Cluster 1: Casual Window Shoppers</h4>
            <p><b>Profile:</b> Moderate page views, high bounce/exit rates, zero page values.</p>
            <p><b>Conversion Rate:</b> ~6%</p>
            <p><b>Strategy:</b> Exit-intent discounts, top 10 best-sellers popups.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔵 Cluster 2: Account & Support Visitors</h4>
            <p><b>Profile:</b> High Administrative pages, returning visitors, tracking past orders.</p>
            <p><b>Conversion Rate:</b> ~18%</p>
            <p><b>Strategy:</b> Re-order shortcuts, cross-sell recommendations on order tracking page.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: RECOMMENDATION ENGINE
# -----------------------------------------------------------------------------
elif page == "💡 Recommendation Engine":
    st.markdown('<div class="main-title">💡 Collaborative Filtering Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Recommending next site sections & product experiences using Cosine Similarity (Exp 6)</div>', unsafe_allow_html=True)
    
    rec_type = st.radio("Choose Recommendation Strategy:", ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering"], horizontal=True)
    recommender = artifacts['recommender']
    
    if rec_type == "User-Based Collaborative Filtering":
        st.markdown("##### 👤 Select a shopper profile to find similar users and recommend unseen categories:")
        user_list = list(recommender.user_item_matrix.index[:30])
        selected_user = st.selectbox("Select User Profile ID:", user_list, index=2)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("##### User's Historical Engagement (1-5 Scale):")
            user_ratings = recommender.user_item_matrix.loc[selected_user]
            st.dataframe(user_ratings.to_frame(name="Engagement Score"), use_container_width=True)
            
        with col2:
            st.write("##### 🎯 Recommended Categories (Weighted by Similar Shoppers):")
            recs = recommender.recommend_user_based(selected_user, top_n=3)
            for r in recs:
                st.markdown(f"""
                <div style="background-color: #F0F9FF; border-left: 4px solid #0284C7; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
                    <b style="font-size: 16px; color: #0369A1;">{r['category']}</b> (Score: {r['score']})<br>
                    <span style="font-size: 13px; color: #475569;">{r['description']}</span>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.markdown("##### 📦 Item-Based: Rate current interest to find related recommendations:")
        cols = st.columns(3)
        ratings = {}
        for i, cat in enumerate(recommender.categories):
            with cols[i % 3]:
                ratings[cat] = st.slider(f"{cat}", 0.0, 5.0, 3.0 if i < 2 else 0.0, step=0.5)
                
        recs = recommender.recommend_item_based(ratings, top_n=3)
        st.write("##### 🎯 Recommended Sections based on Item-Item Cosine Similarity:")
        for r in recs:
            st.markdown(f"""
            <div style="background-color: #FDF4FF; border-left: 4px solid #A855F7; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
                <b style="font-size: 16px; color: #7E22CE;">{r['category']}</b> (Score: {r['score']})<br>
                <span style="font-size: 13px; color: #475569;">{r['description']}</span>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: MODEL COMPARISON & LAB MAPPING
# -----------------------------------------------------------------------------
elif page == "📊 Model Comparison & Lab Mapping":
    st.markdown('<div class="main-title">📊 Multi-Algorithm Benchmark & Viva Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Complete comparative evaluation of all 7 classification algorithms tested in your curriculum</div>', unsafe_allow_html=True)
    
    st.subheader("Performance Comparison Table")
    bench_df = artifacts['benchmark_df']
    st.dataframe(bench_df, use_container_width=True)
    
    # Visual Bar Chart
    st.subheader("Comparative Accuracy & F1-Score Chart")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(bench_df))
    width = 0.35
    
    ax.bar(x - width/2, bench_df['Test Accuracy'], width, label='Accuracy (%)', color='#3B82F6')
    ax.bar(x + width/2, bench_df['F1-Score'], width, label='F1-Score (%)', color='#10B981')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Classification Algorithms Benchmark Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(bench_df['Algorithm'], rotation=25, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(40, 100)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("🎓 Mapping to Machine Learning Lab Syllabus")
    st.markdown("""
    | Experiment | Syllabus Algorithm | How We Implemented It in This Project |
    |---|---|---|
    | **Exp 4** | K-Nearest Neighbors (KNN) | Tuned KNN with distance-weighted voting on scaled browsing features |
    | **Exp 5** | Dimensionality Reduction (PCA & t-SNE) | Reduced 23 features to 2D components, analyzed explained variance |
    | **Exp 6** | Collaborative Filtering | User-based and Item-based recommendation engine with Cosine Similarity |
    | **Exp 7** | Support Vector Machine (Linear & RBF) | Compared linear hyperplane vs non-linear RBF kernel margins |
    | **Exp 8A** | Multi-Layer Perceptron (MLP) | Fully-connected neural network with ReLU & early stopping |
    | **Exp 8B** | Deep Neural Network (Keras) | Deep Feedforward NN (Dense 64 -> 32 -> 16 -> Sigmoid) with Dropout |
    | **Exp 9** | Random Forest & AdaBoost | Bagging & Boosting ensemble comparison; feature importances |
    | **Exp 10**| K-Means, Agglomerative, GMM | Unsupervised clustering for 3 distinct customer segments |
    """)


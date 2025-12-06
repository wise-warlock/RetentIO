# FILE: app.py
import streamlit as st
import plotly.graph_objects as go
import sys
import os

# Import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.features import load_and_process_raw_data, format_currency
from src.inference import RealTimeInference
from src.agent import decide_strategy, generate_message

# --- CONFIG ---
st.set_page_config(page_title="RetentIO Pro", layout="wide", page_icon="🏦")
st.title("🏦 RetentIO: AI-Powered Retention System")

# --- 1. LOAD DATA ---
RAW_DATA_PATH = './data/bank_churn_dataset_80k.csv'

with st.spinner('🔄 Loading Data & Initializing AI Engine...'):
    df = load_and_process_raw_data(RAW_DATA_PATH)

if df is None:
    st.error(f"Không tìm thấy file {RAW_DATA_PATH}")
    st.stop()

engine = RealTimeInference(df)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🔍 Control Panel")
    selected_user_str = st.selectbox("Chọn Khách hàng:", df['display_name'].head(100))
    selected_id = selected_user_str.split(" - ")[0]
    st.success("System Status: Online 🟢")
    st.caption("Model: LightGBM + Gemini")

# --- 3. MAIN DASHBOARD ---
profile = engine.predict_customer_metrics(selected_id)

if profile:
    info = profile['info']
    metrics = profile['metrics']
    
    # INFO CARD
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Họ tên", info['name'])
    c2.metric("Segment", info['segment_rfm'])
    c3.metric("Số dư", format_currency(info['balance']))
    c4.metric("Nghề nghiệp", info['occupation'])
    st.divider()
    
    # METRICS VISUALIZATION
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⚠️ Churn Probability")
        # Sử dụng kết quả từ Model thật
        churn_val = metrics['churn_prob'] * 100
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = churn_val,
            title = {'text': "Dự báo bởi LightGBM"},
            gauge = {'axis': {'range': [0, 100]}, 
                     'bar': {'color': "darkred" if churn_val > 50 else "green"}}
        ))
        fig.update_layout(height=200, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
        
    with col2:
        st.subheader("💎 Customer Lifetime Value")
        st.metric("CLV Dự báo", format_currency(metrics['clv']))
        st.info(f"Hạng: **{metrics['clv_segment']}**")
        
    with col3:
        st.subheader("🚀 Uplift Strategy")
        st.metric("Uplift Score", f"{metrics['uplift_score']:.2f}")
        st.caption(f"Chiến lược: **{metrics['uplift_segment']}**")
        
    st.divider()
    
    # AI AGENT ACTION
    st.subheader("🤖 GenAI Agent Activation")
    action, msg_header, color = decide_strategy(metrics)
    
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        if color == 'red': st.error(f"### {msg_header}")
        elif color == 'orange': st.warning(f"### {msg_header}")
        elif color == 'green': st.success(f"### {msg_header}")
        else: st.info(f"### {msg_header}")
        
    with ac2:
        st.markdown("### 📝 Nội dung đề xuất (Powered by Gemini):")
        # Gọi Gemini thật
        with st.spinner("Gemini đang soạn thảo nội dung..."):
            content = generate_message(action, info)
            st.text_area("Draft Content", value=content, height=120)
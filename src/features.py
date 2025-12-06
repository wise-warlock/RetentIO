# FILE: src/features.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_and_process_raw_data(filepath):
    """
    ETL Pipeline: Đọc Raw Data -> Clean -> Feature Engineering -> RFM
    """
    try:
        # 1. Load Data
        df = pd.read_csv(filepath)
        
        # 2. Xử lý ngày tháng (Data Cleaning)
        date_cols = ['created_date', 'last_active_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        
        # 3. Feature Engineering (Tạo đặc trưng mới)
        # Giả định ngày chốt dữ liệu là ngày active cuối cùng + 1
        snapshot_date = df['last_active_date'].max() + pd.Timedelta(days=1)
        
        # Recency: Số ngày chưa quay lại
        df['Recency'] = (snapshot_date - df['last_active_date']).dt.days
        
        # Tenure: Thời gian gắn bó (tháng)
        df['Tenure_Months'] = ((df['last_active_date'] - df['created_date']).dt.days / 30).fillna(0).astype(int)
        
        # 4. Tính toán RFM Score (Logic thay thế K-Means để chạy nhanh trên App)
        # Chia nhóm theo tứ phân vị (Quantile)
        # R_Score: 4 là tốt nhất (mới quay lại), 1 là tệ nhất
        df['R_Score'] = pd.qcut(df['Recency'], q=4, labels=[4, 3, 2, 1]).astype(int)
        
        # F_Score: Dùng engagement_score. 4 là cao nhất.
        df['F_Score'] = pd.qcut(df['engagement_score'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        
        # M_Score: Dùng balance. 4 là giàu nhất.
        df['M_Score'] = pd.qcut(df['balance'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        
        # 5. Phân khúc (Segmentation Logic)
        df['Segment_Label'] = df.apply(assign_segment, axis=1)
        
        # 6. Tạo display name cho UI
        df['id'] = df['id'].astype(str)
        df['display_name'] = df['id'] + " - " + df['full_name']
        
        return df

    except FileNotFoundError:
        return None

def assign_segment(row):
    """Gán nhãn phân khúc dựa trên RFM Score"""
    rfm_sum = row['R_Score'] + row['F_Score'] + row['M_Score']
    
    if rfm_sum >= 11: return 'VIP/Champions'
    elif row['R_Score'] == 1: return 'Hibernating (Ngủ đông)'
    elif row['R_Score'] == 2 and row['M_Score'] >= 3: return 'At Risk (Cần cứu)'
    elif row['F_Score'] >= 3: return 'Loyal Customers'
    else: return 'Mass/Standard'

def format_currency(value):
    if value >= 1_000_000_000: return f"{value/1_000_000_000:.2f} Tỷ"
    elif value >= 1_000_000: return f"{value/1_000_000:.1f} Tr"
    else: return f"{value:,.0f} đ"
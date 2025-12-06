# FILE: src/inference.py
import pandas as pd
import joblib
import os

class RealTimeInference:
    def __init__(self, df):
        self.df = df
        # Load Model Churn từ file .pkl
        model_path = os.path.join(os.path.dirname(__file__), '../models/churn_lgbm.pkl')
        try:
            self.churn_model = joblib.load(model_path)
            print("✅ Đã load LightGBM Model thành công!")
        except FileNotFoundError:
            print("❌ Lỗi: Không tìm thấy file 'models/churn_lgbm.pkl'. Hãy chạy script train trước!")
            self.churn_model = None

    def predict_customer_metrics(self, customer_id):
        user_row = self.df[self.df['id'] == customer_id]
        if user_row.empty: return None
        
        profile = user_row.iloc[0]

        # --- 1. CHURN PREDICTION ---
        churn_prob = 0.0
        if self.churn_model:
            # Tạo DataFrame input đúng với format lúc train
            input_data = pd.DataFrame([{
                'Recency_Days': profile['Recency'],
                'age': profile['age'],
                'credit_sco': profile['credit_sco'],
                'balance': profile['balance'],
                'nums_service': profile['nums_service'],
                'risk_score': profile['risk_score'],
                'occupation': profile['occupation']
            }])
            
            # Convert occupation sang category để model hiểu
            input_data['occupation'] = input_data['occupation'].astype('category')
            
            # Dự báo (lấy xác suất của class 1 - Churn)
            churn_prob = self.churn_model.predict_proba(input_data)[0][1]
        
        # --- 2. CALCULATE CLV (Rule-based Simulation) ---
        # (Để đơn giản, ta vẫn dùng logic tính toán nhanh cho CLV vì chưa lưu model Gamma-Gamma)
        clv_12m = profile['balance'] * 0.15 * (1 - churn_prob)
        clv_segment = 'High Value' if clv_12m > 50_000_000 else ('Mid Value' if clv_12m > 10_000_000 else 'Low Value')

        # --- 3. UPLIFT & STRATEGY ---
        # Logic Uplift dựa trên Churn Prob thực tế từ Model
        if 0.4 <= churn_prob <= 0.8: # Nhóm rủi ro trung bình cao -> Dễ cứu
            uplift_segment = 'Persuadables'
            uplift_score = 0.15
        elif churn_prob < 0.4:
            uplift_segment = 'Sure Things'
            uplift_score = 0.02
        else: # Churn > 0.8
            uplift_segment = 'Sleeping Dogs' # Hoặc Lost Causes tùy strategy
            uplift_score = -0.05

        return {
            'info': {
                'name': profile['full_name'],
                'id': profile['id'],
                'occupation': profile['occupation'],
                'age': profile['age'],
                'balance': profile['balance'],
                'segment_rfm': profile['Segment_Label'],
                'recency': profile['Recency']
            },
            'metrics': {
                'churn_prob': churn_prob,
                'clv': clv_12m,
                'clv_segment': clv_segment,
                'uplift_score': uplift_score,
                'uplift_segment': uplift_segment
            }
        }
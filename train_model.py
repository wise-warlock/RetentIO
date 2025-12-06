# File: train_and_save_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import os

# 1. Load dữ liệu
print("⏳ Đang load dữ liệu...")
df = pd.read_csv('./data/bank_churn_dataset_80k.csv')

# 2. Xử lý dữ liệu
df['created_date'] = pd.to_datetime(df['created_date'], dayfirst=True, errors='coerce')
df['last_active_date'] = pd.to_datetime(df['last_active_date'], dayfirst=True, errors='coerce')
snapshot_date = df['last_active_date'].max() + pd.Timedelta(days=1)

df['Recency_Days'] = (snapshot_date - df['last_active_date']).dt.days
df['occupation'] = df['occupation'].astype('category')
df['target'] = df['exit'].astype(int)

# Chọn features (phải nhớ danh sách này để dùng trong inference.py)
features = ['Recency_Days', 'age', 'credit_sco', 'balance', 'nums_service', 'risk_score', 'occupation']
X = df[features]
y = df['target']

# 3. Train Model
print("⚙️ Đang train LightGBM Model...")
clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
clf.fit(X, y)

# 4. Lưu Model
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(clf, 'models/churn_lgbm.pkl')
print("✅ Đã lưu model thành công tại: models/churn_lgbm.pkl")
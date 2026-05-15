import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


df = pd.read_csv('paddydataset.csv')
df.columns = df.columns.str.strip()

MEDIAN = df['Paddy yield(in Kg)'].median()
df['target'] = (df['Paddy yield(in Kg)'] >= MEDIAN).astype(int)
print(f"Target (rendimiento alto = ≥ {MEDIAN:.0f} Kg): {df['target'].value_counts().to_dict()}")


CAT_COLS = [
    'Agriblock', 'Variety', 'Soil Types', 'Nursery',
    'Wind Direction_D1_D30', 'Wind Direction_D31_D60',
    'Wind Direction_D61_D90', 'Wind Direction_D91_D120'
]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}
    print(f"  {col}: {list(encoders[col].keys())}")



FEATURE_COLS = [c for c in df.columns if c not in ['Paddy yield(in Kg)', 'target']]
X = df[FEATURE_COLS].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)




print("\n[1/2] Entrenando Regresión Logística...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)

print("[2/2] Entrenando Red Neuronal (64-32)...")
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn.fit(X_train_s, y_train)
nn_pred = nn.predict(X_test_s)



def get_metrics(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "name": name,
        "accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
        "precision": round(precision_score(y_true, y_pred) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred)    * 100, 2),
        "f1":        round(f1_score(y_true, y_pred)        * 100, 2),
        "confusion_matrix": cm,
    }
    print(f"\n{name}")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v}%")
    return metrics

lr_metrics = get_metrics(y_test, lr_pred, "Regresión Logística")
nn_metrics = get_metrics(y_test, nn_pred, "Red Neuronal Artificial")




model_data = {
    "feature_cols":    FEATURE_COLS,
    "cat_encoders":    encoders,
    "scaler_mean":     scaler.mean_.tolist(),
    "scaler_scale":    scaler.scale_.tolist(),
    "lr_coef":         lr.coef_.tolist(),
    "lr_intercept":    lr.intercept_.tolist(),
    "nn_coefs":        [c.tolist() for c in nn.coefs_],
    "nn_intercepts":   [i.tolist() for i in nn.intercepts_],
    "median_yield":    float(MEDIAN),
    "lr_metrics":      lr_metrics,
    "nn_metrics":      nn_metrics,
}

js_content = f"// Auto-generated — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n"
js_content += f"const MODEL_DATA = {json.dumps(model_data)};\n"

import os
os.makedirs('models', exist_ok=True)
with open('models/model_data.js', 'w') as f:
    f.write(js_content)

print(f"\n Exportado a models/model_data.js ({len(js_content)//1024} KB)")
print("   Abre index.html con Live Server para usar la app.")

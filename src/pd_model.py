import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score

def train_pd(X_tr, y_tr, X_te, y_te):
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42
    )
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, auc

def save(model, path="models/pd_model.pkl"):
    joblib.dump(model, path)
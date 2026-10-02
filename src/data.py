from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    df["default"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns="class")
    for c in df.select_dtypes(["category", "object"]).columns:
        df[c] = df[c].astype("category")
    return df

def split(df):
    X, y = df.drop(columns="default"), df["default"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
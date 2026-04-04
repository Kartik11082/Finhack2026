import json, pandas as pd, os

print("=== model_comparison.json ===")
print(json.dumps(json.load(open("data/model_comparison.json")), indent=2))

print("\n=== feature_importance_enhanced.json ===")
content = open("data/feature_importance_enhanced.json").read()
print(f"Raw content: '{content}'")
print(f"Length: {len(content)} bytes")

print("\n=== dataset_enhanced.csv exists? ===")
print(os.path.exists("data/dataset_enhanced.csv"))
if os.path.exists("data/dataset_enhanced.csv"):
    e = pd.read_csv("data/dataset_enhanced.csv")
    print(e.shape)
    print(e.columns.tolist())
    print(e.head(3).to_string())

print("\n=== sentiment_raw.csv exists? ===")
print(os.path.exists("data/sentiment_raw.csv"))
if os.path.exists("data/sentiment_raw.csv"):
    s = pd.read_csv("data/sentiment_raw.csv")
    print(s.shape)
    print(s.head(3).to_string())

print("\n=== fold_results.json ===")
print(json.dumps(json.load(open("data/fold_results.json")), indent=2))

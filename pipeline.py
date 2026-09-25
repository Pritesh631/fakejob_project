import pandas as pd, numpy as np, re, json, joblib, os
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)

sns.set_style("whitegrid")
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{BASE}/plots", exist_ok=True)
os.makedirs(f"{BASE}/models", exist_ok=True)
os.makedirs(f"{BASE}/data", exist_ok=True)

# ---------- 1. LOAD + CLEAN ----------
df = pd.read_csv(f"{BASE}/data/job_postings.csv")
for c in ["company_profile", "requirements", "description", "title", "benefits"]:
    df[c] = df[c].fillna("")

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"#[a-z]+_[a-f0-9]+#", " ", t)   # anonymized #URL_hash#, #EMAIL_hash#, #PHONE_hash# tokens
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"&\w+;", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = (df.title + " " + df.company_profile + " " + df.description + " " +
              df.requirements + " " + df.benefits).apply(clean_text)

# ---------- 2. EDA ----------
class_counts = df.fraudulent.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Genuine", "Fraudulent"], class_counts.values, color=["#4C6EF5", "#F03E3E"])
ax.set_title("Class Distribution (Imbalanced)")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+5, str(int(b.get_height())), ha="center")
plt.tight_layout(); plt.savefig(f"{BASE}/plots/class_distribution.png", dpi=110); plt.close()

df["text_len"] = df.text.apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(6, 4))
sns.kdeplot(df[df.fraudulent==0].text_len, label="Genuine", fill=True, ax=ax, color="#4C6EF5")
sns.kdeplot(df[df.fraudulent==1].text_len, label="Fraudulent", fill=True, ax=ax, color="#F03E3E")
ax.set_title("Posting Text Length Distribution"); ax.set_xlabel("Word count"); ax.legend()
plt.tight_layout(); plt.savefig(f"{BASE}/plots/text_length.png", dpi=110); plt.close()

feat_flags = ["telecommuting","has_company_logo","has_questions"]
rates = df.groupby("fraudulent")[feat_flags].mean().T
fig, ax = plt.subplots(figsize=(6,4))
rates.plot(kind="bar", ax=ax, color=["#4C6EF5","#F03E3E"])
ax.set_xticklabels(["Telecommuting","Has Logo","Has Questions"], rotation=0)
ax.set_ylabel("Proportion = 1"); ax.set_title("Structured Feature Rates by Class")
ax.legend(["Genuine","Fraudulent"])
plt.tight_layout(); plt.savefig(f"{BASE}/plots/structured_features.png", dpi=110); plt.close()

# ---------- 3. TF-IDF + SPLIT ----------
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df.text, df.fraudulent, test_size=0.25, random_state=42, stratify=df.fraudulent)

tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,1), min_df=5, stop_words="english")
X_train = tfidf.fit_transform(X_train_txt)
X_test = tfidf.transform(X_test_txt)

# ---------- 4. TRAIN + BENCHMARK MODELS ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight="balanced", max_iter=5000),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42),
}

results = []
roc_data = {}
best_model, best_f1, best_name = None, -1, None
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:,1]
    else:
        scores = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))
    if f1 > best_f1:
        best_f1, best_model, best_name = f1, clf, name

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_df.to_csv(f"{BASE}/models/benchmark_results.csv", index=False)
print(results_df)

# metrics bar chart
fig, ax = plt.subplots(figsize=(8,5))
melt = results_df.melt(id_vars="model", value_vars=["accuracy","precision","recall","f1"])
sns.barplot(data=melt, x="model", y="value", hue="variable", ax=ax)
ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_xlabel("")
ax.set_title("Model Benchmark Comparison")
plt.xticks(rotation=20, ha="right"); plt.legend(title="", loc="lower right")
plt.tight_layout(); plt.savefig(f"{BASE}/plots/model_comparison.png", dpi=110); plt.close()

# confusion matrix for best model
best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(4.5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine","Fraud"], yticklabels=["Genuine","Fraud"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout(); plt.savefig(f"{BASE}/plots/confusion_matrix.png", dpi=110); plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(6,5))
for name,(fpr,tpr,a) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={a:.2f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{BASE}/plots/roc_curves.png", dpi=110); plt.close()

# ---------- 5. EXPORT LIGHTWEIGHT LOGISTIC MODEL FOR FRONTEND DEMO ----------
lr = models["Logistic Regression"]
coefs = lr.coef_[0]
feat_names = tfidf.get_feature_names_out()
top_n = 30
# drop overly generic stopword-like leftovers that carry no real signal
leak_words = {"com","www","https","http"}

sorted_idx = np.argsort(coefs)[::-1]
fraud_keywords, genuine_keywords = {}, {}
for i in sorted_idx:
    w = feat_names[i]
    if w in leak_words: continue
    if len(fraud_keywords) < top_n: fraud_keywords[w] = round(float(coefs[i]),3)
for i in sorted_idx[::-1]:
    w = feat_names[i]
    if w in leak_words: continue
    if len(genuine_keywords) < top_n: genuine_keywords[w] = round(float(coefs[i]),3)

# downsample ROC curves to ~14 points each for compact export
roc_export = {}
for name, (fpr, tpr, a) in roc_data.items():
    idxs = np.linspace(0, len(fpr)-1, min(14, len(fpr))).astype(int)
    roc_export[name] = {"fpr": [round(float(fpr[i]),3) for i in idxs],
                         "tpr": [round(float(tpr[i]),3) for i in idxs],
                         "auc": round(float(a),3)}

# text length histogram (5 bins) per class
bins = np.linspace(df.text_len.min(), df.text_len.max(), 9)
hist_g, _ = np.histogram(df[df.fraudulent==0].text_len, bins=bins)
hist_f, _ = np.histogram(df[df.fraudulent==1].text_len, bins=bins)

cm_best = confusion_matrix(y_test, best_preds).tolist()

export = {
    "bias": round(float(lr.intercept_[0]),3),
    "fraud_keywords": fraud_keywords,
    "genuine_keywords": genuine_keywords,
    "benchmark": results_df.to_dict(orient="records"),
    "best_model": best_name,
    "class_counts": {"genuine": int(class_counts[0]), "fraudulent": int(class_counts[1])},
    "dataset_size": int(len(df)),
    "test_size": int(len(y_test)),
    "feature_rates": {
        "genuine": {k: round(float(v),3) for k,v in rates[0].to_dict().items()},
        "fraudulent": {k: round(float(v),3) for k,v in rates[1].to_dict().items()},
    },
    "text_len_hist": {"bins": [round(float(b),0) for b in bins], "genuine": hist_g.tolist(), "fraudulent": hist_f.tolist()},
    "confusion_matrix": cm_best,
    "roc": roc_export,
    "tfidf_vocab_size": len(feat_names),
    "train_size": int(len(y_train)),
}
with open(f"{BASE}/models/frontend_export.json","w") as f:
    json.dump(export, f, indent=2)

joblib.dump({"tfidf":tfidf, "models":models}, f"{BASE}/models/trained_models.joblib")
print("Best model:", best_name, "F1:", round(best_f1,3))
print("Exported frontend JSON with", len(fraud_keywords), "fraud keywords")

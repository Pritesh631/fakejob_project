# Intelligent Fake Job Posting Detection

A machine learning classification system that flags fraudulent job postings using
TF-IDF text features and structured job attributes.

## Project structure
```
fetch_data.py         # downloads the real EMSCAD dataset (data/job_postings.csv)
pipeline.py            # cleaning, EDA, TF-IDF, model training + benchmarking
data/job_postings.csv  # EMSCAD: 17,880 postings, 866 fraudulent (4.8%)
plots/                  # EDA & evaluation charts (PNG)
models/
  benchmark_results.csv   # accuracy / precision / recall / F1 per model
  frontend_export.json    # compact model weights + stats used by the web page
  trained_models.joblib   # fitted TF-IDF vectorizer + all 5 classifiers
webpage/
  index.html              # the deployable web app — the trained model's
                           # weights are baked in, no backend/server needed
```

## Run the ML pipeline
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python fetch_data.py
python pipeline.py
```

## Deploy the web page
`webpage/index.html` is a single self-contained file (HTML + CSS + JS, model
weights embedded as JSON) with no server, API, or build step required. Any
static host works:

- **Netlify (fastest):** go to app.netlify.com/drop and drag `index.html` in —
  you get a live public URL immediately, no account needed.
- **GitHub Pages:** create a repo, upload `index.html` (rename is optional,
  GitHub Pages looks for `index.html` at the root), then enable
  Settings → Pages → Deploy from branch → `main` / root.
- **Vercel:** `npm i -g vercel`, then run `vercel --prod` inside the `webpage/`
  folder.
- **Anywhere else:** Cloudflare Pages, Surge, S3 + static hosting, or your own
  server — just serve the file.

To refresh the page with a newly retrained model: rerun `pipeline.py`, then
copy the `bias`, `fraud_keywords`, `genuine_keywords`, and `benchmark` fields
from the regenerated `models/frontend_export.json` into the `const D = {...}`
block near the bottom of `webpage/index.html`.


## Approach
1. **Clean** — lowercase, strip anonymized `#URL_hash#`/`#EMAIL_hash#` tokens (EMSCAD
   redacts contact info this way), HTML entities and punctuation, and merge title +
   company profile + description + requirements + benefits into one text field.
2. **EDA** — class imbalance (4.8% fraud), posting length distribution, and how
   structured flags (company logo, remote, screening questions) differ by class.
3. **Feature extraction** — TF-IDF (unigrams, top 4,000 terms, stopwords removed).
4. **Modeling** — Logistic Regression, Random Forest, Multinomial Naive Bayes,
   Linear SVM, Gradient Boosting — all class-weighted where supported, trained
   on a 75/25 stratified split.
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix and ROC/AUC,
   since accuracy alone is misleading on a ~4.8%-fraud-rate dataset.

## Dataset: EMSCAD
The **Employment Scam Aegean Dataset (EMSCAD)** was published by the University of
the Aegean, Laboratory of Information & Communication Systems Security
(http://emscad.samos.aegean.gr), and is distributed on Kaggle as
["\[Real or Fake\] Fake JobPosting Prediction"](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
It contains 17,880 real job postings, 866 of which (4.8%) are confirmed fraudulent.
`fetch_data.py` pulls the same file from a public GitHub mirror so the project runs
without Kaggle credentials; swap in your own copy of the CSV (same 18 columns) to
use a different source.

## Results (this run)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear SVM | 97.9% | 75.6% | 81.9% | **78.7%** |
| Logistic Regression | 97.2% | 65.6% | 87.5% | 75.0% |
| Gradient Boosting | 97.7% | 94.4% | 54.6% | 69.2% |
| Random Forest | 97.8% | 98.4% | 55.6% | 71.0% |
| Naive Bayes | 96.7% | 98.6% | 32.0% | 48.3% |

Linear SVM gives the best precision/recall balance (F1); Random Forest and Naive
Bayes are far more conservative (high precision, lower recall) — worth tuning the
decision threshold or resampling (e.g. SMOTE) depending on whether false positives
or missed fraud matter more for your use case.


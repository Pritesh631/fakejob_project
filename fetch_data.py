"""
Downloads the real Employment Scam Aegean Dataset (EMSCAD).

Original source: University of the Aegean, Laboratory of Information &
Communication Systems Security — http://emscad.samos.aegean.gr
Distributed on Kaggle as: "[Real or Fake] Fake JobPosting Prediction"
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
(17,880 postings; 866 fraudulent — a ~4.8% fraud rate)

If you have Kaggle API credentials configured, you can instead run:
    kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction

This script pulls the same file from a public GitHub mirror so the project
runs without any credentials.
"""
import urllib.request, os

URL = "https://raw.githubusercontent.com/PJDEEPESH/Fake-Job-Prediction/main/data/fake_job_postings.csv"
OUT = os.path.join(os.path.dirname(__file__), "data", "job_postings.csv")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(f"Downloading EMSCAD dataset from {URL} ...")
urllib.request.urlretrieve(URL, OUT)
print(f"Saved to {OUT}")

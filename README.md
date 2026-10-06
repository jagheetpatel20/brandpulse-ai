# BrandPulse AI 🔍
### Twitter Sentiment Analysis – NLP Pipeline & Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09A3D5?logo=spacy)](https://spacy.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)

> End-to-end sentiment-analysis system for Twitter / airline reviews using classical ML (TF-IDF + Logistic Regression) and a deep-learning LSTM model, with an interactive Streamlit dashboard.

---

## 📁 Repository Structure

```
brandpulse-ai/
│
├── data/
│   └── Tweets.csv                        # Raw Kaggle dataset (see below)
│
├── models/
│   ├── classical_tfidf_pipeline.joblib   # Trained TF-IDF + classifier pipeline
│   ├── lstm_model.keras                  # Trained Keras LSTM model
│   └── tokenizer.json                    # Keras Tokenizer vocabulary
│
├── notebooks/
│   ├── 01_Preprocessing_and_Classical.ipynb  # EDA, preprocessing & classical ML
│   └── 02_DeepLearning_LSTM.ipynb            # LSTM training & evaluation
│
├── src/
│   └── preprocessing.py                  # ⭐ Core NLP preprocessing module
│
├── app.py                                # Streamlit dashboard
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ⚙️ Environment Setup

### 1 – Clone / open the repository

```bash
# If using git
git clone <your-repo-url>
cd brandpulse-ai
```

### 2 – Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 – Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 – Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5 – Download NLTK corpora (first-run auto-download, but you can pre-fetch)

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## 📦 Dataset – Twitter US Airline Sentiment

The project uses the **[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)** dataset (CrowdFlower / Figure Eight, ~14,640 tweets).

### Download via Kaggle CLI

```bash
# Install the Kaggle API client (already in requirements.txt)
pip install kaggle

# Place your kaggle.json API token in:
#   Windows: %USERPROFILE%\.kaggle\kaggle.json
#   macOS/Linux: ~/.kaggle/kaggle.json

kaggle datasets download -d crowdflower/twitter-airline-sentiment \
      --unzip -p data/
```

After extraction, rename the file if needed:

```bash
# The archive extracts as 'Tweets.csv' – confirm path:
ls data/Tweets.csv
```

### Manual Download

1. Visit → <https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>
2. Click **Download** → extract the ZIP.
3. Copy `Tweets.csv` into the `data/` directory.

### Key Columns

| Column | Description |
|---|---|
| `airline_sentiment` | Label: `positive`, `neutral`, `negative` |
| `text` | Raw tweet text |
| `airline` | Airline name (United, Delta, American, …) |
| `tweet_created` | Timestamp of the tweet |

---

## 🚀 Quickstart – Preprocessing Module

```python
import pandas as pd
from src.preprocessing import TextPreprocessor, preprocess_dataframe

# ── Single string ──────────────────────────────────────────────────
tp = TextPreprocessor()

tp.clean("@AmericanAir your #service is TERRIBLE!! 😤 http://t.co/abc123")
# → 'american air service terrible'

tp.clean(None)   # → ''
tp.clean("   ")  # → ''

# ── pandas Series ─────────────────────────────────────────────────
from src.preprocessing import preprocess_series

series = pd.Series([
    "@Delta #worst flight ever",
    "I actually love @SouthwestAir 😊",
    None,
    "  ",
])
print(preprocess_series(series))
# 0       worst flight
# 1      actually love
# 2
# 3
# dtype: object

# ── Full DataFrame ────────────────────────────────────────────────
df = pd.read_csv("data/Tweets.csv")
df = preprocess_dataframe(df, text_col="text", out_col="clean_text")
print(df[["text", "clean_text"]].head(3))
```

---

## 🧠 Model Training

Run the notebooks in order:

```bash
# Classical ML (TF-IDF + Logistic Regression / SVM)
jupyter notebook notebooks/01_Preprocessing_and_Classical.ipynb

# Deep Learning (LSTM with Keras)
jupyter notebook notebooks/02_DeepLearning_LSTM.ipynb
```

Trained models are saved automatically to `models/`.

---

## 📊 Streamlit Dashboard

```bash
streamlit run app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`).

---

## 🔧 Preprocessing Pipeline – Stage Reference

| Stage | What it does |
|---|---|
| Unicode normalisation | Strips accents, removes non-ASCII (emoji, CJK) |
| Lowercase | Converts all text to lower case |
| URL removal | Strips `http://`, `https://`, `www.*` |
| Handle removal | Removes `@username` tokens |
| Hashtag cleaning | `#word` → `word` (preserves keyword) |
| Number removal | Drops digit sequences |
| Special-char removal | Keeps only `[a-z ]` |
| Whitespace normalisation | Collapses spaces, strips edges |
| Stop-word removal | English stop-words (spaCy / NLTK) |
| Lemmatisation | Reduces tokens to dictionary form |

---

## 📝 License

This project is for educational and research purposes.  
Dataset: © CrowdFlower / Figure Eight – see [Kaggle Terms](https://www.kaggle.com/terms).

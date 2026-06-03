# 🎯 Sentiment Analysis on Trustpilot Marketing Reviews

> End-to-end NLP sentiment analysis on Trustpilot customer reviews for the advertising & digital marketing industry — built for **Candy Factory Group – Pannipitiya**.

---

## 📌 Project Overview

This project performs a complete NLP pipeline on **3,585 Trustpilot customer reviews** (after cleaning) from **214 companies** in the advertising and digital marketing industry. It was developed as a data science internship project for Candy Factory Group – Pannipitiya, a Sri Lankan advertising and digital marketing company.

The pipeline covers:
- Data cleaning (null handling, deduplication, outlier removal)
- Text preprocessing (tokenization, stop-word removal, lemmatization)
- Exploratory Data Analysis (EDA) with visualizations
- Traditional ML models (Logistic Regression, Naive Bayes, Linear SVM with TF-IDF)
- Pre-trained Transformer model (DistilBERT via HuggingFace)
- Model comparison report
- Theme extraction from positive and negative reviews
- Business recommendations for Candy Factory Group

---

## 📂 Repository Structure

```
sentiment-analysis-trustpilot-marketing/
├── data/
│   ├── raw/                        # trust_pilot_reviews_data_2022_06.csv
│   └── processed/                  # reviews_labelled.csv · reviews_cleaned.csv
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Text Preprocessing Pipeline
│   ├── 03_ml_models.ipynb          # Traditional ML Models (TF-IDF + LR/NB/SVM)
│   └── 04_transformer_model.ipynb  # DistilBERT Fine-tuning & Final Comparison
├── models/
│   ├── best_ml_model.pkl           # Best traditional ML model (Naive Bayes)
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│   └── distilbert_sentiment/       # Fine-tuned DistilBERT model + tokenizer
├── reports/
│   ├── figures/                    # All 14 visualisation PNGs
│   ├── ml_model_results.csv        # Traditional ML metrics
│   ├── final_comparison.csv        # Full model comparison table
│   ├── distilbert_training_history.csv
│   └── model_comparison.md         # Side-by-side model evaluation report
├── presentation/                   # Final presentation slides (.pptx)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [Trustpilot Reviews Dataset – Kaggle](https://www.kaggle.com/datasets/crawlfeeds/trustpilot-reviews-dataset)
- **File:** `trust_pilot_reviews_data_2022_06.csv`
- **Raw size:** 3,698 reviews | 214 companies | 11 columns
- **After cleaning:** 3,585 reviews (113 removed — nulls, duplicates, outliers)
- **Key columns:** `review_text`, `review_title`, `rating`, `author_name`, `name` (company), `reviewed_at`

### Sentiment Label Mapping

| Star Rating | Sentiment Label | Count | Share |
|---|---|---|---|
| ⭐⭐⭐⭐⭐ (4–5) | Positive | 2,873 | 80.1% |
| ⭐⭐⭐ (3) | Neutral | 64 | 1.8% |
| ⭐⭐ (1–2) | Negative | 648 | 18.1% |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| NLP Preprocessing | NLTK, spaCy (en_core_web_sm) |
| ML Models | scikit-learn — TF-IDF, Logistic Regression, Naive Bayes, Linear SVM |
| Transformer | HuggingFace Transformers, DistilBERT, PyTorch |
| Visualization | Matplotlib, Seaborn, WordCloud, Plotly |
| Notebooks | Jupyter Notebook / Google Colab |
| Version Control | Git, GitHub |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/nisansala/sentiment-analysis-trustpilot-marketing.git
cd sentiment-analysis-trustpilot-marketing

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows (PowerShell)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK and spaCy data
python -m nltk.downloader stopwords wordnet punkt punkt_tab
python -m spacy download en_core_web_sm

# 5. Place the dataset
# Copy trust_pilot_reviews_data_2022_06.csv into data/raw/

# 6. Create required folders
mkdir reports\figures             # Windows
mkdir -p reports/figures          # macOS/Linux
mkdir models

# 7. Launch Jupyter
jupyter notebook
```

> **Note for Windows users:** When running `04_transformer_model.ipynb`, ensure `num_workers=0` in DataLoader and `BATCH_SIZE=8` to avoid memory errors on CPU.

---

## 🔬 Pipeline Summary

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Rating & sentiment distribution charts
- Review length analysis by sentiment class
- Word clouds for positive vs. negative reviews (stop-words removed)
- Top 10 companies by review volume
- Saved output: `data/processed/reviews_labelled.csv`

### 2. Text Preprocessing (`02_preprocessing.ipynb`)
- **Data cleaning:** null handling, duplicate removal, outlier removal (< 3 or > 300 words)
- Lowercasing, URL/HTML/punctuation removal
- Tokenisation (NLTK `word_tokenize`)
- Stop-word removal (NLTK + custom stops)
- Lemmatisation (spaCy `en_core_web_sm`)
- Label encoding: Negative=0, Neutral=1, Positive=2
- Saved output: `data/processed/reviews_cleaned.csv`

### 3. Traditional ML Models (`03_ml_models.ipynb`)
- TF-IDF vectorisation (unigrams + bigrams, max 10,000 features)
- Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- `class_weight='balanced'` to handle class imbalance
- 5-fold stratified cross-validation
- Evaluation: Accuracy, Precision, Recall, F1 Macro, Confusion Matrix

### 4. Transformer Model (`04_transformer_model.ipynb`)
- Model: `distilbert-base-uncased` via HuggingFace
- Fine-tuning on original `review_text` (3 epochs, CPU-optimised)
- Full model comparison vs. traditional ML
- Inference function for new text prediction

---

## 📈 Model Comparison Results

| Model | Accuracy | F1 Macro | F1 Weighted | CV F1 Mean |
|---|---|---|---|---|
| Logistic Regression | 92.61% | 0.6021 | 0.9221 | 0.6035 ±0.005 |
| **Naive Bayes** ⭐ | **94.14%** | **0.6153** | **0.9330** | **0.6057 ±0.008** |
| Linear SVM | 92.61% | 0.5994 | 0.9200 | 0.5993 ±0.007 |
| DistilBERT | 94.00% | 0.6110 | 0.9313 | N/A (CPU) |

> ⭐ **Best model:** Naive Bayes — highest accuracy and macro F1. Recommended for CPU production deployment.
>
> ⚠️ **Note on F1 Macro:** Moderate scores across all models are caused by the severely underrepresented Neutral class (only 64 reviews, 1.8%). This is an inherent dataset challenge, not a modelling failure.

---

## 💡 Key Findings

**Positive review themes:** service · excellent · recommend · helpful · professional · fast · quality

**Negative review themes:** no response · refund · money · email · never · bad · delay · charge

**Insight:** Negative reviews are **2.3× longer** on average (80 words) than positive reviews (35 words), indicating dissatisfied customers invest more detail in their complaints — a rich source of actionable feedback.

---

## 📋 Business Recommendations

| Priority | Recommendation | Timeline |
|---|---|---|
| 🔴 Immediate | Set 24-hour email response SLA | 0–30 days |
| 🔴 Immediate | Publish transparent refund & billing policy | 0–30 days |
| 🔴 Immediate | Deploy Naive Bayes sentiment monitoring dashboard | 0–30 days |
| 🟠 Short-Term | Use positive review language in marketing copy | 1–3 months |
| 🟠 Short-Term | Launch post-project NPS survey | 1–3 months |
| 🔵 Long-Term | Retrain DistilBERT on GPU (target F1 > 0.75) | 3–12 months |
| 🔵 Long-Term | Augment Neutral class data and retrain models | 3–12 months |

Full analysis in `reports/model_comparison.md` and `presentation/`.

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `data/processed/reviews_cleaned.csv` | Final cleaned & labelled dataset |
| `models/best_ml_model.pkl` | Saved Naive Bayes model |
| `models/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `models/distilbert_sentiment/` | Fine-tuned DistilBERT model + tokenizer |
| `reports/final_comparison.csv` | Full model metrics comparison |
| `reports/figures/` | 14 visualisation PNG files |

---

## 👤 Author

**Nisansala Ruwan Pathirana**  
📅 June 2026

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: [CrawlFeeds – Trustpilot Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/crawlfeeds/trustpilot-reviews-dataset)
- [HuggingFace Transformers](https://huggingface.co/transformers/) — DistilBERT model
- [spaCy](https://spacy.io/) — Lemmatisation pipeline
- [NLTK](https://www.nltk.org/) — Tokenisation and stop-word removal
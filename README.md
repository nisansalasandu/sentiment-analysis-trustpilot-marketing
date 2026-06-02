# 🎯 Sentiment Analysis on Trustpilot Marketing Reviews

> End-to-end NLP sentiment analysis on Trustpilot customer reviews for the advertising & digital marketing industry — built for **Candy Factory Group – Pannipitiya**.

---

## 📌 Project Overview

This project performs a complete NLP pipeline on **3,698 Trustpilot customer reviews** from **214 companies** in the advertising and digital marketing industry. It was developed as a data science internship project for Candy Factory Group – Pannipitiya, a Sri Lankan advertising and digital marketing company.

The pipeline covers:
- Text preprocessing (tokenization, stop-word removal, lemmatization)
- Exploratory Data Analysis (EDA) with visualizations
- Traditional ML models (Logistic Regression, Naive Bayes, SVM with TF-IDF)
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
│   └── processed/                  # cleaned and labeled CSVs
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Text Preprocessing Pipeline
│   ├── 03_ml_models.ipynb          # Traditional ML Models
│   └── 04_transformer_model.ipynb  # DistilBERT Fine-tuning
├── src/
│   ├── preprocess.py               # Reusable preprocessing functions
│   ├── train_ml.py                 # ML training scripts
│   └── train_transformer.py        # Transformer training script
├── reports/
│   ├── figures/                    # Plots, word clouds, confusion matrices
│   └── model_comparison.md         # Side-by-side model evaluation report
├── presentation/                   # Final presentation slides
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [Trustpilot Reviews Dataset – Kaggle](https://www.kaggle.com/datasets/crawlfeeds/trustpilot-reviews-dataset)
- **File:** `trust_pilot_reviews_data_2022_06.csv`
- **Size:** 3,698 reviews | 214 companies | 11 columns
- **Key columns:** `review_text`, `review_title`, `rating`, `author_name`, `name` (company), `reviewed_at`

### Sentiment Label Mapping

| Star Rating | Sentiment Label |
|---|---|
| ⭐⭐⭐⭐⭐ (4–5) | Positive |
| ⭐⭐⭐ (3) | Neutral |
| ⭐⭐ (1–2) | Negative |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| NLP Preprocessing | NLTK, spaCy |
| ML Models | scikit-learn (LR, NB, SVM, TF-IDF) |
| Transformer | HuggingFace Transformers, DistilBERT, PyTorch |
| Visualization | Matplotlib, Seaborn, WordCloud, Plotly |
| Notebooks | Jupyter Notebook / Google Colab |
| Version Control | Git, GitHub |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sentiment-analysis-trustpilot-marketing.git
cd sentiment-analysis-trustpilot-marketing

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Copy trust_pilot_reviews_data_2022_06.csv into data/raw/

# 5. Launch Jupyter
jupyter notebook
```

---

## 🔬 Pipeline Summary

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Rating distribution, class balance
- Review length analysis
- Word clouds for positive vs. negative reviews
- Top companies by review volume

### 2. Text Preprocessing (`02_preprocessing.ipynb`)
- Lowercasing, punctuation removal
- Tokenization (NLTK)
- Stop-word removal
- Lemmatization (spaCy)
- Sentiment label assignment from ratings

### 3. Traditional ML Models (`03_ml_models.ipynb`)
- TF-IDF vectorization
- Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 4. Transformer Model (`04_transformer_model.ipynb`)
- Model: `distilbert-base-uncased` via HuggingFace
- Fine-tuning on preprocessed review text
- Comparison with traditional ML results

---

## 📈 Model Comparison

| Model | Accuracy | F1-Score (macro) |
|---|---|---|
| Logistic Regression | TBD | TBD |
| Naive Bayes | TBD | TBD |
| SVM | TBD | TBD |
| DistilBERT | TBD | TBD |

*(Results to be updated after training)*

---

## 💡 Business Recommendations

Key findings and recommendations for Candy Factory Group – Pannipitiya are documented in `reports/model_comparison.md` and the final presentation in `presentation/`.

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
- HuggingFace Transformers library

# Sentiment Analysis of Amazon Fine Food Reviews

## Project Overview
This small-scale project performs sentiment analysis on the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). We explore two approaches:

1. **Rule-based Sentiment Analysis** using VADER (Valence Aware Dictionary and sEntiment Reasoner).
2. **Pretrained Transformer-based Classification** using Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` pipeline.

The goal is to compare basic rule-based sentiment scoring with a modern deep-learning approach on user review texts.

## Dataset
- **Source**: Kaggle — Amazon Fine Food Reviews
- **Size**: ~568,000 reviews
- **Key Fields**:
  - `Id`: Review identifier
  - `ProductId`, `UserId`, `ProfileName`
  - `Score`: Star rating (1–5)
  - `Time` (UNIX timestamp)
  - `Summary`, `Text`: Review title and body

## Repository Structure
```
tonycode2-sentiment/
├── README.md               # (This file)
├── Sentiment_Analysis.ipynb
└── reviews.csv             # Dataset (download and place here)
```

## Requirements
- Python 3.8+
- pandas
- numpy
- nltk
- vaderSentiment
- transformers
- scikit-learn

Install via:
```bash
pip install pandas numpy nltk vaderSentiment transformers scikit-learn
```

## Data Preprocessing
1. **Load Data**: Read `reviews.csv` into a Pandas DataFrame.
2. **Cleaning**:
   - Convert review text to lowercase.
   - Remove punctuation.
3. **Tokenization & Lemmatization** (optional for rule-based).

```python
import pandas as pd
import re

data = pd.read_csv('reviews.csv')
data['Text_clean'] = data['Text'].str.lower().str.replace(r"[^\w\s]", "", regex=True)
```

## Methods

### 1. VADER Sentiment Analysis
- Compute compound sentiment scores for each review:
  ```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()
data['vader_score'] = data['Text_clean'].apply(lambda x: vader.polarity_scores(x)['compound'])
```
- Label reviews as **negative**, **neutral**, or **positive** based on score thresholds.
- Visualize distribution with a bar chart.

### 2. Transformer-based Pipeline
- Use Hugging Face pipeline with default DistilBERT model:
  ```python
from transformers import pipeline
clf = pipeline('sentiment-analysis', truncation=True)
labels = clf(data['Text_clean'].tolist()[:100])
```
- Compare labels against Vader scores for a subset of reviews.

## Usage
1. Place `reviews.csv` in the project root.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```
3. Execute cells in order to reproduce results and plots.

## Results & Discussion
- VADER quickly processes the entire dataset and offers interpretable compound scores.
- Transformer-based classification provides fine-grained labels but is slower on large batches.
- For production scenarios on massive datasets, a hybrid or optimized batching strategy is recommended.

## Future Work
- **Model Fine-Tuning**: Fine-tune a transformer on the Amazon reviews for domain-specific performance.
- **Feature Engineering**: Explore n-grams, TF-IDF, or word embeddings.
- **Evaluation**: Compare against ground-truth star ratings for quantitative metrics (accuracy, F1).
- **Deployment**: Wrap the pipeline in a FastAPI service for real-time sentiment predictions.

## License
This project is released under the MIT License. Feel free to use and modify it for academic or personal purposes.


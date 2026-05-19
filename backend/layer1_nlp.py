import re
import os
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download required NLTK data silently
for resource, path in [('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora/stopwords')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


class SentimentClassifier:
    """
    Camada I: NLP com Naive Bayes para classificação de sentimentos.
    Treina sobre o dataset do Kaggle (Coursera reviews) e classifica
    textos como positivo, negativo ou neutro.
    """

    LABEL_MAP_NUMERIC = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = MultinomialNB(alpha=1.0)
        self.accuracy = 0.0
        self._train()

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [
            self.stemmer.stem(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return ' '.join(tokens)

    # ------------------------------------------------------------------
    # Dataset loading — handles the Kaggle Coursera reviews CSV
    # ------------------------------------------------------------------

    def _load_dataset(self) -> pd.DataFrame:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        # Try common filenames for the Kaggle download
        candidates = ['reviews.csv', 'coursera_reviews.csv', 'dataset.csv']
        path = None
        for name in candidates:
            full = os.path.join(data_dir, name)
            if os.path.exists(full):
                path = full
                break

        if path is None:
            print("[NLP] Kaggle CSV not found. Using synthetic fallback dataset.")
            return self._synthetic_dataset()

        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        # --- detect text column ---
        text_col = next((c for c in df.columns if c.lower() in ('reviews', 'reviewtext', 'review_text', 'text', 'review')), None)
        if text_col is None:
            print("[NLP] Could not detect text column. Using synthetic fallback.")
            return self._synthetic_dataset()

        # --- detect label ---
        label_col = next((c for c in df.columns if c.lower() == 'label'), None)
        rating_col = next((c for c in df.columns if c.lower() in ('rating', 'ratings', 'stars')), None)

        df = df.dropna(subset=[text_col])

        if label_col and df[label_col].nunique() <= 5:
            numeric = pd.to_numeric(df[label_col], errors='coerce')
            if numeric.notna().any() and set(numeric.dropna().astype(int).unique()).issubset(self.LABEL_MAP_NUMERIC.keys()):
                # Numeric 1-5 star labels (Kaggle Coursera dataset)
                df = df[numeric.notna()].copy()
                df['sentiment'] = numeric[df.index].astype(int).map(self.LABEL_MAP_NUMERIC)
            else:
                # String labels ('positive'/'negative'/'neutral' or '-1'/'0'/'1'/'2')
                labels = df[label_col].astype(str).str.lower().str.strip()
                valid_labels = {'positive', 'negative', 'neutral', '1', '0', '-1', '2'}
                df = df[labels.isin(valid_labels)].copy()
                mapping = {'1': 'positive', '0': 'negative', '-1': 'negative', '2': 'neutral'}
                df['sentiment'] = labels[df.index].replace(mapping)
        elif rating_col:
            # Derive sentiment from star rating
            df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
            df = df.dropna(subset=[rating_col])
            df['sentiment'] = df[rating_col].round().astype(int).map(self.LABEL_MAP_NUMERIC)
        else:
            print("[NLP] No label or rating column found. Using synthetic fallback.")
            return self._synthetic_dataset()

        df = df.dropna(subset=['sentiment'])
        df = df.rename(columns={text_col: 'text'})

        # Balance classes and cap at 20k total for fast training
        per_class = min(6000, df.groupby('sentiment').size().min())
        df = df.groupby('sentiment').sample(n=per_class, random_state=42).reset_index(drop=True)
        print(f"[NLP] Loaded dataset: {len(df)} samples, distribution:\n{df['sentiment'].value_counts().to_dict()}")
        return df[['text', 'sentiment']]

    def _synthetic_dataset(self) -> pd.DataFrame:
        """Minimal fallback so the system still runs without the Kaggle file."""
        rows = [
            ("This lecture was incredibly clear and the examples were excellent.", "positive"),
            ("Really enjoyed this module. Well structured and engaging content.", "positive"),
            ("Amazing explanations! I finally understand this topic.", "positive"),
            ("The instructor did a great job making complex ideas accessible.", "positive"),
            ("Fantastic material. I learned a lot and feel very confident now.", "positive"),
            ("Best course I have taken. Very informative and well organized.", "positive"),
            ("Loved the practical examples. Everything made sense after this.", "positive"),
            ("Outstanding content that really helped me grasp the concept.", "positive"),
            ("Very helpful module. The step-by-step approach was perfect.", "positive"),
            ("Excellent course! Clear objectives and great delivery throughout.", "positive"),
            ("This was very confusing. I could not follow the explanations.", "negative"),
            ("Terrible module. The material was disorganized and unclear.", "negative"),
            ("I struggled through this entire content. Very poorly explained.", "negative"),
            ("The explanations were vague and the examples were irrelevant.", "negative"),
            ("Disappointing content. I learned nothing from this module.", "negative"),
            ("Too complex with no proper introduction to foundational concepts.", "negative"),
            ("Very frustrating. The instructor skipped important steps.", "negative"),
            ("Worst learning material I have seen. Confusing and poorly made.", "negative"),
            ("The content was hard to follow and left me more confused.", "negative"),
            ("Poor quality. The module needs major improvement.", "negative"),
            ("The content was okay. Not amazing but covered the basics.", "neutral"),
            ("Average module. Some parts were clear, others less so.", "neutral"),
            ("Decent material overall. Nothing exceptional but functional.", "neutral"),
            ("The module was satisfactory. Met basic expectations.", "neutral"),
            ("Standard content. It covered what was needed without depth.", "neutral"),
            ("This module was acceptable. Not great, not bad.", "neutral"),
            ("Fair content. Some useful information but could be better.", "neutral"),
            ("The material was passable. Some good parts and some confusing.", "neutral"),
            ("Adequate explanations. Sufficient for a basic understanding.", "neutral"),
            ("Reasonable module. It served its purpose without excelling.", "neutral"),
        ]
        return pd.DataFrame(rows, columns=['text', 'sentiment'])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self):
        df = self._load_dataset()
        processed = [self.preprocess(t) for t in df['text']]

        X_train, X_test, y_train, y_test = train_test_split(
            processed, df['sentiment'].tolist(), test_size=0.2, random_state=42, stratify=df['sentiment']
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        self.accuracy = self.model.score(X_test_vec, y_test)
        print(f"[NLP] Naive Bayes accuracy: {self.accuracy:.2%}")
        print(classification_report(y_test, self.model.predict(X_test_vec), zero_division=0))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        processed = self.preprocess(text)
        X = self.vectorizer.transform([processed])
        sentiment = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        proba_dict = {c: round(float(p), 4) for c, p in zip(self.model.classes_, proba)}
        return {
            'sentiment': sentiment,
            'probabilities': proba_dict,
            'positive_prob': proba_dict.get('positive', 0.0),
            'preprocessed': processed,
        }

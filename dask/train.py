import os
import re
import joblib

import dask.dataframe as dd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('stopwords', quiet=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_sentimientos_500.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')

STOPWORDS_ES = set(stopwords.words('spanish'))
LABEL_MAP = {'positivo': 0, 'negativo': 1, 'neutral': 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    tokens = [t for t in text.split() if t not in STOPWORDS_ES and len(t) > 2]
    return ' '.join(tokens)


def train():
    print("[Train] Loading dataset with Dask...")
    ddf = dd.read_csv(DATA_PATH)

    print("[Train] Computing and cleaning text...")
    df = ddf[['texto', 'sentimiento']].compute()
    df['texto_limpio'] = df['texto'].apply(clean_text)
    df['label'] = df['sentimiento'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['texto_limpio'], df['label'].astype(int),
        test_size=0.2, random_state=42, stratify=df['label']
    )

    print("[Train] Fitting: HashingTF → IDF → Naive Bayes...")
    pipeline = Pipeline([
        ('hasher', HashingVectorizer(n_features=2 ** 14, alternate_sign=False, norm=None)),
        ('tfidf', TfidfTransformer()),
        ('nb', MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['positivo', 'negativo', 'neutral']))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'label_map': LABEL_MAP, 'label_names': LABEL_NAMES}, MODEL_PATH)
    print(f"[Train] Model saved → {MODEL_PATH}")


if __name__ == '__main__':
    train()

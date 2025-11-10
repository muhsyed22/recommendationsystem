# model.py – streaming + 100 k sample + tqdm progress
import os
import json
import gzip
import random
import numpy as np
import nltk
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Downloading NLTK data...")
nltk.download('punkt',    quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',  quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

DATA_PATH = "data/clothing.json.gz"
MODEL_PATH = "word2vec.model"
VECS_PATH  = "vectors.npy"
PKL_PATH   = "sampled_products.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Put clothing.json.gz in: {os.path.dirname(DATA_PATH)}")

TARGET_SAMPLE = 100_000
products = []

print(f"Streaming {DATA_PATH} → sampling {TARGET_SAMPLE:,} items …")
with gzip.open(DATA_PATH, 'rt', encoding='utf-8') as f:
    # tqdm on the file object (shows line count)
    for line in tqdm(f, desc="Reading lines", unit="line"):
        if random.random() < (TARGET_SAMPLE / (len(products) + 1)):
            try:
                data = json.loads(line)
                products.append(data)
                if len(products) >= TARGET_SAMPLE:
                    break
            except Exception:
                continue

print(f"Sampled {len(products):,} products")

# ---------- 2. Build corpus ----------
def preprocess(text: str):
    if not text or not str(text).strip():
        return []
    tokens = nltk.word_tokenize(str(text).lower())
    return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

def full_text(row):
    parts = []
    if row.get("title"): parts.append(str(row["title"]))
    desc = row.get("description", "")
    if isinstance(desc, list): desc = " ".join(desc)
    if desc: parts.append(str(desc))
    if "feature" in row and isinstance(row["feature"], list):
        parts.append(" ".join(row["feature"]))
    return " ".join(parts)

print("Building corpus …")
corpus = [preprocess(full_text(p)) for p in tqdm(products, desc="Tokenizing", unit="product")]

# ---------- 3. Train Word2Vec ----------
print("Training Word2Vec …")
model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)
model.save(MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

def get_vector(text):
    toks = preprocess(text)
    if not toks: return np.zeros(100)
    vecs = [model.wv[t] for t in toks if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

print("Generating product vectors …")
product_vectors = np.array([
    get_vector(full_text(p)) for p in tqdm(products, desc="Vectorising", unit="product")
])
np.save(VECS_PATH, product_vectors)
print(f"Vectors saved → {VECS_PATH}")
import pickle
with open(PKL_PATH, "wb") as f:
    pickle.dump(products, f)
print(f"Sampled products saved → {PKL_PATH}")
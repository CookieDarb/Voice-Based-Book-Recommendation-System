import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import pickle
import os

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TFIDF_PATH = os.path.join(CACHE_DIR, "tfidf.pkl")
DF_PATH = os.path.join(CACHE_DIR, "df.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

def load_or_process():
    if os.path.exists(DF_PATH) and os.path.exists(TFIDF_PATH):
        print("Loading cached models...")
        df = pd.read_pickle(DF_PATH)
        tfidf = pickle.load(open(TFIDF_PATH, 'rb'))
    else:
        print("First-time processing...")
        df = pd.read_csv('data/books.csv', on_bad_lines='skip')
        
        df['description'] = df['description'].fillna("")
        df['categories'] = df['categories'].fillna("")
        df['authors'] = df['authors'].fillna("Unknown Author")
        
        df['text'] = (df['description'] + " " + df['categories']).apply(clean_text)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit_transform(df['text'])
        
        df.to_pickle(DF_PATH)
        pickle.dump(tfidf, open(TFIDF_PATH, 'wb'))
    
    return df, tfidf

df, tfidf = load_or_process()

def recommend(query):
    clean_query = clean_text(query)
    query_vec = tfidf.transform([clean_query])
    tfidf_matrix = tfidf.transform(df['text'])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
    sim_scores = sorted(enumerate(cosine_sim[0]), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for idx, score in sim_scores[:5]:
        if score > 0:
            recommendations.append({
                'title': str(df['title'].iloc[idx]),
                'authors': str(df['authors'].iloc[idx]),
                'categories': str(df['categories'].iloc[idx]),
                'score': float(round(score, 3)),
                'description': str(df['description'].iloc[idx])
            })
    return recommendations
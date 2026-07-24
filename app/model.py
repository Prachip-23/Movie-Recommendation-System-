import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== HELPERS ==================

def convert(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except:
        return []
    return L

def create_soup(x):
    return " ".join(x['genres']) + " " + \
           " ".join(x['keywords']) + " " + \
           " ".join(x['cast']) + " " + \
           x['overview']

# ================== LOAD DATA ==================

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

# ================== CLEAN ==================

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)

movies['cast'] = movies['cast'].apply(lambda x: x[:3])

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['soup'] = movies.apply(create_soup, axis=1)

# ================== VECTORIZE ==================

cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(movies['soup'])

cosine_sim = cosine_similarity(vectors)

# ================== INDEX ==================

movies['title_lower'] = movies['title'].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# ================== SAVE ==================

pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(cosine_sim, open('similarity.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))

print("✅ Model trained & saved!")
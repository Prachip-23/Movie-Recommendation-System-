import pandas as pd
import ast
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== API KEY ==================
OMDB_API_KEY = "c15bcb63"

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

# ================== CLEAN DATA ==================

movies['overview'] = movies['overview'].fillna('')

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)

movies['cast'] = movies['cast'].apply(lambda x: x[:3])

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

# fix release_date issue
movies['release_date'] = movies['release_date'].fillna('')
movies['release_date'] = movies['release_date'].astype(str)

movies['soup'] = movies.apply(create_soup, axis=1)

# ================== MODEL ==================

cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(movies['soup'])

cosine_sim = cosine_similarity(vectors)

movies['title_lower'] = movies['title'].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# ================== POSTER ==================

poster_cache = {}

def fetch_poster(movie_name):
    if movie_name in poster_cache:
        return poster_cache[movie_name]

    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={movie_name}"

    try:
        data = requests.get(url, timeout=3).json()
        if data.get('Response') == 'True':
            poster = data.get('Poster')
            if poster and poster != "N/A":
                poster_cache[movie_name] = poster
                return poster
    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"

# ================== RECOMMENDATION ==================

def get_recommendations(title):
    title = title.lower().strip()

    matches = [t for t in indices.index if title in t]

    if not matches:
        return pd.DataFrame()

    idx = indices[matches[0]]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:15]

    movie_indices = [i[0] for i in sim_scores]

    df = movies.iloc[movie_indices]

    df = df.sort_values(by='popularity', ascending=False)

    return df

# ================== FLASK ==================

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('title')

    df = get_recommendations(movie_name)

    if df.empty:
        return render_template('result.html', titles=[], images=[])

    titles = df['title'].tolist()
    posters = [fetch_poster(t) for t in titles]

    return render_template('result.html', titles=titles, images=posters)

@app.route('/movie/<name>')
def movie_page(name):
    movie = movies[movies['title'] == name]

    if movie.empty:
        return "Movie not found"

    movie = movie.iloc[0]

    poster = fetch_poster(movie['title'])

    return render_template(
        'moviepage.html',
        movie=movie,
        poster=poster
    )

# ================== 🔥 AUTOCOMPLETE FEATURE ==================

@app.route('/search_suggestions')
def search_suggestions():
    q = request.args.get('q', '').lower().strip()

    if not q:
        return jsonify({"results": []})

    matches = [t for t in indices.index if t.startswith(q)]

    return jsonify({"results": matches[:8]})

# ================== RUN ==================

if __name__ == "__main__":
    app.run(debug=True)
import sqlite3
from flask import Flask, g, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Specify the path of the database to be used
DATABASE = 'rss.db'

# Create the application instance
app = Flask(__name__) 

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def prepare_dataset(c, split):
    if split == "train":
        c.execute(
            "SELECT rowid, title, link, description, published, score FROM rss WHERE seen = 1"
        )
    else:
        c.execute(
            "SELECT rowid, title, link, description, published, score FROM rss WHERE seen = 0"
        )

    query = c.fetchall()

    # Extract the values of 'title', 'link', 'description', 'published' and 'score' from the fetched data
    rowids = [q[0] for q in query]
    titles = [q[1] for q in query]
    links = [q[2] for q in query]
    descriptions = [q[3] for q in query]
    published = ["" if q[4] is None else q[4] for q in query]
    scores = [q[5] for q in query]

    # Encode the values of 'title', 'link', 'description' and 'published' using the pre-trained Sentence Transformer model
    titles_embedding = model.encode(titles)
    links_embedding = model.encode(links)
    descriptions_embedding = model.encode(descriptions)
    published_embedding = model.encode(published)

    # Calculate the average of the embeddings
    avg_embeddings = np.mean(
        [
            titles_embedding,
            links_embedding,
            descriptions_embedding,
            published_embedding,
        ],
        axis=0,
    )

    # Normalize the average embeddings
    avg_embeddings = avg_embeddings / np.linalg.norm(avg_embeddings)

    # Convert the score values to a numpy array
    labels = np.array([label for label in scores])

    return avg_embeddings, labels, rowids, titles, links

def train_model(c):
    # Prepare the training dataset
    X_train, y_train, _, _, _ = prepare_dataset(c, "train")

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model on the average embeddings and scores
    clf.fit(X_train, y_train)

    # Prepare the test dataset
    X_test, _, rowids, titles, links = prepare_dataset(c, "test")

    # Calculate the decision function values for the average embeddings
    preds = clf.predict(X_test)

    # Calculate the probability estimates for the average embeddings
    probas = clf.predict_proba(X_test)

    sorted_idx = np.argsort(probas[:, 0])

    return rowids, titles, links, preds, probas, sorted_idx

def sample_model(c):
    top_n = 100
    top_k = 25
    temperature = 0.5
    rowids, titles, links, preds, probas, sorted_idx = train_model(c)
    better_idx = sorted_idx[:top_n]
    p = better_idx ** temperature / np.sum(better_idx ** temperature)
    better_idx = np.random.choice(better_idx, size=top_k, replace=False, p=p)
    return [(rowids[i], titles[i], links[i], probas[i]) for i in better_idx]

samples = None

@app.route('/seen', methods=['POST'])
def seen():
    cur = get_db().cursor()
    rowid = request.form['rowid']
    cur.execute("UPDATE rss SET seen = 1 WHERE rowid = ?", (rowid,))
    get_db().commit()
    return redirect(url_for('index'))

@app.route('/like', methods=['POST'])
def like():
    cur = get_db().cursor()
    rowid = request.form['rowid']
    cur.execute("UPDATE rss SET score = 1 WHERE rowid = ?", (rowid,))
    get_db().commit()
    return redirect(url_for('index'))

@app.route('/dislike', methods=['POST'])
def dislike():
    cur = get_db().cursor()
    rowid = request.form['rowid']
    cur.execute("UPDATE rss SET score = -1 WHERE rowid = ?", (rowid,))
    get_db().commit()
    return redirect(url_for('index'))

@app.route('/refresh', methods=['POST'])
def refresh():
    global samples
    samples = None
    return redirect(url_for('index'))

@app.route('/')
def index():
    global samples
    if samples is None:
        cur = get_db().cursor()
        samples = sample_model(cur)
    return render_template('index.html', samples=samples)

if __name__ == '__main__': 
    app.run(debug=True)

# Importing necessary libraries
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import sys

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Specify the path of the database to be used
DB = "rss.db"

# Connect to the SQLite database
conn = sqlite3.connect(DB)
c = conn.cursor()

# Select data from the 'rss' table and fetch all the rows
# c.execute("SELECT title, link, description, published, score FROM rss WHERE seen = 1")

def prepare_dataset(split):
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


# Prepare the training dataset
X_train, y_train, _, _, _ = prepare_dataset("train")

# Create a Linear Support Vector Classification model with specified parameters
clf = RandomForestClassifier(n_estimators=500)
#clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)

# Train the model on the average embeddings and scores
clf.fit(X_train, y_train)

# Prepare the test dataset
X_test, _, rowids, titles, links = prepare_dataset("test")

# Calculate the decision function values for the average embeddings
preds = clf.predict(X_test)

# Calculate the probability estimates for the average embeddings
probas = clf.predict_proba(X_test)
#probas = clf.decision_function(X_test)

# Sort the decision function values in descending order and get the index positions
sorted_idx = np.argsort(probas[:, 0])

if __name__ == "__main__":
    args = sys.argv[1:]

    top_n = 100
    top_k = 10
    temperature = 0.5

    while args:
        if args[0] == "--temp":
            temperature = float(args[1])
            args = args[2:]
        elif args[0] == "--top-n":
            top_n = int(args[1])
            args = args[2:]
        elif args[0] == "--top-k":
            top_k = int(args[1])
            args = args[2:]
        else:
            raise ValueError(f"Unknown argument {args[0]}")

    better_idx = sorted_idx[:top_n]
    p = better_idx ** temperature / np.sum(better_idx ** temperature)
    better_idx = np.random.choice(better_idx, size=top_k, replace=False, p=p)    

    for k in better_idx:
        print(f"{rowids[k]} {preds[k]} {probas[k]} {titles[k]}, {links[k]}")

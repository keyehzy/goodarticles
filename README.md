# Personalized RSS Feed

RSS is a good form of obtaining the latest news, but as the number of news
sources increase, you just have too much data to read through.

This project organize you RSS as sqlite database and have a simple
recommendation system based on the articles you've seen and like or disliked.

# Usage

Create a virtual environment, activate it and install the requirements.

``` bash
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

To add new RSS feeds, edit the `feeds.txt` file and add the RSS feed URL, which
is read by default, or pass the file as an argument to the script.

``` bash
python3 rss.py --file feeds.txt
```

The first time you run the script, it will create a database file called
`rss.db` and populate it with the articles from the RSS feeds. You can also add
individual RSS feeds by passing them as arguments.

``` bash
python3 rss.py --list https://www.reddit.com/r/programming/.rss
```

These are the actions that you can perform on the articles:

* '--list' - List the articles in the database
* '--list-read' - List the articles that you've read
* '--list-unread' - List the articles that you haven't read
* '--list-like' - List the articles that you've liked
* '--list-unliked' - List the articles that you've disliked
* '--open' - Open the article in the browser
* '--mark-read' - Mark the article as read
* '--mark-unread' - Mark the article as unread
* '--like' - Like the article
* '--dislike' - Dislike the article
* '--reset-score' - Reset the score of the article
* '--search' - Search for articles in the database
* '--fetch' - Fetch the articles from the RSS feeds
' '--file' - The file containing the RSS feeds
* '--random' - Open a random article

The `scores.py` script will process the articles and calculate the score of each
article based on the articles that you've liked or disliked. Then it will sample
10 out of the highest scored articles and print them.

``` bash
python3 scores.py
```

Alternatively, an Flask web application is provided to view the articles and
perform the actions. To run the web application, run the `main.py` script.

``` bash
python3 main.py
```

The web application will be available at `http://localhost:5000/` and it will
show the articles that you haven't read yet. You can click on the article to
open it in the browser, or click on the like or dislike button to like or
dislike the article. You can also click on the 'Refresh' button to refresh the
articles.

# Requirements

Python 3.6 or higher is required. See [requirements.txt](requirements.txt) for
the list of requirements.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file

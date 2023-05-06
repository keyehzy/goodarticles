import feedparser
import sqlite3
import webbrowser
import sys

class SqliteCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self) -> sqlite3.Cursor:
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, type, value, traceback):
        if type is None:
            self.connection.commit()


DB = "rss.db"
DEFAULT_FEEDS = "feeds.txt"


def add_rss(conn, feed):
    """Add RSS feed to database"""

    with SqliteCursor(conn) as c:
        # Create table if it doesn't exist
        c.execute(
            "CREATE TABLE IF NOT EXISTS rss (title text UNIQUE, link text UNIQUE, description text UNIQUE, published text UNIQUE, seen integer default 0, score float default 0 )"
        )

        # Parse RSS feed
        d = feedparser.parse(feed)

        # Insert data into table
        for post in d.entries:
            title = getattr(post, "title", None)
            link = getattr(post, "link", None)
            description = getattr(post, "description", None)
            published = getattr(post, "published", None)
            seen = 0
            score = 0
            c.execute(
                "INSERT OR IGNORE INTO rss VALUES (?, ?, ?, ?, ?, ?)",
                (title, link, description, published, seen, score),
            )


def list_articles(conn, query=None):
    """List RSS feeds"""
    conn = sqlite3.connect(DB)
    with SqliteCursor(conn) as c:
        if query == "liked":
            c.execute(
                "SELECT rowid, title, link FROM rss WHERE score = 1 ORDER BY date(published) ASC"
            )
        elif query == "read":
            c.execute(
                "SELECT rowid, title, link FROM rss WHERE seen = 1 ORDER BY date(published) ASC"
            )
        elif query == "unread":
            c.execute(
                "SELECT rowid, title, link FROM rss WHERE seen = 0 ORDER BY date(published) ASC"
            )
        else:
            c.execute("SELECT rowid, title, link FROM rss ORDER BY date(published) ASC")
        for row in c.fetchall():
            print(f"{row[0]}: {row[1]}, {row[2]}")


def open_article(conn, idx):
    """Open article in browser"""
    with SqliteCursor(conn) as c:
        c.execute("SELECT title, link FROM rss WHERE rowid = ?", (idx,))
        for row in c.fetchall():
            print(f"{row[0]}: {row[1]}")
            webbrowser.open(row[1])


def article_action(conn, idx, act):
    """Perform action on article"""
    with SqliteCursor(conn) as c:
        if act == "like":
            c.execute("UPDATE rss SET score = 1 WHERE rowid = ?", (idx,))
        elif act == "reset-score":
            c.execute("UPDATE rss SET score = 0 WHERE rowid = ?", (idx,))
        elif act == "dislike":
            c.execute("UPDATE rss SET score = -1 WHERE rowid = ?", (idx,))
        elif act == "read":
            c.execute("UPDATE rss SET seen = 1 WHERE rowid = ?", (idx,))
        elif act == "unread":
            c.execute("UPDATE rss SET seen = 0 WHERE rowid = ?", (idx,))
        else:
            raise ValueError("Invalid action")


def search_article(conn, s):
    """Search for article"""
    with SqliteCursor(conn) as c:
        c.execute("SELECT rowid, title, link FROM rss WHERE title LIKE ? OR description LIKE ? OR link LIKE ?", (f"%{s}%", f"%{s}%", f"%{s}%"))
        for row in c.fetchall():
            print(f"{row[0]}: {row[1]}, {row[2]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    conn = sqlite3.connect(DB)

    while args:
        if args[0] == "--add":
            add_rss(conn, args[1])
            args = args[2:]

        elif args[0] == "--list":
            list_articles(conn)
            args = args[1:]

        elif args[0] == "--open":
            open_article(conn, args[1])
            args = args[2:]

        elif args[0] == "--list-read":
            list_articles("read")
            args = args[1:]

        elif args[0] == "--list-unread":
            list_articles("unread")
            args = args[1:]

        elif args[0] == "--like":
            article_action(conn, int(args[1]), "like")
            args = args[2:]

        elif args[0] == "--reset-score":
            article_action(conn, int(args[1]), "reset-score")
            args = args[2:]

        elif args[0] == "--dislike":
            article_action(conn, int(args[1]), "dislike")
            args = args[2:]

        elif args[0] == "--search":
            search_article(conn, args[1])
            args = args[2:]

        elif args[0] == "--mark-read":
            article_action(conn, int(args[1]), "read")
            args = args[2:]

        elif args[0] == "--mark-unread":
            article_action(conn, int(args[1]), "unread")
            args = args[2:]

        elif args[0] == "--list-liked":
            list_articles("liked")
            args = args[1:]

        elif args[0] == "--list-unliked":
            list_articles("unliked")
            args = args[1:]

        elif args[0] == "--file":
            with open(args[1], "r") as f:
                for line in f:
                    add_rss(conn, line)
            args = args[2:]

        elif args[0] == "--fetch":
            try:
                with open(DEFAULT_FEEDS, "r") as f:
                    for line in f:
                        add_rss(conn, line)
            except FileNotFoundError:
                print("No feeds.txt file found")
            args = args[1:]
        
        elif args[0] == "--random":
            with SqliteCursor(conn) as c:
                c.execute("SELECT rowid, title, link FROM rss WHERE seen = 0 ORDER BY RANDOM() LIMIT 1")
                for row in c.fetchall():
                    print(f"{row[0]}: {row[1]}, {row[2]}")
                    webbrowser.open(row[2])
                    article_action(conn, row[0], "read")
            args = args[1:]
                
        else:
            print(f"Unknown command {args[0]}")
            break

        
    conn.close()

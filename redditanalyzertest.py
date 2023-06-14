'''
Test script for the reddit analysis part of the program using praw to scrape
25 most recent posts on wsb
'''
import praw
from Keys import client_id, client_secret, user_agent,username,password

def main():
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent,
                         username=username, password=password)
    subreddit = reddit.subreddit('wallstreetbets')
    top_subreddit = subreddit.new(limit=25)
    words_collection = []

    for submission in top_subreddit:
        title = submission.title
        title_words = title.split()
        words_collection.append(title_words)

    print(words_collection)
main()
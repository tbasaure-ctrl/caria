
import os
import sys

# Set env vars provided by user
os.environ["REDDIT_CLIENT_ID"] = "your-reddit-client-id"
os.environ["REDDIT_CLIENT_SECRET"] = "your-reddit-client-secret"
os.environ["REDDIT_USER_AGENT"] = "Caria/1.0"

try:
    import praw
    print("PRAW is installed.")
except ImportError:
    print("PRAW is NOT installed.")
    sys.exit(1)

def verify_reddit():
    print("Connecting to Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
            check_for_async=False
        )
        
        print("Fetching top post from r/stocks...")
        subreddit = reddit.subreddit("stocks")
        count = 0
        for submission in subreddit.hot(limit=3):
            print(f"- {submission.title} (Score: {submission.score})")
            count += 1
            
        if count > 0:
            print("\nSUCCESS: Retrieved data from Reddit API.")
        else:
            print("\nWARNING: No posts found (or API issue).")

    except Exception as e:
        print(f"\nERROR: Reddit API failed: {e}")

if __name__ == "__main__":
    verify_reddit()

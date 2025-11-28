import os
import pandas as pd
import argparse
from twarc import Twarc2, expansions
import json

def hydrate_tweets(input_file, output_file, api_key=None, api_secret=None, bearer_token=None):
    """
    Hydrates tweets using Twarc2 (Twitter API v2).
    """
    
    if not bearer_token and not (api_key and api_secret):
        print("Error: Twitter API credentials required.")
        print("Please set TWITTER_BEARER_TOKEN or (TWITTER_API_KEY and TWITTER_API_SECRET) environment variables.")
        return

    print(f"Initializing Twarc client...")
    client = Twarc2(consumer_key=api_key, consumer_secret=api_secret, bearer_token=bearer_token)
    
    print(f"Reading input file: {input_file}")
    # Detect separator
    try:
        df = pd.read_csv(input_file, sep=None, engine='python')
    except:
        df = pd.read_csv(input_file)
        
    # Ensure TweetID is string to avoid scientific notation issues
    if 'TweetID' in df.columns:
        tweet_ids = df['TweetID'].astype(str).tolist()
    elif df.columns[0].lower() == 'tweetid': # Handle case where header might be different
         tweet_ids = df.iloc[:, 0].astype(str).tolist()
    else:
        # Fallback: assume first column is ID
        tweet_ids = df.iloc[:, 0].astype(str).tolist()
        
    print(f"Found {len(tweet_ids)} tweet IDs to hydrate.")
    
    # Chunking is handled by Twarc, but we pass a generator
    lookup = client.tweet_lookup(tweet_ids=tweet_ids)
    
    hydrated_data = []
    
    print("Starting hydration... (this may take a while)")
    count = 0
    for page in lookup:
        if 'data' in page:
            for tweet in page['data']:
                hydrated_data.append({
                    'TweetID': tweet['id'],
                    'text': tweet['text'],
                    'created_at': tweet['created_at'],
                    'author_id': tweet['author_id']
                })
                count += 1
                if count % 100 == 0:
                    print(f"Hydrated {count} tweets...")
    
    print(f"Finished. Successfully hydrated {len(hydrated_data)} tweets.")
    
    if len(hydrated_data) == 0:
        print("Warning: No tweets were hydrated. They might be deleted or the API keys are invalid.")
        return

    hydrated_df = pd.DataFrame(hydrated_data)
    
    # Merge with original labels if possible
    # Ensure ID columns match type
    df['TweetID'] = df.iloc[:, 0].astype(str)
    
    merged_df = pd.merge(df, hydrated_df, on='TweetID', how='inner')
    
    print(f"Saving {len(merged_df)} merged records to {output_file}")
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hydrate tweets from CSV.')
    parser.add_argument('--input', type=str, default='../data/NAACL_SRW_2016.csv', help='Path to input CSV with Tweet IDs')
    parser.add_argument('--output', type=str, default='../data/hydrated_tweets.csv', help='Path to output CSV')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if they are relative
    if not os.path.isabs(args.input):
        args.input = os.path.join(os.path.dirname(__file__), args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(os.path.dirname(__file__), args.output)

    # Get credentials from env
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    
    hydrate_tweets(args.input, args.output, api_key, api_secret, bearer_token)

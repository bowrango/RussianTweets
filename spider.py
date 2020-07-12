from twitterscraper import query_tweets_from_user as get_tweets
from twitterscraper import query_user_info
import pandas as pd
from multiprocessing import Pool


def get_user_info(twitter_user):
    """
    :param twitter_user: the twitter handle to capture data from
    :return: twitter_user_data: returns a dictionary of twitter user data
    """
    user_info = query_user_info(user=twitter_user)

    # This is all the data we want collected from each user
    twitter_user_data = {"user": user_info.user,
                         "fullname": user_info.full_name,
                         "location": user_info.location,
                         "following": user_info.following,
                         "followers": user_info.followers,
                         }

    return twitter_user_data

def process_tweets(user_tweet_list):
    """
    :param user_tweet_list: the list of tweets each scraped as a Twitter object
    :return: list containing the formatted text for use by Word2Vec
    """

    tweets = []

    # this basically removes clutter from the text
    for Tobj in user_tweet_list:
        if 'https:' in Tobj.text:
            continue
        else:
            tweets.append(Tobj.text.lower())

    training_data = [s.split() for s in tweets]

    return training_data


# number of tweets scraped from each user.
sample = 50

# list of all the twitter handles to be scraped and processed
users = ['realDonaldTrump'
         ]

# list of dictionaries containing user-data relations
twitter_user_info = []

def main():

    # queuing up users
    pool = Pool(8)
    for user_dict in pool.map(get_user_info, users):
        twitter_user_info.append(user_dict)

    # additional user data to be collected
    collections = ['user',
                   'fullname',
                   'location',
                   'following',
                   'followers',
                   ]

    # I tag on the tweets column separately because a Twitter object already has a tweet attribute
    # representing the number of tweets. This was an easy work around if I'm going to use dictionaries.

    df = pd.DataFrame(twitter_user_info, columns=collections)
    df['tweet data'] = ''

    i = 0
    for user in df['user'].tolist():
        user_tweets = get_tweets(user, limit=sample)

        training_data = process_tweets(user_tweets)
        df.iat[i, 5] = training_data

        i += 1


if __name__ == '__main__':
    main()



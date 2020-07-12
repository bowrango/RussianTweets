import pandas as pd
import re
import pickle as p
import time
import multiprocessing as mp
from nltk.tokenize import word_tokenize

rawfiles = [
        'IRAhandle_tweets_1.csv',
        'IRAhandle_tweets_2.csv',
        # 'IRAhandle_tweets_3.csv',
        # 'IRAhandle_tweets_4.csv',
        # 'IRAhandle_tweets_5.csv',
        # 'IRAhandle_tweets_7.csv',
        # 'IRAhandle_tweets_8.csv',
        # 'IRAhandle_tweets_9.csv',
        # 'IRAhandle_tweets_10.csv',
        # 'IRAhandle_tweets_11.csv',
        # 'IRAhandle_tweets_12.csv',
        # 'IRAhandle_tweets_13.csv',
        ]

def process_csv(filename):
    """
    :param filename: the CSV file that contains the raw tweet data
    :return: a pandas list containing all the tweets within the file
    """
    return pd.read_csv(filename).content


def process_tweets(tweet_set):
    """
    :param tweet_set: the list of tweets from the CSV file content
    :return a list of formatted tweets for Word2Vec
    """
    start = time.time()
    tweet_list = []

    for tweet in tweet_set:

        # removes unwanted punctuation
        tweet = re.sub('[~!$%^&*()_+|}{:"?><`=;/.,]', "", tweet).lower()

        # removes links
        tweet = re.sub(r"http\S+", "", tweet)

        # removes @users
        tweet = re.sub(r"@\S+", "", tweet)

        # removes hashtags
        tweet = re.sub(r"#\S+", "", tweet)

        # splitting tweet into tokenized words
        s = word_tokenize(tweet)

        tweet_list.append(s)

    print('Clean Time: ' + str(time.time() - start)[:5] + ' s')

    return tweet_list


def main():

    import_start = time.time()

    data = []

    pool = mp.Pool(8)
    dataset = pool.map(process_csv, rawfiles)

    import_time = time.time() - import_start
    print('Import Time: ' + str(import_time)[:5] + ' s')

    for tweet_column in dataset:
        d = process_tweets(tweet_column)
        data.append(d)

    training_data = [item for sublist in data for item in sublist]

    file = open('training_data', 'wb')
    p.dump(training_data, file)


if __name__ == '__main__':
    main()
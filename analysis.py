import pickle
import time
import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch"""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Total Loss {} {}'.format(self.epoch, loss))
        self.epoch += 1


WINDOW_SIZE = 4
MIN_COUNT = 10
WORKERS = 8
NEGATIVE = 15
MAX_VOCAB_SIZE = 15000
EPOCH = 4

start = time.time()

file = open('training_data', 'rb')
training_data = pickle.load(file)
file.close()

import_time = time.time() - start
print('Pickle Open: ' + str(import_time)[:5] + ' s')

epochs = []

model_start = time.time()

model = gensim.models.Word2Vec(sentences=training_data,
                               iter=EPOCH,
                               window=WINDOW_SIZE,
                               workers=WORKERS,
                               negative=NEGATIVE,
                               min_count=MIN_COUNT,
                               compute_loss=True,
                               max_vocab_size=MAX_VOCAB_SIZE,
                               callbacks=[callback()]
                               )

model_time = time.time() - model_start
print('Model Generation: ' + str(model_time)[:5] + ' s')

L = len(model.wv.vocab)
print('Vocab Size {}'.format(L))

keys = ['trump', 'bernie', 'republican', 'democrat']

embedding_clusters = []
word_clusters = []

for word in keys:

    embeddings = []
    words = []

    for similar_word, _ in model.wv.most_similar(word, topn=30):

        words.append(similar_word)
        embeddings.append(model.wv[similar_word])

    embedding_clusters.append(embeddings)
    word_clusters.append(words)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_in_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_in_2d = np.array(tsne_model_in_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):

    plt.figure(figsize=(16, 9))

    for label, embeddings, words in zip(labels, embedding_clusters, word_clusters):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, alpha=a, label=label)

        for i, word in enumerate(words):

            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)

    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Test', keys, embeddings_in_2d, word_clusters, 0.7)
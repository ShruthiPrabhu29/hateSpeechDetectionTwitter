from data_handler import get_data
import argparse
import sys
import numpy as np
import pdb
import pickle
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
from my_tokenizer import glove_tokenize
from nltk.tokenize import TweetTokenizer


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


# logistic, gradient_boosting, random_forest, svm_linear, svm_rbf
GLOVE_MODEL_FILE = None
EMBEDDING_DIM = None
MODEL_TYPE = None
CLASS_WEIGHT = None
N_ESTIMATORS = None
LOSS_FUN = None
KERNEL = None
TOKENIZER = None

SEED=42
MAX_NB_WORDS = None
NO_OF_FOLDS=10
N_JOBS=1


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}

word2vec_model = None


def select_tweets_whose_embedding_exists():
    # selects the tweets as in mean_glove_embedding method
    # In this function, we are only checking whether an embedding exists
    # for at least one word within the tweet. If it does, we "accept" the tweet
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = TOKENIZER(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    return tweet_return


def gen_data():
    # In this function, for all accepted tweets, we turn them into an
    # embedding of EMBEDDING_DIM. We then sum the embeddings of all
    # words within the tweet that have an embedding and divide
    # by the number of words. Hence, the final embedding of the tweet
    # will be the average of the embeddings of its words.

    X_file = "BoWV_X.pickle"
    y_file = "BoWV_y.pickle"

    # Load if pickled files are available
    try:
        X = pickle.load(open(X_file, "rb"))
        y = pickle.load(open(y_file, "rb"))
        print("Features and labels loaded from pickled files.")

    # Create and save otherwise
    except (OSError, IOError) as e:
        print("Creating features and labels...")

        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2
                }

        X, y = [], []
        for tweet in tweets:
            words = glove_tokenize(tweet['text'].lower())
            emb = np.zeros(EMBEDDING_DIM)
            for word in words:
                try:
                    emb += word2vec_model[word]
                except:
                    pass
            emb /= len(words)
            X.append(emb)
            y.append(y_map[tweet['label']])

        pickle.dump(X, open(X_file, "wb"))
        pickle.dump(y, open(y_file, "wb"))

    return X, y


def get_model(m_type=None):
    if not m_type:
        print("ERROR: Please specify a model type!")
        return None
    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        logreg = GradientBoostingClassifier(loss=LOSS_FUN, n_estimators=N_ESTIMATORS)
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(class_weight=CLASS_WEIGHT, n_estimators=N_ESTIMATORS)
    elif m_type == "svm":
        logreg = SVC(class_weight=CLASS_WEIGHT, kernel=KERNEL, gamma='scale')
    elif m_type == "svm_linear":
        logreg = LinearSVC(loss=LOSS_FUN, class_weight=CLASS_WEIGHT)
    else:
        print("ERROR: Please specify a correct model")
        return None

    return logreg


def classification_model(X, Y, model_type=None):
    X, Y = shuffle(X, Y, random_state=SEED)
    print("Model Type:", model_type)

    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)

    scorers = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring=scorers, verbose=1, n_jobs=N_JOBS)

    scores1 = scores["test_precision_weighted"]
    scores2 = scores["test_recall_weighted"]
    scores3 = scores["test_f1_weighted"]

    print("Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2))
    print("Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2))
    print("F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BagOfWords model for twitter Hate speech detection')
    parser.add_argument('-m', '--model', choices=['logistic', 'gradient_boosting', 'random_forest', 'svm', 'svm_linear'], required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--estimators', default=N_ESTIMATORS)
    parser.add_argument('--loss', default=LOSS_FUN)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--jobs', default=N_JOBS)
    parser.add_argument('--class_weight')


    args = parser.parse_args()
    MODEL_TYPE = args.model
    GLOVE_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    N_ESTIMATORS = int(args.estimators) if args.estimators else None
    LOSS_FUN = args.loss
    KERNEL = args.kernel
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = TweetTokenizer().tokenize

    print('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print('Embedding Dimension: %d' %(EMBEDDING_DIM))

    ### NOTE: If you have downloaded the glove model file from the GloVe website,
    # you need to include the vocab size and embedding dimension at the top of the
    # .txt file.  For example, for the 'glove.twitter.27B.25d' file,
    # you would append '1193514 25' as the first line.
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE, binary=False)
    print("GloVe model loaded successfully.")

    #filter_vocab(20000)

    tweets = select_tweets_whose_embedding_exists()
    X, Y = gen_data()

    classification_model(X, Y, MODEL_TYPE)
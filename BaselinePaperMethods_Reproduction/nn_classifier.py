from data_handler import get_data
import sys
from keras.models import load_model
import numpy as np
import pdb, json
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
import pickle
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
import xgboost as xgb

### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# logistic, gradient_boosting, random_forest, svm, tfidf_svm_linear, tfidf_svm_rbf
word_embed_size = 200
MODEL_TYPE=sys.argv[1]
MODEL_FILE=sys.argv[2]
#MODEL_TYPE="gradient_boosting"
#MODEL_FILE= "lstm_random.h5"

print('Embedding Dimension: %d' %(word_embed_size))

pretrained_model = load_model(MODEL_FILE)
pretrained_embedding = pretrained_model.layers[0].get_weights()[0]
vocab = pickle.load(open("cnn_vocab.pickle", 'rb'))
word2vec_model = {}
for k,v in vocab.items():
    word2vec_model[k] = pretrained_embedding[int(v)]
del pretrained_model

SEED=42
MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def select_tweets_whose_embedding_exists():
    # selects the tweets as in mean_glove_embedding method
    # Processing

    tweet_return_file = "cnn_tweets.pickle"

    # Load if pickled files are available
    try:
        tweet_return = pickle.load(open(tweet_return_file, "rb"))
        print("Tweets loaded from pickled file.")

    # Create and save otherwise
    except (OSError, IOError) as e:

        print ("Loading tweets with embeddings available...")
        tweets = get_data()
        tweet_return = []
        for tweet in tweets:
            _emb = 0
            words = TOKENIZER(tweet['text'].lower())
            for w in words:
                if w in word2vec_model:  # Check if embeeding there in GLove model
                    _emb+=1
            if _emb:   # Not a blank tweet
                tweet_return.append(tweet)

        pickle.dump(tweet_return, open(tweet_return_file, "wb"))
    print('Tweets selected:', len(tweet_return))
    return tweet_return


def gen_data():
    # Generate features and labels
    # Features will be given by the average
    # embedding for all words in the sentence.

    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    for tweet in tweets:
        words = glove_tokenize(tweet['text']) # .lower()
        emb = np.zeros(word_embed_size)
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(words)
        X.append(emb)
        y.append(y_map[tweet['label']])

    X = np.array(X)
    y = np.array(y)

    return X, y


def get_model(m_type=None):
    if not m_type:
        print('ERROR: Please provide a valid method name')
        return None

    if m_type == 'logistic':
        logreg = LogisticRegression(multi_class='multinomial')
    elif m_type == "gradient_boosting":
        #logreg = GradientBoostingClassifier(n_estimators=10)
        logreg = xgb.XGBClassifier(nthread=-1)
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif m_type == "svm_rbf":
        logreg = SVC(class_weight="balanced", kernel='rbf')
    elif m_type == "svm_linear":
        logreg = LinearSVC(class_weight="balanced")
    else:
        print("ERROR: Please specify a correst model")
        return None

    return logreg


def classification_model(X, Y, model_type="logistic"):
    NO_OF_FOLDS=10
    X, Y = shuffle(X, Y, random_state=SEED)
    print("Model Type:", model_type)

    scorers = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring=scorers, verbose=1, n_jobs=-2)

    scores1 = scores["test_precision_weighted"]
    scores2 = scores["test_recall_weighted"]
    scores3 = scores["test_f1_weighted"]

    print("Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2))
    print("Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2))
    print("F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2))

if __name__ == "__main__":

    #filter_vocab(20000)

    tweets = select_tweets_whose_embedding_exists()
    X, Y = gen_data()

    classification_model(X, Y, MODEL_TYPE)
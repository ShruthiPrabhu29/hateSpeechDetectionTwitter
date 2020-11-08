from data_handler import get_data
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers  import Merge
import numpy as np
import pickle
import pdb
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
from string import punctuation
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
import sys

from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize
from keras.optimizers import Adam


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}



EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
NO_OF_CLASSES=3

SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
TOKENIZER = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 128
SCALE_LOSS_FUN = None
FILTER_SZ = 3  #kernel size


word2vec_model = None



def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception as e:
        print('Encoding not found: %s' %(word))
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print("%d embedding missed"%n)
    #pdb.set_trace()
    return embedding


def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing

    tweet_return_file = "cnn_tweets.pickle"   

    # Load if pickled files are available
    try:
        tweet_return = pickle.load(open(tweet_return_file, "rb"))
        print("Tweets loaded from pickled file.")

    # Create and save otherwise
    except (OSError, IOError) as e:

        print("Loading tweets with embeddings available...")
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



def gen_vocab():

    global vocab, reverse_vocab
    vocab_file = "cnn_vocab.pickle"
    reverse_vocab_file = "cnn_reverse_vocab.pickle"
      

    # Load if pickled files are available
    try:
        vocab = pickle.load(open(vocab_file, "rb"))
        reverse_vocab = pickle.load(open(reverse_vocab_file, "rb"))
        print("Vocabs loaded from pickled files.")

    # Create and save otherwise
    except (OSError, IOError) as e:

        print("Generating vocab files.")
        # Processing
        vocab_index = 1
        for tweet in tweets:
            text = TOKENIZER(tweet['text'].lower())
            text = ''.join([c for c in ' '.join(text) if c not in punctuation])
            words = text.split()
            words = [word for word in words if word not in STOPWORDS]

            for word in words:
                if word not in vocab:
                    vocab[word] = vocab_index
                    reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                    vocab_index += 1
                freq[word] += 1
        vocab['UNK'] = len(vocab) + 1
        reverse_vocab[len(vocab)] = 'UNK'

        pickle.dump(vocab, open(vocab_file, "wb"))
        pickle.dump(reverse_vocab, open(reverse_vocab_file, "wb"))


def filter_vocab(k):
    global freq, vocab
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence():

    X_file = "cnn_X.pickle"
    y_file = "cnn_y.pickle"
       

    # Load if pickled files are available
    try:
        X = pickle.load(open(X_file, "rb"))
        y = pickle.load(open(y_file, "rb"))
        print("X and y loaded from pickled files.")

    # Create and save otherwise
    except (OSError, IOError) as e:

        print("Generating X and y files.")
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2
                }

        X, y = [], []
        for tweet in tweets:
            text = TOKENIZER(tweet['text'].lower())
            text = ''.join([c for c in ' '.join(text) if c not in punctuation])
            words = text.split()
            words = [word for word in words if word not in STOPWORDS]
            seq, _emb = [], []
            for word in words:
                seq.append(vocab.get(word, vocab['UNK']))
            X.append(seq)
            y.append(y_map[tweet['label']])

        pickle.dump(X, open(X_file, "wb"))
        pickle.dump(y, open(y_file, "wb"))

    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

#with globalmax pool layer and dense layer after LSTM
def cnn_lstm_model1(sequence_length, embedding_dim):
    model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = NO_OF_CLASSES
    embedding_dim = EMBEDDING_DIM
    dropout_prob = (0.20, 0.5)    

    # main sequential model
    model = Sequential()
    #if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(dropout_prob[0]))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Convolution1D(100, FILTER_SZ, activation='relu'))   
    model.add(MaxPooling1D(pool_length=4))  #pool_length in keras 1.2.2 not pool_size   
    model.add(LSTM(50,return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])            
    print (model.summary())
    return model

#with dropout layer after lstm
def cnn_lstm_model(sequence_length, embedding_dim):
    model_variation = 'CNN-LSTM'  #  CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = NO_OF_CLASSES
    embedding_dim = EMBEDDING_DIM   
    dropout_prob = (0.20, 0.5)

    # main sequential model
    model = Sequential()
    #if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(dropout_prob[0]))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Convolution1D(100, FILTER_SZ, activation='relu'))   
    model.add(MaxPooling1D(pool_length=4))  #pool_length in keras 1.2.2 not pool_size  
    model.add(LSTM(50))    
    model.add(Dropout(dropout_prob[1]))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])           
    print (model.summary())
    return model


def train_CNN_LSTM(X, y, inp_dim, model, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            shuffle_weights(model)
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print("ERROR!")
            return

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0
            for i, X_batch in enumerate(batch_gen(X_temp, batch_size), 1):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len] # Last column  will be y due to np.hstack

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                    class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

                try:
                    y_temp = np_utils.to_categorical(y_temp, nb_classes=3)
                except Exception as e:
                    print(e)
                    print(y_temp)

                _loss, _acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                train_loss += _loss
                train_acc += _acc
                if i % 35 == 0:
                    print("Epoch: %d/%d.\tBatch: %d.\tLoss: %f.\tAccuracy: %f" % (epoch,epochs, i, train_loss / i, train_acc/i))

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print("\n", classification_report(y_test, y_pred))
        p += precision_score(y_test, y_pred, average='weighted')
        r += recall_score(y_test, y_pred, average='weighted')
        f1 += f1_score(y_test, y_pred, average='weighted')

    print("weighted results are")
    print("average precision is %f" %(p/NO_OF_FOLDS))
    print("average recall is %f" %(r/NO_OF_FOLDS))
    print("average f1 is %f" %(f1/NO_OF_FOLDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN based models for twitter Hate speech detection')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)
    parser.add_argument('--kernel', default=FILTER_SZ)
    args = parser.parse_args()

    GLOVE_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function
    FILTER_SZ = int(args.kernel)



    print('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print('Embedding Dimension: %d' %(EMBEDDING_DIM))
    print('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE, binary=False)
    np.random.seed(SEED)


    tweets = select_tweets()  # Get tweets which contain at least one word with an embedding.
    gen_vocab()
    #filter_vocab(20000)

    X, y = gen_sequence()

    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    model = cnn_lstm_model(data.shape[1], EMBEDDING_DIM)
    train_CNN_LSTM(data, y, EMBEDDING_DIM, model, W, EPOCHS)
    print("Saving model...")
    model.save('cnn_lstm_' + INITIALIZE_WEIGHTS_WITH + '.h5')
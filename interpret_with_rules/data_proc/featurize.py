from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import operator

def get_train_test_split(x, y, test_ratio = 0.1, seed = 0):
    '''
    Stratified split of training data_proc into training and validation sets
    :param x: original training feats
    :param y: original training labels
    :return: new train feats, test feats, train labels, and test labels
    '''
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_ratio, random_state = seed)
    for train_idx, val_idx in sss.split(x, y):
        x_train, y_train = [x[i] for i in train_idx.tolist()], y[train_idx]
        x_val, y_val = [x[i] for i in val_idx.tolist()], y[val_idx]

    return x_train, x_val, y_train, y_val

def vectorize_data(x_train, x_val, x_test):
    '''
    Featurizes the dataset to corresponding TF-IDF values
    :param x_train: training feats
    :param x_val: validation feats
    :param x_test: test feats
    :return: vocabulary (dictionary) and TF-IDF vectors (scipy sparse matrix)
    '''
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors_train = vectorizer.fit_transform(x_train)
    vectors_val = vectorizer.transform(x_val)
    vectors_test = vectorizer.transform(x_test)

    print("Vocabulary size: ",  len(vectorizer.vocabulary_))

    return vectorizer.vocabulary_, vectors_train, vectors_val, vectors_test

def select_best_feats(x_train, y_train, x_val, x_test, k = 1000):
    '''
    Select best feats
    :param x_train: training feature vector
    :param y_train: labels for training set
    :param x_val: validation features
    :param x_test: test features
    :param k: num of features to select
    :return: indices of selected features, and the modified training and test sets
    '''
    selector = SelectKBest(mutual_info_classif, k)
    x_train = selector.fit_transform(x_train, y_train)

    if x_val is not None:
        x_val = selector.transform(x_val)

    if x_test is not None:
        x_test = selector.transform(x_test)

    selected_indices = selector.get_support(indices = True)
    return x_train, x_val, x_test, selected_indices

def get_vocab_subset(vocab_dict, select_indices):
    '''
    Return a subset of vocabulary based on a list of vocabulary indices to keep
    :param vocab_dict: dictionary of {item:idx}
    :param select_indices: indices to keep in the vocabulary
    :return: subset of vocabulary only with the indices that are being kept
    '''
    sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    vocab_dict = {sorted_vocab[idx][0]: i for i, idx in enumerate(select_indices)}
    return vocab_dict

def select_best_feats_vocab(x_train, y_train, x_val, x_test, vocab_dict, k = 1000):
    '''
    Select best features and the corresponding subset of vocabulary
    :param x_train: training feature vector
    :param y_train: labels for training set
    :param x_val: validation features
    :param x_test: test features
    :param vocab_dict: dictionary of {item:idx}
    :param k: num of features to select
    :return:the modified training and test sets, and subset of the selected vocabulary
    '''
    new_x_train, new_x_val, new_x_test, sel_idx = select_best_feats(x_train, y_train, x_val, x_test, k)
    vocab = get_vocab_subset(vocab_dict, sel_idx)

    return new_x_train, new_x_val, new_x_test, vocab

def scale_feats(feats_train, feats_val = None, feats_test = None, feat_range=(0, 1)):
    '''
    Rescale data between 0 and 1
    :param feats_train: training data
    :param feats_val: validation data
    :param feats_test: test data
    :param feat_range: range of values for final feature points
    :return: rescaled training, val, test feats
    '''
    scaler = MinMaxScaler(feature_range=feat_range)
    feats_train = scaler.fit_transform(feats_train)
    if feats_val is not None:
        feats_val = scaler.transform(feats_val)
    if feats_test is not None:
        feats_test = scaler.transform(feats_test)

    return feats_train, feats_val, feats_test

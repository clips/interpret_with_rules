from data_proc.featurize import get_train_test_split, vectorize_data, select_best_feats, get_vocab_subset

from sklearn.datasets import fetch_20newsgroups

def _get_data(cats):
    '''
    Gets 20 newsgroups dataset after removing headers, footers and quotes
    :param cats: List of categories to limit to
    :return: train subset, test subset
    '''
    remove_cont = ('headers', 'footers', 'quotes')

    return fetch_20newsgroups(subset='train', remove=remove_cont, categories=cats), \
           fetch_20newsgroups(subset='test', remove=remove_cont, categories=cats)

def get_featurized_data(is_select_best = False, cats = None):
    '''
    Gathers and prepares the 20 newsgroups dataset
    :param is_select_best: True to select best features
    :param cats: list of categories to limit the dataset to
    :return: vocab dictionary {vocab_item:index}, list of class names in index order,
             feats_train, feats_val, feats_test,
             labels_train, labels_val, labels_test
    '''

    ds_train, ds_test = _get_data(cats)
    x_train, x_val, y_train, y_val = get_train_test_split(ds_train.data, ds_train.target)
    x_test, y_test = ds_test.data, ds_test.target

    vocab_dict, vectors_train, vectors_val, vectors_test = vectorize_data(x_train, x_val, x_test)

    if is_select_best:
        vectors_train, vectors_val, vectors_test, selected_indices = select_best_feats(vectors_train, y_train, vectors_val, vectors_test)
        get_vocab_subset(vocab_dict, selected_indices)

    return vocab_dict, ds_train.target_names, vectors_train, vectors_val, vectors_test, y_train, y_val, y_test

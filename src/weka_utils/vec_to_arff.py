import operator
import os

from scipy.sparse.base import issparse

def get_feat_dict(vocab_dict, vec_type):
    '''
    @todo: Support discrete features
    :param vocab_dict: dictionary {vocab_item, index}
    :param vec_type: (tf-idf/tf/rescaled) type of vectors
    :return: dictionary {feat_name:[possible_vals/data_type]}
    '''

    print("Getting feature dictionary")

    feat_list = _get_feat_names(vocab_dict)

    if vec_type in ['tf', 'tf-idf']:
        data_type = _get_cont_dtype(vec_type)
    elif vec_type in ['rescaled']:
        data_type = _get_discrete_type(vec_type)

    return {i:data_type for i in feat_list}

def write_arff_file(rel_name, feat_dict, class_names, data_vec, data_classes, dir_name, fname):
    _write_relation(rel_name, dir_name, fname)
    _write_attributes(feat_dict, class_names, dir_name, fname)
    _write_data(data_vec, data_classes, class_names, dir_name, fname)


def _write_relation(rel_name, dir_name, fname):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print("Writing relations...")
    with open(os.path.join(dir_name, fname), 'w') as f:
        f.write("@RELATION " + rel_name.replace(" ", "_") +"\n")
    print("Done!")

def _write_attributes(feat_dict, class_names, dir_name, fname):
    '''
    Creates an attribute file with all feature names, their data_proc types (if continuous) and their possible values
    (if discrete), and the possible classes.
    :param feat_dict: Dict of feature names in the order of featurized data_proc indices and list of their corresponding data_proc type or possible values
    :param class_names: List of class names in corresponding numerical order of label encoding
    :param dir_name: directory name for data_proc files
    :param fname: ripper attribute file name
    :return:
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print("Writing attributes...")
    with open(os.path.join(dir_name, fname), 'a') as f:
        for feat, vals in feat_dict.items():
            _write_feat_attr(feat, vals, f)
        _write_class_attr(class_names, f)
    print("Done!")

def _get_feat_names(vocab_dict):
    '''
    Returns list of feature names in increasing order of their indices
    :param vocab_dict: either a dictionary of vocab items and their index, or a trained sklearn vectorizer object
    :return: list
    '''
    # print("Getting feature names")
    return [k for k, v in sorted(vocab_dict.items(), key=operator.itemgetter(1))]

def _get_discrete_type(vec_type):
    '''
    Returns the set of values of discrete features.
    :param vec_type: description of vector generation method (rescaled)
    :return: string representing all values of a feature type
    '''
    # print("Getting feature types")
    type_dict = {'rescaled': ['-1', '0', '1']}

    try:
        return type_dict[vec_type]
    except KeyError:
        print('Continuous feature value description not recognized. Please enter the correct type')

def _get_cont_dtype(vec_type):
    '''
    Returns the data_proc type as string (float/int) if TF-IDF or count vectorization technique has been used for continuous feature generation.
    Extend this function to add other vectorization methods
    :param vec_type: description of vector generation method (tf-idf/tf/rescaled)
    :return: string representing feature type
    '''
    # print("Getting feature types")
    type_dict = {'tf-idf': 'NUMERIC', 'tf': 'NUMERIC'}

    try:
        return [type_dict[vec_type]]
    except KeyError:
        print('Continuous feature value description not recognized, returning NUMERIC by default.')
        return ['NUMERIC']

def _write_feat_attr(feat, vals, f):
    if vals == ['NUMERIC'] or vals == ['numeric']:
        try:
            vals = [str(i).replace(" ", "_") for i in vals]
        except ValueError:
            print("Please enter valid values")
        vals = ",".join(vals)
    else:
        vals = '{' + ','.join(vals) + '}'

    f.write("@ATTRIBUTE " + feat.replace(" ", "_") + " " + vals + "\n")

def _write_class_attr(class_names, f):
    try:
        class_names = [str(i).replace(" ", "_") for i in class_names]
    except ValueError:
        print("Please enter valid class names")
    else:
        f.write("@ATTRIBUTE text_class {"+ ",".join(class_names)+"}\n")


def _write_data(vec, class_labels, class_names, dir_name, fname):
    '''
    Create a file (can be train or test data_proc) with feature values and class labels
    :param vec: numpy matrix or 2D list with every row indicating one instance, and every column indicating a feature
    :param class_labels: list of class labels (indices) for all instances
    :param class_names: list with class names in the order of indices of class labels
    :param dir_name: directory name for data_proc files
    :param fname: text file name to write features and class labels to
    '''

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(os.path.join(dir_name, fname), 'a') as f:
        f.write("@DATA\n")
        for feats, class_label in zip(vec, class_labels):
            _write_instance(feats, class_names[class_label], f)

def _write_instance(feats, class_label, f):

    if issparse(feats):
        feats = feats.toarray().reshape(-1,)

    try:
        feat_list = [str(i).replace(" ","_") for i in feats]
    except ValueError:
        print("Please enter correct feature values")
    else:
        try:
            class_label = str(class_label)
        except ValueError:
            print("Please enter correct class label")
        else:
            f.write(",".join(feat_list)+","+class_label+"\n")

def create_weka_files(vocab_dict, class_names, x_train, x_val, x_test, y_train, y_val, y_test, data_prefix, vec_type = 'tf-idf', data_dir ='../data/'):
    '''
    Create the data_proc files for ripper compatibility
    :param vocab_dict: dictionary of {vocab_item: index}
    :param class_names: list of class names that can occur in the dataset in increasing index order
    :param x_train: training feats (sparse/dense matrix with rows as instances)
    :param x_test: testing feats (sparse/dense matrix with rows as instances)
    :param y_train: training class labels (names) list
    :param y_test: testing class labels (names) list
    :param data_prefix: string prefix indicating details of data_proc, e.g., dataset name and vector type
    :param vec_type: type of feature vectors ('tf-idf' | 'tf' | 'rescaled')
    :param data_dir: directory path for input data_proc files in model input formats
    '''

    feat_dict = get_feat_dict(vocab_dict, vec_type)
    print("Writing train file...")
    write_arff_file(data_prefix, feat_dict, class_names, x_train, y_train, data_dir, data_prefix + '-train.arff')
    print("Done")
    print("Writing val file...")
    write_arff_file(data_prefix, feat_dict, class_names, x_val, y_val, data_dir, data_prefix + '-val.arff')
    print("Done")
    print("Writing test file")
    write_arff_file(data_prefix, feat_dict, class_names, x_test, y_test, data_dir, data_prefix + '-test.arff')
    print("Done")
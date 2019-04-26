from nn.dataset import SparseDataset
from nn.gradients import get_out_gradient, get_max_rms_gradient
from nn.utils import get_dataloader

from data_proc.featurize import select_best_feats, scale_feats, get_vocab_subset
from weka_utils.vec_to_arff import get_feat_dict, write_arff_file


def get_model_transformed_input(net, x_test, y_test_pred,
                                vocab_dict, class_labels,
                                out_fname_prefix,
                                x_train, y_train = None,
                                feat_sel_algo = 'sa', k = 1000,
                                data_dir = '../data/'
                                ):
    """
    Write the transformed input space along with labels (predicted for test, gold for train) to explain a trained model.
    This output is in the Weka format.
    :param net: trained, differentiable model
    :param x_test: test input features
    :param y_test_pred: model test predictions
    :param vocab_dict: feature vocabulary dictionary of {item:idx}
    :param class_labels: list of class names for each class index
    :param out_fname_prefix: Prefix for output jrip files.
    :param feat_sel_algo: (sa|mi) Sensitivity analysis or mutual information feature selection
    :param k: number of features to select
    :param x_train: training features.
    :param y_train: training labels. Required for mi feature selection.
    :param data_dir: output directory for transformed input files in weka format.
    """

    if y_train is None and feat_sel_algo == 'mi':
        raise ValueError("Please provide training labels to use 'mi' feature selection. Use 'sa' otherwise")

    x_test_pred, x_train_gold = get_reweighed_data(net, x_test, y_test_pred, x_train, y_train)

    out_fname_prefix = out_fname_prefix + '-reweighed-' + feat_sel_algo

    #convert features to their corresponding sign
    x_test_pred = reduce_to_sign(x_test_pred)
    x_train_gold = reduce_to_sign(x_train_gold)

    if feat_sel_algo == 'sa':
        vocab_dict, x_test_pred, x_train_gold = get_most_sensitive_feats(net, x_train, vocab_dict, k, x_test_pred, x_train_gold)

    elif feat_sel_algo == 'mi':
        vocab_dict, x_test_pred, x_train_gold = get_best_feats_mi(k, x_train_gold, y_train, vocab_dict, x_test_pred)

    write_transformed_data(vocab_dict, class_labels, out_fname_prefix, data_dir,
                           x_test_pred, y_test_pred,
                           x_train_gold, y_train)

def get_reweighed_data(net, x_test, y_test_pred,  x_train, y_train):
    """
    Transform the input data according to the trained model
    :param net: Trained model
    :param x_test: Test features
    :param y_test_pred: Test predictions
    :param x_train: Training features
    :param y_train: Training labels
    :return: Transformed test and train features
    """
    print("Getting reweighed feats for test predictions...")
    x_test_pred = get_reweighed_feats(net, x_test, y_test_pred)
    print("Done")

    if y_train is not None:
        print("Getting reweighed feats for train gold...")
        x_train_gold = get_reweighed_feats(net, x_train, y_train)
        print("Done")
    else:
        x_train_gold =  None

    return x_test_pred, x_train_gold


def get_reweighed_feats(net, feats, labels = None):
    '''
    Compute reweighed input in a trained model by multiplying the original feats with gradient weights.
    These gradient features are treated as input importance.
    :param net: trained model
    :param feats: original feats
    :param labels: labels to get gradients of
    :return: reweighed feature matrix (supports scipy sparse) and the training gradients
    '''
    gradients = get_gradients(net, feats, labels)
    feats = reweigh_feats(feats, gradients)
    return feats

def get_gradients(net, feats, labels = None):
    '''
    Compute gradients of output wrt input for every instance and feature pair
    :param net: trained model
    :param feats: original feats
    :param labels: labels to get gradients for
    :return: reweighed feature matrix (supports scipy sparse) and the training gradients
    '''

    ds = SparseDataset(feats, labels)
    dataloader = get_dataloader(ds, shuffle=False)
    gradients = get_out_gradient(net, dataloader)
    return gradients


def reweigh_feats(feats, weights):
    '''
    Reweigh feats according to their importance scores. These importance scores can be computed using any technique.
    :param feats: Features that need to be reweighed.
    :param weights: Importance weights of features for every instance
    :return: Features weighted according to their importance
    '''
    return feats.multiply(weights)

def get_most_sensitive_feats(net, x_train, vocab_dict, k, x_test_pred, x_train_gold):
    """
    Select the top features based on sensitivity analysis, which is an unsupervised technique.
    The features are selected based only on parameters of a trained model.
    :param net: Trained model
    :param x_train: Original input training features
    :param vocab_dict: vocabulary dictionary {item:idx}
    :param k: number of features to select
    :param x_test_pred: Input test features
    :param x_train_gold: Transformed input train features
    :return: Subset of vocabulary, and features
    """
    print("Getting the most sensitive features of the trained model for the training set")
    sensitivity, most_sens_idx = get_feat_sensitivity(net, x_train)

    k_most_sens_idx = most_sens_idx[:k]

    vocab_dict = get_vocab_subset(vocab_dict, k_most_sens_idx)
    x_test_pred = x_test_pred[:, k_most_sens_idx]

    if x_train_gold is not None:
        x_train_gold = x_train_gold[:,k_most_sens_idx]

    return vocab_dict, x_test_pred,  x_train_gold

def get_feat_sensitivity(net, data):
    '''
    Calculate feature sensitivity scores from a trained neural network.
    :param net: Trained network
    :param data: Data (containing feature values of different instances)
    :return: Sensitivity scores of all the features in the data based on the trained model (1D array),
    :return: 1D array with indices of the top scoring features in decreasing order
    '''
    ds = SparseDataset(data)
    dataloader = get_dataloader(ds, shuffle=False)
    gradients, max_grad_idx = get_max_rms_gradient(net, dataloader)
    return gradients, max_grad_idx

def print_k_most_sensitive_feats(most_sens_idx, vocab_dict, k = 100):
    '''
    Print the top most sensitive features
    :param most_sens_idx: 1D array containing indices of top scoring features in decreasing order
    :param vocab_dict: {vocab_item: index} mapping
    :param k: number of features to print
    '''
    rev_vocab_dict = {v:k for k, v in vocab_dict.items()}

    if k > most_sens_idx.shape[0]:
        k = most_sens_idx.shape[0]

    print("The {} most sensitive features are: ".format(k))
    for i in range(k):
        print(rev_vocab_dict[most_sens_idx[i]])

def get_best_feats_mi(k, x_train_gold, y_train, vocab_dict, x_test_pred):
    """
    Select the best features based on mutual info between transformed features and gold training labels
    :param k: number of features to select
    :param x_train_gold: transformed training input features
    :param y_train: training gold labels
    :param vocab_dict: vocabulary dictionary {item:idx}
    :param x_test_pred: transformed test features
    :return: subset of vocabulary and features
    """
    x_train_gold, __, x_test_pred, sel_idx = select_best_feats(x_train_gold, y_train, None, x_test_pred, k)
    vocab_dict = get_vocab_subset(vocab_dict, sel_idx)

    return vocab_dict, x_test_pred, x_train_gold

def reduce_to_sign(feats):
    '''
    Reduce a feature value to its sign. The resulting values will be in the set {-1,0,1}.
    :param feats: Features to discretize
    :return: Corresponding signs of feature values
    '''
    return feats.sign().astype(int)


def write_transformed_data(vocab_dict, class_labels, rel_name, data_dir,
                           x_test_pred, y_test_pred,
                           x_train_gold, y_train):
    """
    Write the transformed input required to induce rules in the format for weka.
    For test data, the transformed features are coupled with model test predictions.
    For training data, the transformed features are coupled with gold labels.
    :param vocab_dict: vocab dict {item:idx}
    :param class_labels: list of class names corresponding to indices
    :param rel_name: relation name
    :param data_dir: output directory for weka files
    :param x_test_pred: transformed test input features
    :param y_test_pred: test predictions
    :param x_train_gold: transformed training features
    :param y_train: training gold labels
    """
    print("Writing reweighed data files for JRIP input")

    feat_dict = get_feat_dict(vocab_dict, vec_type = 'rescaled')
    print("Writing transformed test prediction data")
    write_arff_file(rel_name, feat_dict, class_labels, x_test_pred, y_test_pred, data_dir, rel_name + '-test-pred.arff')

    if x_train_gold is not None:
        print("Writing transformed training gold data")
        write_arff_file(rel_name, feat_dict, class_labels, x_train_gold, y_train, data_dir,
                        rel_name + '-train-gold.arff')

from weka_utils.vec_to_arff import get_feat_dict, write_arff_file
from data_proc.featurize import select_best_feats_vocab

def get_reduced_train_input(x_train, y_train, vocab_dict, class_labels, out_fname_prefix, data_dir ='../data/', k = 1000):
    """
    Select the top features of the training set based on mutual information between features and labels.
    Write the top features to weka format.
    :param x_train: train set features
    :param y_train: labels train
    :param vocab_dict: vocabulary {item:idx}
    :param class_labels: list of class labels corresponding to their index
    :param out_fname_prefix: dataset properties for outfile name
    :param data_dir: directory to store arff file to
    """
    x_train = x_train.sign().astype(int)
    
    x_train, __, __, vocab_dict = select_best_feats_vocab( x_train, y_train, None, None, vocab_dict, k)
  
    out_fname_prefix += '-mi'

    print("Writing original train set for JRIP input")

    feat_dict = get_feat_dict(vocab_dict, vec_type='rescaled')
    print("Writing transformed test prediction data")
    write_arff_file(out_fname_prefix, feat_dict, class_labels, x_train, y_train, data_dir, out_fname_prefix + '-train-gold.arff')



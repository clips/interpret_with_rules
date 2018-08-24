from data_proc.dataset.newsgroups import get_featurized_data
from data_proc.training import train_baseline
from data_proc.prediction import pred_baseline
from evaluation.score import get_f1_score
from dataset_rules import get_reduced_train_input
from gradient_rules import get_model_transformed_input
from data_proc.training import train_nn, load_nn
from data_proc.prediction import pred_nn
from rule_induction.jrip import induce_ripper_rules

import argparse

def main(dataset = 'newsgroups', get_baseline = True, test_type = 'test',
         rule_type = 'gradient', feat_sel_algo = 'sa',
         load_trained_model = False, model_name = 'nn-model.tar', get_train_rules = True, data_dir = '../data/', dir_out = '../out/'):
    '''
    Model Interpretability pipeline
    '''

    #get featurized data. @todo: This part should be updated for new datasets
    if dataset == 'newsgroups':
        print("Getting featurized data for newsgroups dataset")

        # cats = ['alt.atheism', 'soc.religion.christian']
        cats = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']

        vocab_dict, class_labels, x_train, x_val, x_test, y_train, y_val, y_test = get_featurized_data(
            is_select_best=False, cats = cats)
    else:
        raise ValueError("Please enter the correct dataset to process. Currently supported: (newsgroups). ")

    if get_baseline:
        if test_type not in ['val', 'test']:
            raise ValueError("Please enter correct test type (val|test).")

        test_feats = {'val':x_val, 'test':x_test}
        test_labels = {'val':y_val, 'test': y_test}

        run_baseline(x_train, y_train, test_feats[test_type], test_labels[test_type], test_type, dataset+'-baseline.pkl', dir_out)

    if rule_type == 'trainset':
        print("Inducing rules from the original training data")
        data_prefix = dataset
        get_reduced_train_input(x_train, y_train, vocab_dict, class_labels, data_prefix, data_dir )

        induce_ripper_rules(data_prefix + '-mi-train-gold.arff', data_dir=data_dir)

    elif rule_type == 'gradient':
        print("Inducing rules from the data reweighed according to feature importance in a neural network")

        #Pytorch model parameters. @todo: modify these params based on required model
        net_params = {
            'n_hid_layers': 2,
            'n_hid': 100,
            'n_in': x_train.shape[1],
            'n_out': len(class_labels)
        }

        data_prefix = dataset

        if load_trained_model:
            net = load_nn(model_name, dir_out)
        else:
            net = train_nn(x_train, y_train, net_params, model_name, dir_out)

        if test_type == 'val':
            y_val_pred, __ = pred_nn(net, x_val)
            print("Macro average F-score of neural net on validation data: ", get_f1_score(y_val, y_val_pred))

        y_test_pred, __ = pred_nn(net, x_test)
        print("Macro average F-score of neural net on test data: ", get_f1_score(y_test, y_test_pred))

        if feat_sel_algo not in ['sa', 'mi']:
            print("Unsupported feature selection algorithm passed. Use (sa|mi). Using sensitivity analysis by default.")
            feat_sel_algo = 'sa'

        get_model_transformed_input(net, x_test, y_test_pred,
                                    vocab_dict, class_labels,
                                    data_prefix,
                                    feat_sel_algo = feat_sel_algo,
                                    data_dir = data_dir,
                                    x_train=x_train, y_train=y_train) #last parameter is optional for SA when train rules are not required

        induce_ripper_rules(data_prefix + '-reweighed-' + feat_sel_algo + '-test-pred.arff', data_dir =  data_dir)
        
        if get_train_rules:
            induce_ripper_rules(data_prefix + '-reweighed-' + feat_sel_algo + '-train-gold.arff', data_dir=data_dir)


def run_baseline(x_train, y_train, x_test, y_test, test_type, fout, dir_out):
    print("Establishing ", test_type, " baseline: ")
    model = train_baseline(x_train, y_train, fout, dir_out)
    pred_class = pred_baseline(model, x_test)
    score = get_f1_score(y_true = y_test, y_pred = pred_class)
    print("Macro average F1 score on ", test_type," data: ", score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-r", dest='rule_type', nargs='?', const = 'gradient',
                        choices = ['trainset', 'gradient'],
                        help="Whether to induce rules from the original training data (trainset),"
                             "or from the trained model (gradient).",
                        required=True, default='gradient')

    parser.add_argument("-f", dest='feature_sel',
                        choices = ['sa', 'mi'],
                        help="Feature selection algorithm to use (mi|sa) when using gradient rule_type.",
                        required=False, default= 'sa')

    parser.add_argument('-loadmodel',
                        choices = ['True', 'False'],
                        required = True,
                        help = "True to load a pretrained model, False otherwise.")

    parser.add_argument("-m", dest = 'modelname',
                        nargs='?', const = 'nn-model.tar',
                        help="Pretrained model name. If unavailable, new model will be saved with this name.",
                        required=True, default = 'nn-model.tar')

    parser.add_argument("-outdir",
                        help="Path to the directory for the trained model.",
                        required=False, default = '../out/')

    parser.add_argument("-datadir",
                        help="Path to the directory where to store weka files.",
                        required=False, default= '../data/')

    parser.add_argument('--notrainrules', action='store_false',
                        help = "Use this option to suppress rule induction from training data.")

    parser.add_argument('--baseline', action='store_true',
                        help = "Use this option to get baseline classification performance using Logistic Regression.")

    parser.set_defaults(feature_sel='sa', outdir = '../out/', datadir = '../data/')

    args = parser.parse_args()

    rule_type = args.rule_type

    feat_sel_algo = args.feature_sel

    load_trained_model = (args.loadmodel == 'True')

    model_name = args.modelname

    out_dir = args.outdir

    data_dir = args.datadir

    get_train_rules = args.notrainrules

    get_baseline = args.baseline

    main(rule_type = rule_type, feat_sel_algo = feat_sel_algo,
         load_trained_model = load_trained_model, model_name = model_name,
         get_train_rules = get_train_rules,
         get_baseline=get_baseline,
         dir_out = out_dir, data_dir = data_dir)


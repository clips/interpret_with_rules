import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.filters import Filter

import math
import numpy as np
from itertools import combinations


def start_jvm():
    jvm.start()

def stop_jvm():
    jvm.stop()

def load_data(fname, dir_in = "../data/", incremental = False):
    """
    Loads data in weka format
    :param fname: filename for data
    :param dir_in: input directory
    :param incremental: True to read data incrementally.
    :return: The data and the loader object
    """
    loader = Loader(classname="weka.core.converters.ArffLoader")
    if incremental:
        data = loader.load_file(dir_in + fname, incremental=incremental)
    else:
        data = loader.load_file(dir_in + fname)
    data.class_is_last() #Required to specify which attribute is class attribute. For us, it is the last attribute.

    return data, loader

def merge_classes(data, idx_to_merge):
    """
    :param data: The data file to filter
    :param idx_to_merge: String representation of class indices to merge 
    :return: filtered data
    """
    merge_filter = Filter(classname="weka.filters.unsupervised.attribute.MergeManyValues",
                          options=["-C", "last", "-R", idx_to_merge, "-unset-class-temporarily"])
    merge_filter.inputformat(data)
    filtered_data = merge_filter.filter(data)
    return filtered_data


def get_classifier(min_no, seed):
    """
    Return the classifier object given the options
    :param min_no: Minimum number of instances correctly covered by JRIP
    :param seed: Seed for randomizing instance order
    :return: classifier object
    """
    cls = Classifier(classname="weka.classifiers.rules.JRip")
    options = list()
    options.append("-N")
    options.append(str(min_no))
    options.append("-S")
    options.append(str(seed))

    cls.options = options
    return cls

def build_classifier(data, cls, incremental = False, loader = None):
    """
    Build classifier from the corresponding data
    :param data: weka data object
    :param cls: classifier object
    :param incremental: True if data is loaded incrementally
    :param loader: if incremental, the loader to load data
    :return: classifier
    """

    if incremental and loader is None:
        raise ValueError("Please enter a dataloader if incremental model")

    cls.build_classifier(data)

    if incremental:
        for inst in loader:
            cls.update_classifier(inst)

    return cls

def evaluate_classifier(cls, data, crossvalidate = False, n_folds = 10):
    """
    Evaluation
    :param cls: trained classifier
    :param data: data to test the model on
    :param crossvalidate: True to use crossvalidation
    :param n_folds: number of folds to cross validate for
    :return: evaluation object
    """
    evl = Evaluation(data)
    if crossvalidate:
        evl.crossvalidate_model(cls, data, n_folds, Random(5))
    else:
        evl.test_model(cls, data)

    return evl

def optimize_rule_params(data, incremental, dataloader, class_index = None):
    """
    Iterate over different parameter values and train a rule induction model. The best parameters are retained.
    :param data: Data to use for training and evaluating
    :param incremental: True if data is loaded incremetally
    :param dataloader: Data loader object if incremental is True
    :param class_index: Index of the class to compute F-score. None gives a macro-averaged F-score.
    """
    stats = data.attribute_stats(data.class_index)
    min_inst = min(stats.nominal_counts)
    print("Number of instances in the minority class: {}".format(min_inst))

    print("Optimizing over RIPPERk parameters")

    best_n, best_seed, best_model, best_eval, best_score = None, None, None, None, None

    # start_n = math.floor(0.01*min_inst)
    start_n = 2
    # seeds = np.random.randint(0, 5000, 50)
    seeds = np.arange(0, 50, 1) #analyzing performance for 50 seeds

    for seed in seeds:
        # seed = int(seed)
        for n in range(start_n, min_inst, 1):

            cls = get_classifier(n, seed)
            cls = build_classifier(data, cls, incremental, dataloader)

            evl = evaluate_classifier(cls, data, crossvalidate=False)

            if class_index is None:
                cur_score = evl.unweighted_macro_f_measure
            else:
                cur_score = evl.f_measure(class_index)

            if math.isnan(cur_score):
                break  # don't iterate to higher N value if current value covers zero instances for any class.

            # print("Unweighted macro f-measure for N {} and seed {}: {} \n".format(n, seed, cur_score))

            if best_eval is None or cur_score >= best_score:
                best_model = cls
                best_eval = evl
                best_n = n
                best_seed = seed
                best_score = cur_score

    print("Final results: ")
    print("Best performance found for N {} and seed {}".format(best_n, best_seed))
    print("Corresponding model: ", best_model)
    print("Corresponding results: ", best_eval.summary())
    print("Corresponding precision, recall, F-score (of the concerned class for multiclass, unweighted macro f-score for binary): ",
          best_eval.precision(class_index),
          best_eval.recall(class_index),
          best_score)
    print("Corresponding confusion matrix: ", best_eval.confusion_matrix)

def induce_ripper_rules(data_file, data_dir='../data/', out_file = 'ripperk_rules.out', out_dir='../out/jrip/'):
    """
    Induce the rules using RIPPERk
    :param data_file: File contaning training data in arff format
    :param data_dir: directory path for input file
    :param out_file: Filename to write the output model to
    :param out_dir: Directory to write the output file in
    """

    start_jvm()

    try:
        incremental = False
        data, dataloader = load_data(data_file, data_dir, incremental=incremental)

        n_classes = data.get_instance(0).num_classes
        print("Found {} classes".format(n_classes))

        if n_classes > 2: #onevsrest setup for more than 2 classes
            class_list = [str(i) for i in range(1,n_classes+1, 1)]
            for to_merge in combinations(class_list, n_classes-1):
                print("Merging classes ", to_merge)
                new_data = merge_classes(data, ','.join(to_merge))

                optimize_rule_params(new_data, incremental, dataloader, 0) #merged attribute is always the last one, so 0 index for desired class
        else:
            optimize_rule_params(data, incremental, dataloader) #normal learning for binary cases

    except Exception as e:
        print(e)
    finally:
        stop_jvm()

    # f_model = out_prefix+'-model.dat'
    # f_out = out_prefix+'-results.txt'
    #
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

if __name__ == '__main__':

    induce_ripper_rules('newsgroups-reweighed-sa-test-pred.arff',
                      data_dir='/home/madhumita/PycharmProjects/nn_interpretability/data/final_blackboxnlp/')


import os
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
# from weka.filters import Filter
# from weka.classifiers import FilteredClassifier

import math

def start_jvm():
    jvm.start()

def stop_jvm():
    jvm.stop()

def load_data(fname, dir_in = "../data/", incremental = True):
    loader = Loader(classname="weka.core.converters.ArffLoader")
    if incremental:
        data = loader.load_file(dir_in + fname, incremental=incremental)
    else:
        data = loader.load_file(dir_in + fname)
    data.class_is_last()

    return data, loader

# def apply_filter():
#     numeric_transfrom = Filter(classname="weka.filters.unsupervised.attribute.NumericTransform", options=["-M", "signum", "R", "1-1000"])
#     numeric_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal")
#     fc = FilteredClassifier()
#     fc.filter = numeric_transfrom

def get_classifier(min_no, seed):
    cls = Classifier(classname="weka.classifiers.rules.JRip")
    # options = ["-N", "25.0"] #-N: minNo, -F folds, -O num optimizations, -batch-size, -S: seed
    options = list()
    options.append("-N")
    options.append(str(min_no))
    options.append("-S")
    options.append(str(seed))

    cls.options = options
    return cls

def build_classifier(data, cls, incremental = True, loader = None):

    if incremental and loader is None:
        raise ValueError("Please enter a dataloader if incremental model")

    cls.build_classifier(data)

    if incremental:
        for inst in loader:
            cls.update_classifier(inst)

    return cls

def evaluate_classifier(cls, data):
    evl = Evaluation(data)
    evl.test_model(cls, data) #@todo: check if needed
    return evl
    # evl.summary()

def jrip_seed_search(train_file, test_file, out_prefix = None, data_dir='../data/', out_dir='../out/jrip/'):

    print("Learning rules using JRIP: ")
    incremental = False
    start_jvm()
    data_train, dataloader = load_data(train_file, data_dir, incremental)

    stats = data_train.attribute_stats(data_train.class_index)
    min_inst = min(stats.nominal_counts)

    print("Optimizing over minimum number of correct instances and seed")

    # if min_inst >= 5:
    #     start_n = 5
    # else:
    #     start_n = min_inst

    start_n = 2

    start_n = 5
    min_inst = 6

    best_n, best_seed, best_model, best_eval = None, None, None, None

    for seed in range(0, 1000, 10):
        for n in range(start_n, min_inst, 3):
            print("N{}, Seed {} \n".format(n, seed))

            cls = get_classifier(n, seed)
            cls = build_classifier(data_train, cls, incremental, dataloader)

            print("Trained classifier: \n", cls)

            evl = evaluate_classifier(cls, data_train)
            cur_score = evl.unweighted_macro_f_measure

            if math.isnan(cur_score):
                break #don't iterate to higher N value if current value covers zero instances for any class.
            print(evl.summary())
            print("Current Unweighted macro f-measure: {} \n".format(cur_score))

            if best_eval is None or cur_score >= best_eval.unweighted_macro_f_measure:
                best_model = cls
                best_eval = evl
                best_n = n
                best_seed = seed

    print("Best performance found for N {} and seed {}".format(best_n, best_seed))
    print("Corresponding model: ", best_model)
    print("Corresponding results: ", best_eval.summary())
    print("Corresponding macro f-score: ", best_eval.unweighted_macro_f_measure)
    print("Corresponding confusion matrix: ", best_eval.confusion_matrix)

    stop_jvm()

    # f_model = out_prefix+'-model.dat'
    # f_out = out_prefix+'-results.txt'
    #
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

if __name__ == '__main__':
    # jrip_seed_search('newsgroups-2cats-reweighed-best-gold-train.arff',
    #                  'newsgroups-2cats-reweighed-best-gold-train.arff',
    #                  data_dir='/home/madhumita/PycharmProjects/nn_interpretability/data/blackboxnlpruns/atheism_christianity/max_abs_grads,mi1000/')

    # jrip_seed_search('newsgroups-2cats-tfidf-best-train.arff',
    #                  'newsgroups-2cats-tfidf-best-train.arff',
    #                  data_dir='/home/madhumita/PycharmProjects/nn_interpretability/data/blackboxnlpruns/atheism_christianity/dataset/')

    jrip_seed_search('newsgroups-2cats-reweighed-mostsens-gold-val.arff',
                     'newsgroups-2cats-reweighed-mostsens-gold-val.arff',
                     data_dir='/home/madhumita/PycharmProjects/nn_interpretability/data/blackboxnlpruns/atheism_christianity/max_abs_grads,sensitivity1000/')

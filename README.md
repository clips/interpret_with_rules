This project induces rules to explain the predictions of a trained neural network, and optionally also to explain the patterns that the model captures from the training data, and the patterns that are present in the original dataset. This code corresponds to the paper:

**Rule Induction for global explanation of trained models** <br/>
Madhumita Sushil, Simon Šuster and Walter Daelemans <br/>
Workshop on *Analyzing and interpreting neural networks for NLP (BlackboxNLP)*, EMNLP 2018


The packages required are listed in `requirements.txt`. These dependencies must be satisfied to use the code.

The code uses python3, and can be run as follows:

To first train a neural document classifier (for the Science categories in the 20 newsgroups dataset) and then induce rules to explain the classifier's predictions, the following command is used:

```
python3 main.py -r gradient -loadmodel False -m <modelname.tar>
```

If we want to induce rules to explain a pretrained network, we can set the `loadmodel` option to `True` and input the pretrained model name. For replicating the exact results of the paper, we have provided the model we have explained under the name `nn-model.tar`.

To induce rules to identify the patterns in the original training data, the following command is used:

```
python3 main.py -r trainset -loadmodel False -m
```

The complete description of options can be checked by using the `--help` option like:

```
python3 main.py --help
```

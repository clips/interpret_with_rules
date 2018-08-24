This project induces rules to explain test predictions of a trained differentiable model, and optionally also the training data and the trained model performance with respect to the training labels.
The packages required are listed in `requirements.txt`. These dependencies must be satisfied to use the code.

The code uses python3, and can be run as follows:

To induce rules using reweighed gradients from the trained model, the following command is used:

```
python3 main.py -r gradient -loadmodel False -m nn-model.tar
``` 

The loadmodel parameter can be toggled to load a pretrained model.

To induce rules from the original data irrespective of a trained model, the following command is used:

```
python3 main.py -r trainset -loadmodel False -m
```

The complete details of options can be checked by using the `--help` option like:

```
python3 main.py --help
```



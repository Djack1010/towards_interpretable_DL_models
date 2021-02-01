# Towards an Interpretable Deep Learning Model for Mobile Malware Detection and Family Identification

Repository to replicate the experiments presented in 'Towards an Interpretable Deep Learning Model for Mobile Malware Detection and Family Identification' by Iadarola G. et al.

If you are using this repository, please consider [**citing our works**](#publications) (see links at the end of this README file).

This repository contains the code to strictly replicate the experiments, but it is based on the [TAMI](https://github.com/Djack1010/tami) repository, which constitute the main repository.

## Getting Started

##### Ubuntu 20.04

You can run the script `install.sh` to set up all the necessary dependencies (excluding the GPU ones).
Then, you should install all the necessary libraries with `pip`
```
pip install -r requirements.txt 
```
Refers to [TAMI](https://github.com/Djack1010/tami) for further information and documentation on the code.

#### Usage

The tool can be run with the `main.py` and `main_cati.py` scripts. See the README file in the cati folder for further information on the `main_cati.py` script.

`main.py` usage:
```
python main.py --help
usage: main.py [-h] -m {DATA,CNN} -d DATASET [-o OUTPUT_MODEL] [-l LOAD_MODEL] [-t TUNING] [-e EPOCHS] [-b BATCH_SIZE] [-i IMAGE_SIZE] [-w WEIGHTS] [--mode MODE] [--exclude_top]
               [--caching]

Deep Learning Image-based Malware Classification

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m {DATA,CNN}, --model {DATA,CNN}
                        the model to test OR 'DATA' to initialize the dataset
  -d DATASET, --dataset DATASET
                        the dataset path, must have the folder structure: training/train, training/val and test,in each of this folders, one folder per class (see dataset_test)
  -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        Name of model to store
  -l LOAD_MODEL, --load_model LOAD_MODEL
                        Name of model to load
  -t TUNING, --tuning TUNING
                        Run Keras Tuner for tuning hyperparameters, chose: [hyperband, random, bayesian]
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -i IMAGE_SIZE, --image_size IMAGE_SIZE
                        FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input (reshape will be applied)
  -w WEIGHTS, --weights WEIGHTS
                        If you do not want random initialization of the model weights (ex. 'imagenet' or path to weights to be loaded), not available for all models!
  --mode MODE           Choose which mode run between 'train-val' (default), 'train-test', 'test' or 'gradcam'. The 'train-val' mode will run a phase of training and validation on the
                        training and validation set, the 'train-test' mode will run a phase of training on the training+validation sets and then test on the test set, the 'test' mode
                        will run only a phase of test on the test set. The 'gradcam' will run the gradcam analysis on the modelprovided.
  --exclude_top         Exclude the fully-connected layer at the top of the network (default INCLUDE)
  --caching             Caching dataset on file and loading per batches (IF db too big for memory)
```

Logs, figure and performance results are stored in `results` and `tuning` folders.
Tensorboard can be used to print graph of training and validation trend.
```
tensorboard --logdir results/tensorboard/fit/
```

## Authors & References

* **Giacomo Iadarola** - *main contributor* - [Djack1010](https://github.com/Djack1010) giacomo.iadarola(at)iit.cnr.it

<a name="publications"></a>
If you are using this repository, please cite our work by referring to our publications (BibTex format):
```
@article{iadarola2021towards,
  title={Towards an Interpretable Deep Learning Model for Mobile Malware Detection and Family Identification},
  author={Iadarola, Giacomo and Martinelli, Fabio and Mercaldo, Francesco and Santone, Antonella},
  journal={Computers \& Security},
  pages={102198},
  year={2021},
  publisher={Elsevier}
}
```

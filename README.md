Final Project in Evaluation Selection of RS School Machine Learning course.

This Priject uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for predicting the forest cover type (the predominant kind of tree cover)
from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell
was determined from US Forest Service (USFS) Region 2 Resource Information System data.
Independent variables were then derived from data obtained from the US Geological Survey and USFS.
The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.



1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the poetryfollowing command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```

Different train Score at this time:

http://127.0.0.1:5000

The intermediate results of the model training on Kaggle are as follows:

https://www.kaggle.com/competitions/forest-cover-type-prediction/submissions







# Machine Learning
Machine learning experiments, and tutorials based from Google's videos of [@random-forests](https://github.com/random-forests)

See full videos [here](https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)

Some of the code was not compatible with recent versions of TensorFlow (>= 1.6.0), which are updated in this repo.

## Requirements
- Python3 (any version about 3.0.0)
- Pip3 
Guide from Python's website [here](https://www.python.org/downloads/) or via brew for Mac Users [here](https://docs.brew.sh/Homebrew-and-Python)
- Pipenv Installation guide [here](https://packaging.python.org/tutorials/managing-dependencies/) 
If you are not familiar with pip3 and pipenv.
Pipenv is a dependency manager for Python, just as NPM for Node. `Pipfile` and `Pipfile.lock` are the equivalents of the package.json for Node apps.
`Pipfile` takes care of dependency management.
`Pipfile.lock` is used for building, testing, etc.

- Graphviz Installation guide [here](http://brewformulas.org/Graphviz)

- TensorFlow
Full guide from [TensorFlow website](https://www.tensorflow.org/install/])
Note: You can also do a global installation through pip.
```sh
pip3 install tensorflow
```

## Installation

### Clone the repository
```sh
git clone git@github.com:Kenny407/machine-learning.git
```
### Change directory to the repo
```sh
cd machine-learning/
```
### Install dependencies
```sh
pipenv install
```
### Run 🚀
```sh
python3 tutorials/visualizeTree.py
```
### Visualize the output
Open the pdf in the directory tutorials
```sh
open iris.pdf
```
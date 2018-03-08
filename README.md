# Machine Learning

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
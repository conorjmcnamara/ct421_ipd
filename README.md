# CT421 - Artificial Intelligence - Iterated Prisoner's Dilemma

A genetic algorithm implementation for evolving strategies to play the Iterated Prisoner's Dilemma.

## Student Details
Student Name: Conor McNamara

Student ID: 21378116

## Prerequisites
- Python 3.x

## Installation
Clone the repository and install the required dependencies:

```sh
git clone https://github.com/conorjmcnamara/ct421_ipd.git

pip install -r requirements/requirements.txt
```

To install development dependencies (for linting and testing), run:

```sh
pip install -r requirements/requirements.dev.txt
```

## Usage
### Running the Program
The Jupyter notebook, `notebooks/main.ipynb`, contains the entrypoints for the experiments.

Alternatively, run the experiments from the `src/main.py` script:
```sh
python -m src.main 
```

### Linting
```sh
flake8 .
```

### Testing
```sh
pytest
```
# ML Experiment Runner

A command-line tool for running and comparing ML training experiments.
Built with OOP design, type hints, and full test coverage.

## Features

- Run baseline and regularised experiments form the terminal
- Configurable learning rate, epochs, and regularisastion penalty
- Clean loss curve output per run
- 10 units tests with pytest

## Setup

'''bash
git clone https://github.com/Indolz/ml-experiment-runner.git
cd ml-experiment-runner
python -m venv venv
source venv/Scripts/activate    # Windows
pip install pytest
'''

## Usage
'''bash
# Baseline experiment
python main.py --name "baseline" --lr 0.01 --epochs 10

# With regularisation
python main.py --name "reg-run" --lr 0.01 --epochs 10 --regularise --lambda-reg 0.005

## Run Test
'''bash
pytest -v
'''
## Project Structure
ml-experiment-runner/
|-- experiment.py       # MLExperiment and RegularisedExperiment classes
|-- data_loader.py      # Generator-based data streaming utilities
|-- main.py             # CLI entry point
|-- test_classes.py     # 10 unit tests with paytest
|-- test_generator.py   # Generator tests

## Concepts Demonstrated
- Objected-oriented design with inheritance
- Type hints throughout
- Decorators (@timer)
- Generator funtions for memory-efficient data loading
- argparse CLI
- pytest unit testing 


# highfrek

## Install

### Ubuntu
To install on Ubuntu, first run the install.sh script.

### MacOs
You will need python 3.10. Create a new venv and install the requirements:

```
python3 -m venv venv
source venv/bin/activate

pip install wheel
pip install -r requirements.txt
```

Remember to activate your environment before launching a jupyter notebook.

## Getting started

In your jupyter notebook or jupyter lab, you can run the following to setup:

```
%load_ext autoreload
%autoreload 2


import os
import sys

module_path = os.path.abspath(os.path.join('../../highfrek'))
if module_path not in sys.path:
    sys.path.append(module_path)
```
### Loading data

To load a pickle file, simply use the `load` util function:

```
from utils.io_utils import load
features = load(os.path.join(data_dir, 'features.pkl'))
candles = load(os.path.join(data_dir, 'candles.pkl'))
aligned_slices = load(os.path.join(data_dir, 'aligned_slices.pkl'))
```

# Theoretical introduction

## Concepts

### Agent

An agent receives at a fixed frequency the latest state of the world.
This state is made out of:
* The last candle for the time series (open, close, low, high, volume at t)
* The current repartition of the portfiolo (amount of usd and btc owned at t)

It outputs a decision after receiving those two values. 
For details about the architecture of the class see `agents.abstract.AbstractAgent`

See `documentation/theory.md` for an explanation of the core ideas of the mathematical modelisation.


### Exchange

The Agent interacts with the exchange, putting in trades that the exchange then puts on the market.
The market dictates whether the transaction goes through or not. 
See `exchange_api.abstract.AbstractExchangeAPI` for details about the implementation.




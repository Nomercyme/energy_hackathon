# Project Setup

## Prerequisites

Make sure you have `uv` installed. If not, you can install it using the following command in the terminal:

```bash
pip install uv
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Nomercyme/energy_hackathon.git
    cd energy_hackathon
    ```

2. Install the dependencies:
    ```bash
    uv install -r requirements.txt
    ```

## Running the Scripts

To run the Python scripts, use the following command in the terminal:
```bash
uv run main.py
```

Alternatively, you can also just run the `main.ipynb` notebook, but make sure to select the correct Python interpreter!

## Adding packages

To install new Python packages to the environment, simply run:
```bash
uv add PACKAGE_NAME
```


## Notes
### To Add
- Index column
- Holidays FE
- Clear input format for LCP to rerun my code
- Model B, when aggregating don't just include summarization, but also actual halfhourly values as features, also averages of highest / lowest SPs -> Trader's logic when setting DFR prices; What's the highest / lowest export price I'm missing out on?
- Include some data from the EFA blocks before and after -> Traders will also be thinking "How much will it cost me to charge up/down to the Response Energy Volume?"

### Next-steps
- Make "long" format, x6 train dataset
- A model for each market 
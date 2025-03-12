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
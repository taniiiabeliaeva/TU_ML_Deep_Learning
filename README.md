# Text Generation Project

This repository contains the implementation of a text generation project using LSTM, Transforme machine learning models for next-word prediction. Follow the instructions below to set up your environment, install dependencies, and run the project.

## Setup Instructions

### 1. Create a Python Environment

Create a Python virtual environment to manage project dependencies. 
```sh
python -m venv myenv
```
Activate the virtual environment:
- On Windows:
```sh
myenv\Scripts\activate
```

- On macOS and Linux:
```sh
source myenv/bin/activate
```

### 2. Install Dependencies

Install the required dependencies from the `requirements.txt` file:
```sh
pip install -r requirements.txt
```


### 3. Configure Parameters

You can change the parameters for the models and training process in the `config.yaml` file. This file contains various settings that control the behavior of the training scripts.

### 4. Run the Script
Run the `run.py`to start the training process with giving parameters:
```sh
python run.py
```

### 5. Qualitative results analysis
Once the training phase is completed, to evaluate the predictions qualitatively, we include the Jupyter notebook `analysis.ipynb` where the trained models generate next-word predictions for given input sequences.


# **Text Generation Project**

This repository contains the implementation of a **text generation project** using **LSTM and Transformer-based models** for next-word prediction. The project includes model training, evaluation, and qualitative analysis of generated text.


## **Project Structure**
```
├── data/                # Stores processed datasets 
├── dataset/             # Scripts for dataset loading and preprocessing
├── models/              # Implementation of LSTM and Transformer models
├── trained_models/      # Directory for saving trained model checkpoints
├── trainer/             # Training utilities, learning rate scheduler, and early stopping
│   ├── trainer.py       
│   ├── scheduler.py     
├── analysis.ipynb       # Jupyter notebook for qualitative results analysis
├── config.yaml          # Configuration file for model and training parameters
├── main.py              # Main script for initializing and running the training process
├── run.py               # Script to start training with defined parameters
├── requirements.txt     # Dependencies needed for the project
└── README.md            # Project documentation
```


## **Setup Instructions**

### **1. Create a Python Environment**
To ensure a controlled environment for dependencies, create a Python virtual environment:
```sh
python -m venv myenv
```
Activate the virtual environment:

- **On Windows:**
  ```sh
  myenv\Scripts\activate
  ```
- **On macOS and Linux:**
  ```sh
  source myenv/bin/activate
  ```


### **2. Install Dependencies**
Install all required packages from the `requirements.txt` file:
```sh
pip install -r requirements.txt
```


### **3. Configure Parameters**
Modify the `config.yaml` file to adjust hyperparameters, dataset configurations, and training settings according to requirements.


### **4. Train the Models**
Run the `run.py` script to start the training process:
```sh
python run.py
```
This script trains both LSTM and Transformer models for next-word prediction.

- The **trainer** (located in `trainer/trainer.py`) handles training loops and model optimization.
- The **scheduler** (located in `trainer/scheduler.py`) adjusts the learning rate dynamically during training.


### **5. Evaluate and Analyze Results**
- Once training is complete, you can evaluate model performance using metrics such as **perplexity, accuracy, and loss**.
- For qualitative evaluation, use `analysis.ipynb` to generate next-word predictions based on input text sequences.

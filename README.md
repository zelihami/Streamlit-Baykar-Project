# Baykar AI Interface | Streamlit-Based Dashboard

This project was developed for **Baykar** as part of the **Artificial Intelligence Specialization Program Graduation Project**. It demonstrates how Streamlit can be used to build an interactive and intuitive AI-powered dashboard for evaluating and visualizing machine learning model performance.

## Project Description

The main goal of this project is to provide a **user-friendly web interface** for training, evaluating, and visualizing the performance of machine learning models. The interface was built using [Streamlit](https://streamlit.io/), allowing users to interactively select model configurations, view metrics, and explore results in real time.

Key features:

- Train and evaluate ML models through a visual interface  
- Dynamic selection of layer sizes, activation functions, and optimizers  
- Visualize training accuracy, loss, and model architecture  
- Streamlined UI for experimentation and hyperparameter tuning  
- Built-in performance metrics such as accuracy and classification reports  

## Technologies Used

- Python   
- Streamlit 
- PyTorch 
- NumPy / Pandas  
- Matplotlib / Plotly  
- Scikit-learn  
- Seaborn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zelihami/Streamlit-Baykar-Project.git
   cd Streamlit-Baykar-Project

## Project File Structure

To ensure the project runs correctly, the file structure should be as follows:

.
├──  main_app.py # Main application with homepage and navigation

├──  ml_methods.py # Machine Learning page

├──  dl_methods.py # Deep Learning page

├──  home.py # Homepage layout and content

├──  data.csv # Wisconsin Breast Cancer dataset

├──  requirements.txt # List of required libraries

└──  README.md

## Install required dependencies:

```bash
pip install -r requirements.txt
```

## Run project

```bash
streamlit run main_app.py
```


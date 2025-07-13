# Baykar AI Interface | Streamlit-Based Dashboard

This project was developed for **Baykar** as part of the **Artificial Intelligence Specialization Program Graduation Project**. It demonstrates how Streamlit can be used to build an interactive and intuitive AI-powered dashboard for evaluating and visualizing machine learning model performance.

## Project Description
The goal of this project is to enable users to configure, train, and visualize the performance of various AI models through a simple and intuitive interface. The dashboard includes options for customizing model layers, selecting activation functions, choosing optimizers, and reviewing performance metrics.

The user interface consists of **two main sections**:

1. **Machine Learning Section (ml_methods.py)**  
   - Offers an overview of the dataset and allows preprocessing operations.  
   - Applies classical machine learning algorithms (e.g., Random Forest, SVM, Logistic Regression).  
   - Provides interactive hyperparameter configuration for each model in real time.  
   - Aimed at understanding the impact of preprocessing and model selection on performance.

2. **Deep Learning Section (dl_methods.py)**  
   - Uses the same dataset but applies deep learning approaches.  
   - Users can dynamically build their own neural networks by adjusting:  
     - Number of layers  
     - Neurons per layer  
     - Activation functions  
     - Optimizer and learning rate  
   - This allows learners to experiment and observe how neural network architecture influences performance.


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


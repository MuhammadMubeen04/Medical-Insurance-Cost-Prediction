# Medical Insurance Cost Prediction

## Project Overview
This project predicts the medical insurance cost of an individual based on personal and lifestyle factors such as age, BMI, gender, number of children, smoking status, and region using **Linear Regression**.

## Problem Statement
Insurance companies need to estimate the expected medical cost of a customer to set fair premiums. Manually calculating this is difficult and time-consuming. This machine learning model automates the prediction process.

## Dataset
- **Source**: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File**: `insurance.csv`
- **Records**: 1338
- **Features**: age, sex, bmi, children, smoker, region
- **Target**: charges

## Project Structure
```
Medical_Insurance_Cost_Prediction/
│
├── insurance.csv                          # Dataset
├── Medical_Insurance_Cost_Prediction.ipynb # Complete Jupyter Notebook
├── streamlit_app.py                       # Streamlit Web Application
├── requirements.txt                       # Required libraries
├── README.md                              # Project documentation
│
└── src/
    ├── data_preprocessing.py              # Data loading & encoding
    ├── model_training.py                  # Model training & evaluation
    └── predict.py                         # Prediction function
```

## Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit

## How to Run the Streamlit App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## How to Run the Notebook
1. Open `Medical_Insurance_Cost_Prediction.ipynb` in Google Colab or Jupyter
2. Make sure `insurance.csv` is in the same folder
3. Run all cells

## Author
Your Name  
Artificial Intelligence Internship – Big Brains

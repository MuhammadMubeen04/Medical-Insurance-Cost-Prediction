import pandas as pd

def load_data(filepath="insurance.csv"):
    """Load the insurance dataset"""
    df = pd.read_csv(filepath)
    return df

def encode_features(df):
    """Encode categorical features into numerical form"""
    df = df.copy()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = df['region'].map({
        'southwest': 0,
        'southeast': 1,
        'northwest': 2,
        'northeast': 3
    })
    return df

def prepare_features(df):
    """Separate features and target variable"""
    X = df.drop('charges', axis=1)
    y = df['charges']
    return X, y

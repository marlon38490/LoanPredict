import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    
    df = df.dropna()
    
    df = pd.get_dummies(df, drop_first=True)
    
   
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if _name_ == "_main_":
    from data_collection import load_data
    
    data = load_data('../data/loan.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
   

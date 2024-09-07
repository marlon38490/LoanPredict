import pickle
from sklearn.linear_model import LogisticRegression

def train_model_one(X_train, y_train):
   
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
   
    with open('../models/model_one.pkl', 'wb') as file:
        pickle.dump(model, file)

if _name_ == "_main_":
    from data_preprocessing import preprocess_data
    from data_collection import load_data

    data = load_data('../data/loan.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    train_model_one(X_train, y_train)
    print("Model One Trained and Saved.")
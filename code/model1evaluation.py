import pickle
from sklearn.metrics import accuracy_score
from datetime import datetime

def evaluate_model_one(X_test, y_test):
   
    with open('../models/model_one.pkl', 'rb') as file:
        model = pickle.load(file)
    
    
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model One Accuracy: {accuracy}")
    
    
    with open('../documentation/model_one_performance.txt', 'a') as log_file:
        log_file.write(f"Evaluation Date: {datetime.now()}\n")
        log_file.write(f"Model One Accuracy: {accuracy}\n\n")

if _name_ == "_main_":
    from data_preprocessing import preprocess_data
    from data_collection import load_data

    data = load_data('../data/loan.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    evaluate_model_one(X_test, y_test)
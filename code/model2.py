mport pickle
from sklearn.ensemble import RandomForestClassifier

def train_model_two(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    

    with open('../models/model_two.pkl', 'wb') as file:
        pickle.dump(model, file)

if _name_ == "_main_":
    from data_preprocessing import preprocess_data
    from data_collection import load_data

    data = load_data('../data/loan.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    train_model_two(X_train, y_train)
    print("Model Two Trained and Saved.")
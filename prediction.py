import joblib
def predict(data):
    model = joblib.load('musicmodel.joblib')
    
    return model.predict(data)
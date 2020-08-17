import pickle
import xgboost as xgb
import pandas as pd
file_name = "model/model.pkl"

# load
xgb_model_loaded = pickle.load(open(file_name, "rb"))

test = pd.read_csv('example.csv')
test.set_index('Unnamed: 0',inplace = True)

# test
print(xgb_model_loaded.predict(xgb.DMatrix(test)))

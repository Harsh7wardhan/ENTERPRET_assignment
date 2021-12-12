import  pickle

import pandas as pd
loaded_model_1 = pickle.load(open('model.pkl','rb'))


test_file=pd.read_csv("C:/Users/harsh/Downloads/test.csv")



import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('C:/Users/HP/Desktop/Diabetes Prediction/trained_model.sav','rb')) 

input_data=(1,89,66,23,94,28.1,0.167,21)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
standardized_input_data=input_data_reshaped
print(standardized_input_data)
prediction = loaded_model.predict(standardized_input_data)
print(prediction) 
if (prediction[0]==0):
    print('The person is not Diabetic')
else:
    print('The person is Diabetic')
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:24:10 2020

@author: Ahmed
"""



import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras import backend as K 

K.clear_session()

dataframe = pd.read_csv("C:\\Users\\Ahmed\\Desktop\\diabetes\\datasets_diabetes.csv")
dataframe.head()

df_label = dataframe['Outcome']
df_features = dataframe.drop('Outcome', 1)
df_features.replace('?', -99999, inplace=True)
df_features.head()

label = []
for lab in df_label:
    if lab == 1:
        label.append([1, 0])  # class 1
    elif lab == 0:
        label.append([0, 1])  # class 0
        data = np.array(df_features)
        
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
x_train.shape

model1 = Sequential()
model1.add(Dense(500, input_dim=8, activation='sigmoid'))
model1.add(Dense(100, activation='sigmoid'))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model1.fit(x_train,y_train, epochs=1000, batch_size=70, validation_data=(x_test, y_test))

scoers=model1.evaluate(data,label)


y_pred= model1.predict(x_test)
y_pred_prob = model1.predict_proba(x_test)


#b=(model.predict_proba(np.array([[1,85,66,29,0,26.6,0.351,31]])))


#s=round(b[0,0])


    
pickle.dump(model1, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict(np.array([[1,85,66,29,0,26.6,0.351,31]])))


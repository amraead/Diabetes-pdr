#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import pickle

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

#----------------------------------------------------

#load breast cancer data
heart_df=pd.read_csv("C:\\Users\\Ahmed\\Desktop\\prmodel\\framingham.csv")
heart_df.head()
heart_df = heart_df.drop(['education'], axis=1)

#X Data
X = heart_df.iloc[:, :-1].values
y = heart_df.iloc[:, -1].values
#X =( heart_df.male , heart_df.age , heart_df.education , heart_df.currentSmoker , heart_df.cigsPerDay , heart_df.BPMeds , heart_df.prevalentStroke , heart_df.prevalentHyp , heart_df.diabetes , heart_df.totChol , heart_df.sysBP , heart_df.diaBP , heart_df.BMI , heart_df.heartRate , heart_df.glucose) 
#print('X shape is ' , X.shape)
#print('X Features are \n' , BreastData.feature_names)

#y Data
#y = heart_df.TenYearCHD
#print('y shape is ' , y.shape)
#print('y Columns are \n' , BreastData.target_names)

#----------------------------------------------------
#Splitting data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying GaussianNB Model 

'''
#sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
'''

GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)



#Calculating Prediction
y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)


# metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
#l=y_pred.predict(f)
#prediction = GaussianNBModel.predict(0,70,1,0,0,0,1,1,0,107,143,93,25.8,68,62)
#print(prediction)
#output = prediction[0]
#predictions =GaussianNBModel.predict([[1,43,1,0,0,0,0,0,0,126,152,96.5,25.65,86,82]])
#print ( predictions )



pickle.dump(GaussianNBModel, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))




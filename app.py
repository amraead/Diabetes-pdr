import numpy as np
import time
import threading
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    predictionsProb =model.predict_proba(final_features)
    
    
    c=(predictionsProb)
    for b,value in enumerate(c,1):
      x=round(max(value*100))
     
    if prediction==1 :
     return render_template('index.html', prediction_text ='you are at risk of infection by heart disease over the next ten years',  predictions_prob = 'with ratio  ',p = x,per = '%')
    elif  prediction==0 :
     return render_template('index.html', prediction_text ='You are not threatened with heart disease during the next ten years',  predictions_prob = 'with ratio  ',p = x, per = '%')

   # output = round(prediction[0], 2)
   
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# Load the models
models = {}
model_names = ['model_1', 'model_2', 'model_3', 'model_4']

for name in model_names:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET",'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        TOP_TEMP1 = float(request.form['TOP_TEMP1'])
        TOP_TEMP2 = float(request.form['TOP_TEMP2'])
        TOP_TEMP3 = float(request.form['TOP_TEMP3'])
        TOP_TEMP4 = float(request.form['TOP_TEMP4'])
        TOP_TEMP = float(request.form['TOP_TEMP'])

        features = [TOP_TEMP1, TOP_TEMP2, TOP_TEMP3, TOP_TEMP4, TOP_TEMP]

        features = np.array(features).reshape(1, -1)
        pred_next_1hr = models['model_1'].predict(features)
        # Append prediction and predict next_2hr
        features_next_2hr = np.column_stack([features, pred_next_1hr])
        pred_next_2hr = models['model_2'].predict(features_next_2hr)
        # Append prediction and predict next_3hr
        features_next_3hr = np.column_stack([features_next_2hr, pred_next_2hr])
        pred_next_3hr = models['model_3'].predict(features_next_3hr)
        # Append prediction and predict next_4hr
        features_next_4hr = np.column_stack([features_next_3hr, pred_next_3hr])
        pred_next_4hr = models['model_4'].predict(features_next_4hr)

        return render_template('result.html', next_1hr=pred_next_1hr[0], next_2hr=pred_next_2hr[0], next_3hr=pred_next_3hr[0], next_4hr=pred_next_4hr[0])   

if __name__ == '__main__':
    app.run(debug=True)












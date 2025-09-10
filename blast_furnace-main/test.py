# load the models 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the models
models = {}
model_names = ['model_1', 'model_2', 'model_3', 'model_4']

for name in model_names:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

# Predict next_1hr
data = {
    "features": [
        [123, 139, 117, 132, 128],
        [117, 138, 126, 129, 128],
        [132, 155, 150, 139, 144],
        [130, 151, 149, 133, 141],
        [137, 145, 124, 140, 136]
    ]
}

for features in data['features']:
    features = np.array(features).reshape(1, -1)

    next_1hr = []
    next_2hr = []
    next_3hr = []
    next_4hr = []

    pred_next_1hr = models['model_1'].predict(features)
    print("Next 1hr: ", pred_next_1hr)
    next_1hr.append(pred_next_1hr[0])

    # Append prediction and predict next_2hr
    features_next_2hr = np.column_stack([features, pred_next_1hr])
    pred_next_2hr = models['model_2'].predict(features_next_2hr)
    print("Next 2hr: ", pred_next_2hr)
    next_2hr.append(pred_next_2hr[0])

    # Append prediction and predict next_3hr
    features_next_3hr = np.column_stack([features_next_2hr, pred_next_2hr])
    pred_next_3hr = models['model_3'].predict(features_next_3hr)
    print("Next 3hr: ", pred_next_3hr)
    next_3hr.append(pred_next_3hr[0])

    # Append prediction and predict next_4hr
    features_next_4hr = np.column_stack([features_next_3hr, pred_next_3hr])
    pred_next_4hr = models['model_4'].predict(features_next_4hr)
    print("Next 4hr: ", pred_next_4hr)
    next_4hr.append(pred_next_4hr[0])


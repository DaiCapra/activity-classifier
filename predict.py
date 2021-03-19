import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

from data import load
from properties import get_path_data, get_path_model, get_time_steps, get_features


def get_activity(activity, df, start, end):
    data_frame = df[df['activity'] == activity][['x', 'y', 'z']][start:end]
    return data_frame


def plot_activity(activity, df, start, end):
    data = df[df['activity'] == activity][['x', 'y', 'z']][start:end]
    axis = data.plot(subplots=True, figsize=(16, 12),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

file_path_model = get_path_model()
file_path_data = get_path_data()

data = load(file_path_data)

time_steps = get_time_steps()
features = get_features()

a = "Walking"
start = 1200
end = start + time_steps

activity = get_activity(a, data, start, end)
print(activity);

values = pd.np.asarray(activity, dtype=pd.np.float32).reshape(-1, time_steps, features)
model = load_model(file_path_model)

x = model.predict(values)
prediction = x.flatten()


objects = ('Jogging', 'Sitting', 'Standing', 'Walking')
y_pos = np.arange(len(objects))

plot_activity(a, data, start, end)
plt.show()

plt.bar(y_pos, prediction, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Probability')
plt.title('Prediction')
plt.show()


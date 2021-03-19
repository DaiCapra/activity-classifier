import os

from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from data import load, split_data_into_steps
from properties import get_time_steps, get_categories, get_hidden_size, get_features, get_step, get_random_seed, \
    get_path_data
import matplotlib.pyplot as plt

print("loading data...")
file_path_data = get_path_data()
data = load(file_path_data)

time_steps = get_time_steps()
features = get_features()
random_seed = get_random_seed()
step = get_step()
categories = get_categories()
hidden_size = get_hidden_size()

#data['activity'].value_counts().plot(kind='bar', title='Activity types')
#plt.show()

x_train, x_test, y_train, y_test, activities = split_data_into_steps(data, time_steps, features, step, random_seed)


model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(time_steps, features)))
model.add(LSTM(hidden_size))
model.add(Dropout(0.1))
model.add(Dense(categories, activation='sigmoid'))

# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

file_path_best = "models/epoch-loss-acc {epoch:02d}-{loss:.5f}-{acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(file_path_best, monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
callbacks_list = [checkpoint]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks_list)

# Plot training & validation accuracy values
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

# Plot training & validation loss values
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
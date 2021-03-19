from keras.models import load_model

from data import load, split_data_into_steps
from properties import get_path_model, get_path_data, get_time_steps, get_features, get_random_seed, get_step, \
    get_categories, get_hidden_size

file_path_model = get_path_model()
file_path_data = get_path_data()
time_steps = get_time_steps()
features = get_features()
random_seed = get_random_seed()
step = get_step()
categories = get_categories()
hidden_size = get_hidden_size()

data = load(file_path_data)
model = load_model(file_path_model)

x_train, x_test, y_train, y_test, activities = split_data_into_steps(data, time_steps, features, step, random_seed)

results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)
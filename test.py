from data import load, split_data_into_steps
from properties import get_path_data, get_time_steps, get_features, get_random_seed, get_step

file_path_data = get_path_data()
data = load(file_path_data)

time_steps = get_time_steps()
features = get_features()
random_seed = get_random_seed()
step = get_step()

x_train, x_test, y_train, y_test, activities = split_data_into_steps(data, time_steps, features, step, random_seed)

print(activities)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

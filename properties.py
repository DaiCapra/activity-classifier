time_steps = 100
features = 3
random_seed = 42
validation_split = 0.25
step = 5
categories = 4
hidden_size = 40

file_path_model = "models/epoch-loss-acc 20-0.00399-0.99858.hdf5"
file_path_data = "train/WISDM_ar_v1.1_processed.txt"


def get_path_data():
    return file_path_data


def get_path_model():
    return file_path_model


def get_validation_split():
    return validation_split


def get_categories():
    return categories


def get_hidden_size():
    return hidden_size


def get_time_steps():
    return time_steps


def get_features():
    return features


def get_step():
    return step


def get_random_seed():
    return random_seed

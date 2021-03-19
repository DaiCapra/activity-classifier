import pandas as pd
from scipy.stats import stats
from sklearn.model_selection import train_test_split

from properties import get_validation_split


def load(file_path):
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    data = pd.read_csv(file_path, names=columns)
    data = data.drop(data.query('activity=="Downstairs"').sample(frac=1).index)
    data = data.drop(data.query('activity=="Upstairs"').sample(frac=1).index)
    data = data.drop(data.query('activity=="LyingDown"').sample(frac=1).index)
    data = data.drop(data.query('activity=="Stairs"').sample(frac=1).index)
    return data


def split_data_into_steps(data, N_TIME_STEPS, N_FEATURES, step, RANDOM_SEED):
    segments = []
    labels = []
    activities = set()
    for i in range(0, len(data) - N_TIME_STEPS, step):
        xs = data['x'].values[i: i + N_TIME_STEPS]
        ys = data['y'].values[i: i + N_TIME_STEPS]
        zs = data['z'].values[i: i + N_TIME_STEPS]
        label = stats.mode(data['activity'][i: i + N_TIME_STEPS])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
        activities.add(label)

    reshaped_segments = pd.np.asarray(segments, dtype=pd.np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    labels = pd.np.asarray(pd.get_dummies(labels), dtype=pd.np.float32)

    validation_split = get_validation_split()
    x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=validation_split,
                                                        random_state=RANDOM_SEED)
    return x_train, x_test, y_train, y_test, activities

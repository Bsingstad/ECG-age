import os
from typing import Generator, Optional

import numpy as np
import tqdm
from scipy import signal
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x["val"], dtype=np.float64)
    new_file = filename.replace(".mat", ".hea")
    input_header_file = os.path.join(new_file)
    with open(input_header_file, "r") as f:
        header_data = f.readlines()
    return data, header_data


def import_key_data(path):
    gender = []
    age = []
    labels = []
    ecg_filenames = []
    ecg_len = []
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in tqdm.tqdm(files):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                if (
                    int(header_data[0].split(" ")[3])
                    // int(header_data[0].split(" ")[2])
                    == 10
                ):
                    labels.append(header_data[15][5:-1])
                    ecg_filenames.append(filepath)
                    gender.append(header_data[14][6:-1])
                    age.append(header_data[13][6:-1])
                    ecg_len.append(
                        int(header_data[0].split(" ")[3])
                        // int(header_data[0].split(" ")[2])
                    )
    return gender, age, labels, ecg_len, ecg_filenames


def only_ten_sec(ecg_len, age, gender, filename, labels):
    idx = np.where(ecg_len == 10)[0]
    gender = gender[idx]
    age = age[idx]
    filename = filename[idx]
    labels = labels[idx]
    return age, gender, filename, labels


def clean_up_age_data(age):
    age[np.where(age == "60.")] = 60
    age = age.astype(int)
    return age


def clean_up_gender_data(gender):
    gender[np.where(gender == "Male")] = 0
    gender[np.where(gender == "male")] = 0
    gender[np.where(gender == "M")] = 0
    gender[np.where(gender == "Female")] = 1
    gender[np.where(gender == "female")] = 1
    gender[np.where(gender == "F")] = 1
    gender[np.where(gender == "NaN")] = 2
    gender = gender.astype(np.int)
    return gender


def split_data(age, gender, n_splits=5):
    folds = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            gender, age
        )
    )
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds


def male_or_female(gender, age, ecg_filenames, labels, g="female"):
    if g == "female":
        gender_idx = 0
    elif g == "male":
        gender_idx = 1
    else:
        print("unsupported category")
    age = np.delete(age, np.where(gender == gender_idx))
    ecg_filenames = np.delete(ecg_filenames, np.where(gender == gender_idx))
    labels = np.delete(labels, np.where(gender == gender_idx))
    gender = np.delete(gender, np.where(gender == gender_idx))
    return gender, age, ecg_filenames, labels


def remove_nan_and_unknown_values(
    ecg_filenames, gender: np.ndarray, age: np.ndarray, labels
):

    ecg_filenames = np.delete(ecg_filenames, np.where(age == "NaN"))
    gender = np.delete(gender, np.where(age == "NaN"))
    labels = np.delete(labels, np.where(age == "NaN"))
    age = np.delete(age, np.where(age == "NaN"))

    age = np.delete(age, np.where(gender == "NaN"))
    age = np.delete(age, np.where(gender == "Unknown"))
    ecg_filenames = np.delete(ecg_filenames, np.where(gender == "NaN"))
    ecg_filenames = np.delete(ecg_filenames, np.where(gender == "Unknown"))
    labels = np.delete(labels, np.where(gender == "NaN"))
    labels = np.delete(labels, np.where(gender == "Unknown"))
    gender = np.delete(gender, np.where(gender == "NaN"))
    gender = np.delete(gender, np.where(gender == "Unknown"))
    return ecg_filenames, gender, age, labels


def shuffle_batch_generator_age(
    batch_size: int,
    gen_x: Generator,
    gen_y: Generator,
    num_leads: int,
    samp_freq: int,
    time: int,
):
    batch_features = np.zeros((batch_size, samp_freq * time, num_leads))
    batch_labels_1 = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels_1[i] = next(gen_y)

        yield batch_features, batch_labels_1


def generate_y_age(y_train: np.ndarray):
    while True:
        for i in y_train:
            yield i


def generate_X_age(X_train: np.ndarray, samp_freq: int, num_leads: int):
    while True:
        for h in X_train:
            data, header_data = load_challenge_data(h)
            if int(header_data[0].split(" ")[2]) != samp_freq:
                data_new = np.ones(
                    [
                        num_leads,
                        int(
                            (
                                int(header_data[0].split(" ")[3])
                                / int(header_data[0].split(" ")[2])
                            )
                            * samp_freq
                        ),
                    ]
                )
                for i, j in enumerate(data):
                    data_new[i] = signal.resample(
                        j,
                        int(
                            (
                                int(header_data[0].split(" ")[3])
                                / int(header_data[0].split(" ")[2])
                            )
                            * samp_freq
                        ),
                    )
                data = data_new
                data = pad_sequences(
                    data, maxlen=samp_freq * 10, truncating="post", padding="post"
                )
            data = np.moveaxis(data, 0, -1)
            yield data


def load_header(header_file) -> str:
    with open(header_file, "r") as f:
        header = f.read()
    return header


def get_labels(header: str) -> list:
    labels = list()
    for l in header.split("\n"):
        if l.startswith("#Dx"):
            try:
                entries = l.split(": ")[1].split(",")
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels


def is_integer(x: Optional[str]) -> Optional[int]:
    if is_number(x):
        return float(x).is_integer()
    else:
        return False


def is_number(x: Optional[str]) -> Optional[float]:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

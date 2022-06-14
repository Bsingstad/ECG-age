from scipy.io import loadmat
import numpy as np
import os
import tqdm
from scipy import signal
from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def import_key_data(path):
    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    ecg_len = []
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in tqdm.tqdm(files):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                if int(header_data[0].split(" ")[3])//int(header_data[0].split(" ")[2]) == 10:
                    labels.append(header_data[15][5:-1])
                    ecg_filenames.append(filepath)
                    gender.append(header_data[14][6:-1])
                    age.append(header_data[13][6:-1])
                    ecg_len.append(int(header_data[0].split(" ")[3])//int(header_data[0].split(" ")[2]))
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
    gender[np.where(gender == "NaN")] = 0 # only one nan
    np.unique(gender)
    gender = gender.astype(np.int)
    return gender

def split_data(age, gender,n_splits=5):
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(gender,age))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds


def shuffle_batch_generator_age(batch_size, gen_x,gen_y,num_leads, samp_freq,time): 
    batch_features = np.zeros((batch_size,samp_freq*time, num_leads))
    batch_labels_1 = np.zeros((batch_size,1)) 

    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels_1[i] = next(gen_y)

        yield batch_features, batch_labels_1

def generate_y_age(y_train):
    while True:
        for i in y_train:
            yield i


def generate_X_age(X_train, samp_freq, num_leads):
    while True:
        for h in X_train:
            data, header_data = load_challenge_data(h)
            if int(header_data[0].split(" ")[2]) != samp_freq:
                data_new = np.ones([num_leads,int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*samp_freq)])
                for i,j in enumerate(data):
                    data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*samp_freq))
                data = data_new
                data = pad_sequences(data, maxlen=samp_freq*10, truncating='post',padding="post")
            data = np.moveaxis(data, 0, -1)
            yield data
            


def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

def get_labels(header):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels
    
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False
        
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False
           
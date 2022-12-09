import serial
import time
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import sys
import numpy as np
import neurokit2 as nk
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../")
from src.models.models import *

model = build_model((1000,1),1)
model.load_weights("./model_weights_leadI.h5")

NEW_SAMP_FREQ = 100
ECG_LEN = 10
cnt = 0
elapsed = 0
start = time.time()
ecg = []

ser = serial.Serial('COM3', 9800, timeout=1)

while  elapsed < ECG_LEN:
    elapsed = time.time() - start
    
    line = ser.readline()
    print(line)
    string = line.decode()
    stripped_string = string.strip()
    try:
        num_int = int(stripped_string)
    except:
        try: 
            num_int = num_int
        except:
            num_int = 0
    ecg.append(num_int)
    print(num_int)
ser.close()
ecg = np.asarray(ecg)
scaler = MinMaxScaler()
ecg = scaler.fit_transform(np.expand_dims(ecg,1))
ecg_resamp = signal.resample(ecg[:,0], NEW_SAMP_FREQ*ECG_LEN)
ecg_clean = nk.ecg.ecg_clean(ecg_resamp, sampling_rate=100)
ecg_clean = np.expand_dims(np.expand_dims(ecg_clean,0),-1)
plt.plot(ecg_clean[0,:,0])
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('ECG reading 10 seconds')
plt.show()

print("Predicted age: {} years old".format(int(model.predict(ecg_clean)[0][0])))
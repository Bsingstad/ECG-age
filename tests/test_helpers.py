import pytest
import numpy as np
from src.helpers.helpers import clean_up_age_data, only_ten_sec, clean_up_gender_data

def test_clean_up_age_data():
    a = np.array([40,50,60,"60."])
    b = np.array([40,50,60,60])
    c = clean_up_age_data(a)

    assert np.array_equal(b,c)
    
def test_only_ten_sec():
    age = np.array([40,50,60,70])
    ecg_len = np.array([10,9,8,10])
    age = np.array([40,50,60,70])
    gender = np.array([1,0,1,0])
    filename = np.array(["test1","test2","test3","test4"])
    labels = np.array(["label1", "label2", "label3", "label4"])
    result_age, result_gender, result_filename, result_labels = only_ten_sec(ecg_len, age, gender, filename, labels)
    assert np.array_equal(age[[1,2]],result_age)
    assert np.array_equal(gender[[1,2]],result_gender)
    assert np.array_equal(filename[[1,2]],result_filename)
    assert np.array_equal(labels[[1,2]],result_labels)

def test_clean_up_gender_data():
    genders = np.array(["M","F","male","Male","Female","female","NaN",1])
    expected_result = np.array([0,1,0,0,1,1,2,1])

    result_gender = clean_up_gender_data(genders)
    assert np.array_equal(result_gender,expected_result)
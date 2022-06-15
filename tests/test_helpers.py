import pytest
import numpy as np
from src.helpers.helpers import clean_up_age_data

def test_clean_up_age_data():
    a = np.array([40,50,60,"60."])
    b = np.array([40,50,60,60])
    c = clean_up_age_data(a)

    assert np.array_equal(b,c)

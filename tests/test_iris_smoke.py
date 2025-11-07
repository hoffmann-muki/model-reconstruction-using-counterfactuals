import os
from utils_v8 import generate_query_data

def test_iris_smoke(tmp_path):
    out_dir = os.path.join(str(tmp_path), 'iris_smoke')
    res = generate_query_data(out_dir,
                              'iris',
                              True,
                              2,
                              'naivedat',
                              'onesided',
                              'knn',
                              2,
                              'TF',
                              'random',
                              1,
                              1,
                              [16],
                              2,
                              0.01,
                              [[8]],
                              3,
                              0.01,
                              [-1],
                              [-1],
                              16)
    assert os.path.exists(res)
    assert os.path.isfile(os.path.join(res, 'info.csv'))

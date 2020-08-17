import pandas as pd
import numpy as np

def load_csv():
    file = []
    with open('data.csv', 'r') as f:
        f.readline()
        while True:
            try:
                file.append([int(i) for i in f.readline().replace('\n', '').split(',')])
            except:
                break
    data_all = [file[i][:324] for i in range(len(file))]
    target = [file[i][324] for i in range(len(file))]
    data = np.array([[[[arr[i + j * 9 + k * 54] for i in range(9)] for j in range(6)] for k in range(6)] for arr in data_all])
    return (data, target)

load_csv()
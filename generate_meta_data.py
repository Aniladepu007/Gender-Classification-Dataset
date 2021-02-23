import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
import os, random, cv2, copy
from os import path

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

path_m = './SEM_10/TA_work/DL/Gender Classification Dataset/Male400'
path_f = './SEM_10/TA_work/DL/Gender Classification Dataset/Female400'
files_m = return_list(path_m, '.jpg')
files_f = return_list(path_f, '.jpg')

gender = ['male']*len(files_m) + ['female']*len(files_f)
target = [0]*len(files_m) + [1]*len(files_f)

for i in range(len(files_m)):
    files_m[i] = files_m[i][:-4]
for i in range(len(files_f)):
    files_f[i] = files_f[i][:-4]
image_name = files_m + files_f
data = [[image_name[i], gender[i], target[i]] for i in range(len(image_name))]
meta_data = pd.DataFrame(data=data, columns=['image_name', 'gender', 'target'])
meta_data.to_csv('meta_data.csv', index=False)

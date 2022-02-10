import os
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# %%
"""
load a model from the model path,whict me had train and save with model.save(modeldir).
"""


def load(modelPath):
    model = None
    if os.path.exists(modelPath):
        model = tf.keras.models.load_model(modelPath)
    return model


# %%

"""
get the last num (lastNum) data from the dataset,and parse to array.
return two array ,the first is valuse  array ,the second is tabel array .
"""


def getPreDataArray(data=None, lastNum=3):
    x_vs = []
    x_ws = []
    if data != None:
        for i in range(len(data) - lastNum, len(data)):
            x_vs.append(data[i][0])
            x_ws.append(data[i][1])
        x_vs = np.array(x_vs)
        x_vs = np.reshape(x_vs, (1, lastNum, 1))
        x_ws = np.array(x_ws)
        x_ws = np.reshape(x_ws, (1, lastNum, 1))
    return x_vs, x_ws


# %%
"""
init pre import models,it use long time to load,so we have do it befor.
return two models,firsh is data models second is label models.
"""


def init():
    global model_labels
    global model_nums
    global model_nums_offset
    global models_data
    global models_label

    print('predict init ...')
    folder_list = searchSubfolder('.\predict')
    for folder in folder_list:
        print(folder)
        fs = folder.split('_')
        if len(fs) == 3:
            if fs[0] == 'predict':
                if not fs[1] in model_labels:
                    model_labels.append(fs[1])
                if not int(fs[2]) in model_nums:
                    model_nums.append(int(fs[2]))

    model_labels.sort()
    model_nums.sort()

    if len(model_nums) > 0:
        model_nums_offset = model_nums[0] - 1

    print('model_labels {}'.format(model_labels))
    print('model_nums {}'.format(model_nums))
    print('model_nums_offset {}'.format(model_nums_offset))

    totalTime = int(time.time() * 1000)
    for num in model_nums:
        for label in model_labels:
            modeldir = model_dir.format(label=label, num=num)
            t = int(time.time() * 1000)
            print('start to load model {} ...'.format(modeldir))
            model = load(modeldir)
            if model != None:
                if label == 'label':
                    models_label[num] = model
                else:
                    models_data[num] = model
                print('success to load model {}...use time {} ms'.format(modeldir, int(time.time() * 1000) - t))
            else:
                print('fail to load model {}...use time {} ms'.format(modeldir, int(time.time() * 1000) - t))
    print('had use time {} ms to pre import models'.format(int(time.time() * 1000) - totalTime))


# %%
"""
predict data and return a val.
"""


# %%
def predict(data):
    totalTime = int(time.time() * 1000)
    bval = 0
    pval = 0
    val = 0
    if data != None:
        dataLen = len(data) if len(data) < len(model_nums) + model_nums_offset else len(model_nums) + model_nums_offset
        # start num from 3 to 15
        for num in range(model_nums_offset, dataLen):
            num = num + 1
            x_vs, x_ws = getPreDataArray(data, lastNum=num)
            for label in model_labels:
                model = models_data.get(num) if label != 'label' else models_label.get(num)
                if model != None:
                    predictTest = x_vs if label != 'label' else x_ws
                    predictTest = predictTest[-1]
                    sc = MinMaxScaler(feature_range=(0, 1))
                    predictVal = sc.fit_transform(predictTest)
                    predictVal = np.reshape(predictVal, (1, predictVal.shape[0], 1))
                    predictVal = model.predict(predictVal)
                    predictVal = sc.inverse_transform(predictVal)
                    predictVal = predictVal[0][0]
                    realVal = predictTest[-1][0]
                    print('predict data:{:.6f}'.format(predictVal))
                    print('real data:{:.6f}'.format(realVal))
                    if label != 'label':
                        val = 'P' if predictVal < realVal else 'B'
                    else:
                        val = 'P' if predictVal > 1.5 else 'B'
                    if val == 'P':
                        pval += 1
                    else:
                        bval += 1
                    print('predict result {} from {} {}'.format(val, label, num))
                else:
                    print('model is None {} {}'.format(label, num))
    print('had use time {} ms to predict data'.format(int(time.time() * 1000) - totalTime))

    if pval > bval:
        val = 2
    elif bval > pval:
        val = 1
    else:
        val = 0
    print('the result predict val is {}'.format(val))
    return val


# %%
"""
search subfolder
"""


def searchSubfolder(path):
    folder_list = []
    for root, dirs, files in os.walk(path):
        folder_list = dirs
        break
    return folder_list


# %%
model_labels = ['label', 'data']
model_nums = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
model_nums_offset = 2
# auto search
model_labels = []
model_nums = []
model_nums_offset = 0
model_dir = '.\predict\predict_{label}_{num}'
models_data = {}
models_label = {}

# %%


if __name__ == '__main__':
    # %%
    model_nums = [3, 4]
    init()
    # %%
    val = predict([[0.17, 1], [0.24, 1], [0.42, 1], [0.32, 2], [-0.08, 2], [0.58, 1], [1.75, 1], [1.75, 1.5]])
    print(val)

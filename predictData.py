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
predict batch data and return a batch val.
"""


def predict_batch(data):
    totalTime = int(time.time() * 1000)
    v = {"label": [], "data": []}
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

                    if label != 'label':
                        val = 'P' if predictVal < realVal else 'B'
                    else:
                        val = 'P' if predictVal > 1.5 else 'B'
                    v[label].append([num, 1 if val == 'B' else 2])
                    if val == 'P':
                        pval += 1
                    else:
                        bval += 1
                    log_print('predict data:{:.6f}'.format(predictVal))
                    log_print('real data:{:.6f}'.format(realVal))
                    log_print('predict result {} from {} {}'.format(val, label, num))
                else:
                    log_print('model is None {} {}'.format(label, num))
    log_print('had use time {} ms to predict data'.format(int(time.time() * 1000) - totalTime))

    if pval > bval:
        val = 2
    elif bval > pval:
        val = 1
    else:
        val = 0
    log_print('the result predict val is {} val = {} pval = {}'.format(val, bval, pval))
    return v


# %%

"""
predict data and return a val.
"""


def predict(label, data):
    totalTime = int(time.time() * 1000)
    v = {"label": [], "data": []}
    bval = 0
    pval = 0
    val = 0
    if data != None:
        dataLen = len(data) if len(data) < len(model_nums) + model_nums_offset else len(model_nums) + model_nums_offset
        # start num from 3 to 15
        num = dataLen
        x_vs, x_ws = getPreDataArray(data, lastNum=num)
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

            if label != 'label':
                val = 'P' if predictVal < realVal else 'B'
            else:
                val = 'P' if predictVal > 1.5 else 'B'
            v[label].append([num, 1 if val == 'B' else 2])
            if val == 'P':
                pval += 1
            else:
                bval += 1
            log_print('predict data:{:.6f}'.format(predictVal))
            log_print('real data:{:.6f}'.format(realVal))
            log_print('predict result {} from {} {}'.format(val, label, num))
        else:
            log_print('model is None {} {}'.format(label, num))

    log_print('had use time {} ms to predict data'.format(int(time.time() * 1000) - totalTime))

    if pval > bval:
        val = 2
    elif bval > pval:
        val = 1
    else:
        val = 0
    log_print('the result predict val is {} val = {} pval = {}'.format(val, bval, pval))
    return v


# %%
def log_print(txt):
    if log:
        print(txt)


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
log = False

# %%


if __name__ == '__main__':
    log = True
    # %%
    init()
    # %%
    data = [[-3.150000095367, 2], [-2.75, 2], [-2.25, 2], [-0.5, 1], [-0.75, 1], [-1.350000023842, 2],
            [-1.016666650772, 2], [0.383333325386, 1], [-0.1166666895151, 2], [0.04999997466803, 1],
            [-0.3500000238419, 1], [-0.6000000238419, 2], [-0.1000000461936, 2], [0.7333332896233, 1],
            [-0.6666666865349, 2]]
    #%%
    v = predict_batch(data)
    print(v)
    #%%
    v = predict('label', data)
    print(v)
    #%%
    v = predict('data', data)
    print(v)

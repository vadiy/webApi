# %%
import os,re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# %% md
# 转换数据集为值有 val,win 属性的表

def getDataSet(filepath):
    data = pd.read_csv(filepath, sep='|', header=None, encoding="GBK")
    # print(data)
    # print(len(data.columns))
    if len(data.columns) >= 10:
        data = data.iloc[:, :10 - len(data.columns)]  # 減去第一行和列的头尾
    if data[0][0] != data[0][1]:
        data = data.iloc[1:, :]
    data.columns = ['vid', 'gmcode', 'bval', 'pval', 'num', 'pair', 'redPair', 'bluePair', 'winType', 'winNum']
    data.bval = data.bval.astype('float')
    data.pval = data.pval.astype('float')
    data.num = data.num.astype('float')
    data.winNum = data.winNum.astype('float')
    data.winType = data.winType.astype('int')  # 训练和测试集不能为str类型??
    data.reset_index(inplace=True)  # 重新设置索引从0开始

    # 创建新的表格文件
    newData = pd.DataFrame(columns=[['val', 'win']])  # 必须这样设置

    val = 0.0
    # 值能够反映bp的牌值和发的牌数和赢家
    # 差 = bval - pval = + - 反映 庄闲点数
    # 差/num = 反映 每张牌权重
    # 差/winnum = 反映 点权重

    for i in range(len(data)):  # 转换为股票K线趋势形式
        bval = data['bval'][i]
        pval = data['pval'][i]
        num = data['num'][i]
        winNum = data['winNum'][i]  # 防止 winNum =0 做除数的情况
        winType = data['winType'][i]
        if winType == 3:  # 和用1.5表示 1庄 2闲
            winType = 1.5
        val = val + (bval - pval) / num
        newData.loc[i] = [val, winType]

    return newData


# %%
def getDataArray(dataset, random=False, num=7, label='label'):
    # 要先吧表格数据转换为数组数据
    datas = dataset.val.values
    labels = dataset.win.values
    x = []
    y = []
    # print(label)
    for i in range(num, len(datas)):  # -1是留给标签位的
        # 采用前 num 局数做特征
        if label != 'label':
            x.append(datas[i - num:i, 0])
            y.append(datas[i, 0])  # 用当前val 坐标作为标签看看
        else:
            x.append(labels[i - num:i, 0])
            y.append(labels[i, 0])  # 使用庄闲和做标签,看loss 值都是在0.43 accuracy 0.46

    if random:
        # 打乱训练集顺序
        np.random.seed(7)
        np.random.shuffle(x)
        np.random.seed(7)
        np.random.shuffle(y)
    x = np.array(x)
    x = np.reshape(x, (len(x), num, 1))
    y = np.array(y)
    return x, y


# %%
# 分割数据集 num 连续天数
def sliceDataSet(da):
    # 归一化 就是最高最低价都在0-1之间
    data = da
    sc = MinMaxScaler(feature_range=(0, 1))
    data.val = sc.fit_transform(data.val.values)

    # 分割与测试数据集
    trainrows = int(len(data) * 0.8)
    train_set = data.iloc[0:trainrows, :]
    test_set = data.iloc[trainrows:, :]

    return sc, train_set, test_set


# %%
def create_model(checkpoint_save_path):
    # 用squential搭建神经网络
    model = Sequential()
    model.add(SimpleRNN(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # cmopile 配置训练方法使用adam优化器
    model.compile(optimizer=Adam(0.001),
                  # 损失函数用均方误差,不可有多余参数,如果有必要设置完,
                  # 应该只观测loss数值,不观测准确率,所以删掉了metrics选项
                  # 一会在每个epoch 迭代显示时只显示loss值
                  loss='mean_squared_error'
                  )

    # 设置断点续训
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    return model


# %%
def trainBAC(model, x_train, y_train, x_test, y_test, checkpoint_save_path):
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            monitor='val_loss')
    # fit 执行训练过程
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[cp_callback]
                        )


    # model.summary()

    return model, history

    # %%
def model_load(modelPath):
    model = None
    if os.path.exists(modelPath):
        model = tf.keras.models.load_model(modelPath)
    return model
#%%
def train():
    # 7 17 37 77 157 317
    nums = [3, 4, 5, 6,7, 8, 9, 10, 11, 12, 13, 14, 15]
    labels = ['data', 'label']
    starDate = 20220111
    onlyOnce = True
    subfolder = searchSubfolder('./AGBigData')
    for folder in subfolder:
        file_list = searchFile('./AGBigData/' + folder + '/', file_type='R_.(.*).txt$')
        fullDataSet = None
        fn = 0
        for file in file_list:
            if starDate <= int(file.split('.')[0].split('_')[2]):
                filepath = './AGBigData/' + folder + '/' + file
                print(filepath)
                dataset = getDataSet(filepath)
                if fullDataSet is not None:
                    fullDataSet = pd.concat([dataset, fullDataSet], ignore_index=True)
                else:
                    fullDataSet = dataset
                fn += 1
                if fn >= 3:
                    # break
                    pass

        for num in nums:
            for label in labels:
                checkpoint_save_path = './checkpoint/predict_' + label + '_' + str(num) + '.ckpt'
                modeldir = 'predict/predict_' + label + '_' + str(num)
                model = model_load(modeldir)
                if model == None:
                    model = create_model(checkpoint_save_path)

                if fullDataSet is not None:
                    print(len(fullDataSet))
                    sc, train_set, test_set = sliceDataSet(fullDataSet)
                    x_train, y_train = getDataArray(train_set, random=True, num=num, label=label)
                    x_test, y_test = getDataArray(test_set, random=False, num=num, label=label)

                    model, history = trainBAC(model, x_train, y_train, x_test, y_test, checkpoint_save_path)

                    model.save(modeldir)
                    # loss可视化
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']

                    plt.plot(loss, label='Trianing Loss')
                    plt.plot(val_loss, label='Validation Loss')
                    plt.title('Trianing and Validation Loss ' + label + '_' + str(num) + ' folder:' + folder)
                    plt.legend()
                    plt.show()
        if onlyOnce == True:
            break





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

def searchFile(file_dir,file_type='*.*'):
    file_list = []
    for root,dirs,files in os.walk(file_dir):
        for f in files:
            if re.search(file_type,f):
                file_list.append(f)

    return file_list
if __name__ == '__main__':
    # %%
    train()

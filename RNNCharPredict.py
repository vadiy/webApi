import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}  # 单词映射到数值id的词典
id_to_onehot = {0: [1., 0., 0., 0., 0.], 1: [0., 1., 0., 0., 0.], 2: [0., 0., 1., 0., 0.], 3: [0., 0., 0., 1., 0.],
                4: [0., 0., 0., 0., 1.]}  # id编码为one-hot

x_train = [id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']],
           id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']]]
# 训练集标签中存入的是其预测对应的字符
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

# 按照相同的方法打乱,使feature和label仍然是一对一的
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)

# 使x_train 符合SimpleRNN输入要求:[送入样本数,循环核时间展开步数,每个时间步输入特征个数].
# 此处整个数据集送入,送入样本数为len(x_train);输入1个字母预测结果,循环核时间展开步数为1;表示为独热码有5个输入特征,每个时间步输入特征个数为5
x_train = np.reshape(x_train, (len(x_train), 1, 5))  # 使用连续字符测试时,这里把1修改为连续字符个数
y_train = np.array(y_train)

model = tf.keras.Sequential([
    SimpleRNN(3),  # 记忆体个数,记忆体越多,记性越好,但是计算量增加
    Dense(5, activation='softmax')  # 输出层yt的计算,一层全连接输出5个数据,所以设置为5
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),  # 设置学习率为0.01,Adam优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )

checkpoint_save_path = './checkpoint/rnn_onehot_1pre1.ckpt'  # 1个输入预测结果输出1个
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------------load the model------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss'  # 由于fit 没有给出测试集,不计算测试集准确率,根据loss,保存最优模型
)

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

model.summary()

#%%
####################predict#########################
# 执行前向传播
preNum = int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet1 = input('input test alphabet:')
    alphabet = [id_to_onehot[w_to_id[alphabet1]]]
    # 使alphabet符合SimpleRNN输入要求:[送入样本数,循环核时间展开步数,每个时间步输入特征个数].
    # 此处验证效果送入了1个样本,送入样本数为1,输入1个字母出结果,所以循环核时间展开步数为1.
    # 表示为独热码有5个输入特征,每个时间步输入特征个数为5
    alphabet = np.reshape(alphabet, (1, 1, 5))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])

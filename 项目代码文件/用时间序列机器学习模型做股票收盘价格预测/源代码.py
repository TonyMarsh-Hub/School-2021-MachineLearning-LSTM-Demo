#!/usr/bin/env python
# coding: utf-8
# 1.展示源数据
# 导包用于数据分析和格式化操作
import matplotlib.pyplot as plt  # 导入matplotlib库用于数据的可视化展示
import numpy as np  # 导入库numpy用于数据格式化操作
import pandas as pd  # 导入数据分析库pandas

# 1.1读取源数据
df = pd.read_csv('data0.csv', names=['Date', 'Close'])
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# 1.2可视化数据集并显示
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='收盘历史价格')
plt.show()

# 2.训练模型前的准备：数据预处理
# 2.1格式转换为pandas的DataFrame
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date','Close'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
# 2.2为其设置索引
new_data.index = new_data.Date
# 2.3删除Date数据(只使用Close数据)
new_data.drop('Date', axis=1, inplace=True)
# 2.4创建训练和验证集(数据集的划分)
dataset = new_data.values
train = dataset[0:987, :]  # 将最开始4年(987个)的数据作为训练集
valid = dataset[987:, :]  # 之后的所有数据设置为验证集
# 2.5使用MinMaxScaler将数据的范围压缩至0到1之间，这么做的目的是为了防止数值爆炸
# (LSTM模型的计算中，时间步长越大，结果所迭代的次数就越多；如1.1的100次方为13 780.61233982，所以使用scaler()函数是必要的)
from sklearn.preprocessing import MinMaxScaler  # 从sklearn.preprocessing库中导入MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)  # 传入dataset，将其用scaler()函数正则化后命名为scaled_data
#2.6 用数组表现数据的时间序列特性，步长设置为60
x_train, y_train = [], []  # 创建x_train, y_train
for i in range(60, len(train)):  # 循环结构，i的值分别为60,61，……直至train的长度(1500)
    x_train.append(scaled_data[i - 60:i, 0])  # 将scaled_data中的数据传递到x_train中(60个为一组)
    y_train.append(scaled_data[i, 0])  # 将scaled_data中的数据传递到y_train中(从第60开始，每1个数据为一组)
x_train, y_train = np.array(x_train), np.array(y_train)  # 使用numpy库中的array()函数将列表x_train和y_train格式化为数组(可以理解为矩阵)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 使用numpy库中的reshape()函数改变数组x_train的形状
# 上述的步骤将训练集的格式从列表(list)变成时间步长为60的时间序列(表现为数组)

# 3.模型的参数设置
# 3.1导入神经网络需要的包
from keras.models import Sequential  # 从keras.models库中导入时间序列模型Sequential
from keras.layers import Dense, LSTM  # 从keras.layers库中导入Dense, Dropout, LSTM用于构建神经网络

# 3.2创建LSTM神经网络
model = Sequential()
# 第一层网络设置
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# unit 决定了一层里面 LSTM 单元的数量。这些单元是并列的，一个时间步长里，输入这个层的信号，会被所有 unit 同时并行处理，形成若干个 unit 个输出。这个设置50个单元
# return_sequence参数表示是否返回LSTM的中间状态，这里设置为TRUE，返回的状态是最后计算后的状态
# input_shape参数包含两个元素的，第一个代表每个输入的样本序列长度，这里是x_train.shape[1]，表示x_train数组中每一个元素的长度即时间步长，这里先前设置为了60
# 第二个元素代表每个序列里面的1个元素具有多少个输入数据(这里是1表示只有1个数据：时间)

# 第二层神经网络，设置50个LSTM单元
model.add(LSTM(units=50))
# 第三层为全连接层
model.add(Dense(1))

# 4.模型训练
model.compile(loss='mean_squared_error', optimizer='adam')  # 设置损失函数compile()参数
# loss参数指标使用MSE(均方根误差) ，optimizer参数设置优化器为AdamOptimizer(自适应矩估计，梯度下降的一种变形)
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)  # 传入数据开始训练模型

# 5.预测
#用过去的每60个数据预测接下来的数据(时间步长为60)
#5.1获取测试集数据，该步骤思路和代码41到45行一样，这里不再赘述
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#5.2获取模型给出的预测值
closing_price = model.predict(X_test)
# 用scaler.inverse()函数将数据重新放大(因为之前使用了scaler()函数对数据进行了压缩——第38行)
closing_price = scaler.inverse_transform(closing_price)

# 6.预测结果可视化展示
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.figure(figsize=(16, 8))
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.show()

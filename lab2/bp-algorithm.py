from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

iris=datasets.load_iris()
df=DataFrame(iris.data,columns=iris.feature_names)
df['target']=list(iris.target)
X=df.iloc[:,0:4]
Y=df.iloc[:,4]
print(df)
#对df进行划分，一共五项数据，前四项是特征为x，第五项是tag为y

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=22111606)#以22111606为随机种子对数据集进行要求比例的可复现随机划分
#正规化
sc=StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)

#可视化
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(standard_train[:, 0],standard_train[:, 1], c=Y_train, edgecolors='k', marker='o', s=100)
# plt.title('SEPAL')
# plt.xlabel('sepal_l')
# plt.ylabel('sepal_w')
# plt.grid()

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 2)
# plt.scatter(standard_train[:, 2],standard_train[:, 3], c=Y_train, edgecolors='k', marker='o', s=100)
# plt.title('PETAL')
# plt.xlabel('petal_l')
# plt.ylabel('petal_w')
# plt.grid()

#构建mlp模型
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=2000, activation='tanh',solver='adam', alpha=0.005, batch_size='auto', learning_rate='constant', learning_rate_init=0.01,shuffle=True, random_state=22111606, tol=0.00001, verbose=False)

mlp.fit(standard_train,Y_train)#用正规化后的数据与tag进行训练

result = mlp.predict(standard_test)#用训练得到的模型和验证集进行计算


#查看模型结果
print("测试集合的y值   ：",list(Y_test))
print("神经网络预测的y值：",list(result))
print("预测的准确率为：",mlp.score(standard_test,Y_test))
print("层数为：",mlp.n_layers_)
print("迭代次数为：",mlp.n_iter_)
print("损失为：",mlp.loss_)
print("激活函数为：",mlp.out_activation_)
# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(result))
print("预测的准确率为：", mlp.score(standard_test, Y_test))
print("层数为：", mlp.n_layers_)
print("迭代次数为：", mlp.n_iter_)
print("损失为：", mlp.loss_)
print("激活函数为：", mlp.out_activation_)


#代码的手动实现
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    #sigmoid(x) = 1/(1+e^-x)
   def sigmoid(self, x): #sigmoid计算方式
       return 1/(1+np.exp(-x))
    # sigmoid'(x) = (1-sigmoid(x))sigmoid(x)
   def sigmoid_derivative(self, x): #sigmoid导数计算方式
       return (1.0-self.sigmoid(x))*self.sigmoid(x)

   def forward(self, X):
       a1 = np.dot(X,self.weights_input_hidden) + self.bias_hidden #直接用权重和偏置计算得到的结果
       z1 = self.sigmoid(a1)    #用sigmoid转化后的概率，下同
       a2 = np.dot(z1,self.weights_hidden_output) + self.bias_output
       y  = self.sigmoid(a2)
       return y

   def backward(self, X, y, output, learning_rate):
       
       grads={} #用字典存储对应梯度，，其实可以不用存储而是在最后一步的反向传播中直接现算现用，但是这样子写可读性更好
       a1 = np.dot(X,self.weights_input_hidden) + self.bias_hidden#这一步很难受，如果把前向和反向放一个方法里面就不用多余的时间空间了
       dy = (output-y)/X.shape[0]   #预测结果的误差
       z1 = self.sigmoid(a1)
       da1 = np.dot(dy,self.weights_hidden_output.T)
       dz1 = self.sigmoid_derivative(a1) * da1
       #通过上面的变量计算出梯度
       grads['weights_hidden_output'] = np.dot(z1.T,dy)
       grads['bias_output'] = np.sum(dy,axis=0)
       grads['weights_input_hidden'] = np.dot(X.T,dz1)
       grads['bias_hidden'] = np.sum(dz1,axis=0)
       #更新params
       self.weights_hidden_output -= learning_rate* grads['weights_hidden_output']
       self.bias_output -= learning_rate* grads['bias_output']
       self.weights_input_hidden -= learning_rate* grads['weights_input_hidden']
       self.bias_hidden -= learning_rate* grads['bias_hidden']

   def train(self, X, y, epochs, learning_rate):
       for epoch in range(epochs):
           output = self.forward(X)
           loss = np.mean(0.5 * (y - output) ** 2)
           self.backward(X, y, output, learning_rate)
           if epoch % 100 == 0:
               print(f"Epoch {epoch + 1}, Loss: {loss}")
            #训练epoch次且每一百次输出对应loss

   def predict(self, X):
       return np.round(self.forward(X))#预测结果且取整数


# 将标签转换为独热编码
def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels

# 构建神经网络
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(Y_train))  # 根据训练集标签确定输出层大小
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 将标签转换为独热编码
Y_train_encoded = one_hot_encode(Y_train)#用独热码标识正确tag再用于监督学习

# 训练神经网络
print('training.......')
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.1)

# 预测测试集
predictions = nn.predict(standard_test)

# 计算准确率
accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

# 查看模型结果
print("测试集合的y值：", list(Y_test))
print("神经网络预测的的y值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)
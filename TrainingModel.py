import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 从csv文件中读取数据，分别为：X列表和对应的Y列表
def get_data(file_name):
    # 1. 用pandas读取csv
    data = pd.read_csv(file_name,  sep='\t')
    # 2. 构造X列表和Y列表
    X_parameter = []
    Y_parameter = []
    for single_square_feet,single_price_value in zip(data['square_feet'],data['price']):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))

    return X_parameter,Y_parameter

# 线性回归分析模型训练、保存
def linear_model(X_parameter, Y_parameter):
    #训练模型
    regr = LinearRegression()
    regr.fit(X_parameter, Y_parameter)
    #保存模型
    pickle.dump(regr, open('model.pkl','wb'))

if __name__ == '__main__':
    # 1. 读取数据
    X, Y = get_data('./house_price.csv')
    print(X, Y)

    # 2. 训练、保存模型
    linear_model(X, Y)

    print("模型保存完成。")
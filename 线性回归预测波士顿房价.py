from sklearn.datasets import load_boston  # 波士顿房价数据
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split  # 拆分数据集
from sklearn.preprocessing import StandardScaler  # 数据标准化处理
from sklearn.metrics import mean_squared_error  # 性能评估
import matplotlib.pyplot as plt  # 画图
from sklearn.externals import joblib  # 模型保存与加载
from matplotlib import font_manager  # 画图里显示中文字体


def myLinear():
    """"
    用线性回归进行房价预测
    :return: None
    """

    # 获取数据
    boston = load_boston()
    print("打印boston数据集:\n", boston)

    # 获取特征值、获取目标值、获取特征名称
    feature = boston["data"]
    print("特征值为:\n", feature)
    print("特征的形状:\n", feature.shape)

    target = boston["target"]
    print("target:\n", target)
    print("target 的形状:\n", target.shape)

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        boston["data"], boston["target"], test_size=0.25, random_state=22)

    print("打印训练集的目标值:\n", y_train)
    print("打印测试集的目标值:\n", y_test)

    # 特征值和目标值都必须进行标准化处理，实例两个标准API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    # 用转化训练集的标准归一化测试集
    x_test = std_x.transform(x_test)

   # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测
    # LinearRegression 基于正规方程的求解方式的线性回归
    # 应用于数据较小，特征较少，模型构建不复杂的情况下

    # （1）构建算法实例
    lr = LinearRegression()
    # （2）训练数据
    lr.fit(x_train, y_train)
    # （3）预测数据，预测房子的价格,记得还原回去
    y_predict = lr.predict(x_test)
    y_predict = std_y.inverse_transform(y_predict)
    # 打印预测值
    print("测试集里面每个房子的预测价格：", y_predict)

    # 获取准确率
    score = lr.score(x_test, y_test)

    # 获取权重与偏置
    weight = lr.coef_
    bias = lr.intercept_

    # 打印W权重参数
    print("准确率为:\n", score)
    print("权重参数为:\n", weight)
    print("偏置为:\n", bias)

    # 回归性能评估：均方误差，要注意先把测试集和测试集的目标值进行还原（因为已经做了标准化处理了）
    # 测试集的目标值要还原回去
    y_test = std_y.inverse_transform(y_test)
    print("均方误差：", mean_squared_error(y_test, y_predict))

    # 画出测试数据的真实值和模型的预测值的拟合图像
    my_font = font_manager.FontProperties(
        fname='/System/Library/Fonts/PingFang.ttc')
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, linewidth=3, label='truth')
    plt.plot(y_predict, linewidth=3, label='predict')
    plt.legend(loc='best')
    plt.xlabel('data_points')
    plt.ylabel('target_value')
    plt.title("波士顿房价预测走势图", fontproperties=my_font)
    plt.savefig("./波士顿房价真实与预测房价走势图.png")
    plt.show()

    #joblib.dump(lr,"/Users/ringzhong/Documents/VS\ code/model_save/线性回归波士顿房价预测.pkl")

    return None


if __name__ == "__main__":

    myLinear()

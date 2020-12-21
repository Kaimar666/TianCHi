import gc
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

train_url = r"..\data\used_car_train_20200313.csv"
test_url = r"..\data\used_car_testB_20200421.csv"
train_output_path = r'..\user_data\used_car_train.csv'
test_output_path = r'..\user_data\used_car_test.csv'

# 显示所有列
pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)

# 载入数据
def load_data(train_url, test_url):
    train_data = pd.read_csv(train_url, delimiter=' ')
    test_data = pd.read_csv(test_url, delimiter=' ')
    return train_data, test_data

# 删除无关列
def del_columns(data, col_name):
    """
    删除无关列
    :param data: 接收Dataframe格式数据
    :param col_name: 删除的列名
    :return: 返回删除后的新数据
    """
    data.drop(columns=col_name, axis=1, inplace=True)
    return data

# 删除缺失值对应的行
def del_null(data, col_name):
    """
    删除缺失值
    :param data: 接收Dataframe格式数据
    :param col_name: 存在缺失值的列名
    :return: 返回删除后的新数据
    """
    data.dropna(axis=0, subset=col_name)
    return data

# 删除异常值对应的行
def del_outliers(data, col_name, scale=3):
    """
    利用箱线图删除异常值
    :param data: 接收Dataframe格式数据
    :param col_name: 存在异常值的列名
    :param scale: 尺度,用于箱线图的IQR尺度,通常为3
    :return: 剔除异常值后的数据
    """
    def box_plot_outliers(data_ser, box_scale):
        """
        箱线图定义
        :param data_ser: 对应的列的series格式数据
        :param box_scale: IQR尺度
        :return:
        """
        # IQR即尺度*(上四分位点-下四分位点)
        IQR = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - IQR  # 计算下边缘
        val_up = data_ser.quantile(0.75) + IQR  # 计算上边缘
        rule_low = (data_ser < val_low)  # 小于下边缘的值
        rule_up = (data_ser > val_up)  # 大于上边缘的值
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    del_index = []
    for name in col_name:
        data_series = data_n[name]
        rule, value = box_plot_outliers(data_series, scale)
        # 对满足在下边缘和上边缘的数进行计数
        index = np.arange(data_series.shape[0])[rule[0] | rule[1]]  # shape[0] , index指对应行的行号
        del_index.extend(index)
        print("{}需要删除: {} 个数据".format(name, len(index)))
        # data_n.reset_index(drop=True, inplace=True)  # 取删除离群点后的数据; reset_index
        # print("剩余: {} 个数据".format(data_n.shape[0]))
        # index_low = np.arange(data_series.shape[0])[rule[0]]  # index_low指
        # outliers = data_series.iloc[index_low]  # 计在下边缘以下的点
        # print("小于下边缘线的数据详细:")
        # print(pd.Series(outliers).describe())
        # index_up = np.arange(data_series.shape[0])[rule[1]]  # 计在上边缘以上的点
        # outliers = data_series.iloc[index_up]  #
        # print("大于上边缘线的数据详细:")
        # print(pd.Series(outliers).describe())
        # 可视化数据
        # f = plt.figure()
        # f.add_subplot(1, 2, 1)
        # sns.boxplot(y=data[name], data=data)
        # f.add_subplot(1, 2, 2)
        # sns.boxplot(y=data_series, data=data_series)
        # plt.title(name)
    del_index = list(set(del_index))
    data_n.drop(data_n.index[del_index], inplace=True)
    return data_n


# 填充缺失值
def fill_null(data, col_name):
    for name in col_name:
        # 查看未处理的缺失情况
        print('{0}填充前：{1}'.format(name, data[name].isnull().sum()))
        # 众数填充(这里要填充的数据都是类别数据) (思考:这里如果使用时序或种类对应的数据均值填充，应该有更好的效果。待检验)
        data[name].fillna(ss.mode(data[name])[0][0], inplace=True)
        # 查看处理后的缺失情况
        print('{0}填充后: {1}'.format(name, data[name].isnull().sum()))
    return data

# 正态变换
def type_convert(data):
    # data = power_transform(np.array(data).reshape(-1,1), method='yeo-johnson')
    data = np.log(np.array(data))
    return data

# 特征构造
def create_feature(data, col_name):
    data['usedTime'] = (pd.to_datetime(data[col_name[0]], format="%Y%m%d", errors='coerce') -
                        pd.to_datetime(data[col_name[1]], format="%Y%m%d", errors='coerce')).dt.days
    print(data['usedTime'].isnull().sum())
    return data


if __name__ == "__main__":
    # 读取数据集
    train_data, test_data = load_data(train_url, test_url)

    # 不均衡数据删除
    col_name = ['name', 'model', 'power', 'regionCode']
    print("before del_outliers : {}".format(train_data.shape))
    train_data = del_outliers(train_data, col_name)
    print("after del_outliers : {}".format(train_data.shape))


    # 取出price
    price_data = train_data['price']
    print(price_data)
    train_data.drop(['price'], axis=1, inplace=True)
    f = plt.figure()
    f.add_subplot(2, 1, 1)
    sns.distplot(price_data)
    price_data = type_convert(price_data)
    f.add_subplot(2, 1, 2)
    sns.distplot(price_data)
    plt.show()

    # 拼接训练集和测试集
    data = pd.concat([train_data, test_data])
    # 记录测试集和训练集的数据量，方便拆分
    print("训练集的shape:{0}\n测试集的shape:{1}".format(train_data.shape, test_data.shape))
    del train_data
    del test_data
    gc.collect()

    # 转换object数据
    data['notRepairedDamage'].replace('-', 2.0, inplace=True)
    print(data['notRepairedDamage'].value_counts())

    # 删除特征
    del_features = ['model', 'offerType', 'seller', 'v_1', 'v_2', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_13', 'v_14']
    data = del_columns(data, del_features)


    # 特征构造
    col_name = ['creatDate', 'regDate']
    data = create_feature(data, col_name)
    data = del_columns(data, ['regDate', 'creatDate'])

    # 缺失值填充
    col_name = ['bodyType', 'fuelType', 'gearbox', 'usedTime']
    data = fill_null(data, col_name)
    print(data.isnull().sum())

    # 切分数据集
    train_data, test_data = data.iloc[:148913, :], data.iloc[148913:, :]
    train_data['price'] = price_data

    del data
    del price_data
    gc.collect()
    print(train_data.shape)
    print(train_data.head())
    print(test_data.shape)
    print(test_data.head())

    # 生成新的训练集和测试集csv文件
    train_data.to_csv(train_output_path, sep=',', index=False, header=True)
    test_data.to_csv(test_output_path, sep=',', index=False, header=True)
    print('特征工程完成。')
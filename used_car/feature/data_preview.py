import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt


warnings.filterwarnings('ignore')

train_url = r"..\data\used_car_train_20200313.csv"
test_url = r"..\data\used_car_testB_20200421.csv"

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有列
#pd.set_option('display.max_rows', None)

# 载入数据
train_data = pd.read_csv(train_url, delimiter=' ')
test_data = pd.read_csv(test_url, delimiter=' ')

# # 查看数据类型
# print(train_data.shape)
# print(train_data.info())

# # 查看数据统计量
# print(train_data.describe())

# # 查看数据前5行数据
# print(train_data.head())

# # 查看每个属性的值统计信息
# columns = train_data.columns.tolist()
# for name in columns:
#     print(train_data[name].value_counts())
# print(train_data['offerType'].value_counts())

# # 离群点查看
#arr = ['name', 'model', 'power', 'regionCode', 'creatDate']
# for name in arr:
#      dic = train_data[name].value_counts()

# # 预测值price查看
# price = train_data['price']
# #sns.distplot(price)
# print("Skewness: %f" % price.skew())
# print("Kurtosis: %f" % price.kurt())
# #plt.show()
# # price分布拟合
# plt.figure(1)
# plt.title('johnsonsu')
# sns.distplot(price, fit=ss.johnsonsu)
# plt.show()
#
# plt.figure(2)
# plt.title('norm')
# sns.distplot(price, fit=ss.norm)
# plt.show()
#
# plt.figure(3)
# plt.title('lognorm')
# sns.distplot(price, fit=ss.lognorm)
# plt.show()


# # 特殊的缺失值
# train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
# # 缺失值查看
# print(train_data.isnull().sum())

# 定量数据和定性数据
numeric_features = ['price', 'power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]

# for f in numeric_features[3:]:
#     print(train_data[f].value_counts())

# # 相关性分析,筛选匿名特征
# price_numeric = train_data[numeric_features]
# correlation = price_numeric.corr()
# print(correlation['price'].sort_values(ascending = False))
# choice = ['v_0', 'v_3', 'v_8', 'v_12']
# for name in choice:
#     print(train_data[name].value_counts())
# # print(train_data[choice].info())
# # print(train_data[choice].head())
# print(train_data[choice].describe())

# 多变量的峰度、偏度关系
for name in numeric_features:
    print('{:15}'.format(name),
          'Skewness: {:05.2f}'.format(train_data[name].skew()),
          '   ',
          'Kurtosis: {:06.2f}'.format(train_data[name].kurt())
          )

# # 多个变量关系可视化
# sns.set()
# sns.pairplot(train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
# plt.show()

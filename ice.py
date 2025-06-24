import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 模拟数据
X = np.random.rand(10000, 3)  # 100个样本，3个特征
y = X[:, 0] + 2*X[:, 1] + 3*X[:, 2] + np.random.randn(10000)  # 目标值

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# 选择要分析的特征
target_feature_index = 0  # 分析第一个特征

# 生成ICE曲线
values = np.linspace(0, 1, 50)  # 目标特征的变化范围
ice_values = []

for i in range(len(X)):
    X_sample = X[i].copy()
    predictions = []
    for value in values:
        X_sample[target_feature_index] = value
        predictions.append(model.predict([X_sample])[0])
    ice_values.append(predictions)

# 可视化ICE曲线
for result in ice_values:
    plt.plot(values, result, alpha=0.3, color='blue')
plt.title('ICE Analysis for Feature {}'.format(target_feature_index))
plt.xlabel('Feature Value')
plt.ylabel('Prediction')
plt.show()

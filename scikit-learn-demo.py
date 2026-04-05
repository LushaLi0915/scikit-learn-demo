# 导入必要的库
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 假设我们有一些模拟的金融数据
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'return': [1.1, 2.1, 3.2, 4.2, 5.3, 6.3, 7.4, 8.4, 9.5, 10.6]  # 这是我们的目标收益率
}

df = pd.DataFrame(data)

# 划分训练集和测试集
X = df[['feature1', 'feature2']]
y = df['return']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 可视化实际值和预测值
plt.figure(figsize=(8, 4))
plt.plot(y_test.values, label='Actual Returns', marker='o')
plt.plot(y_pred, label='Predicted Returns', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Returns')
plt.legend()
plt.title('Actual vs Predicted Returns')
plt.show()

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

# 忽略特定类型的警告信息
warnings.filterwarnings("ignore", message="lbfgs failed to converge")
# 恢复警告设置
# warnings.filterwarnings("default")
import os
print(os.getcwd())

df = pd.read_excel('矿区预测资料/准确矿点数据.xlsx')
print(df.shape, df.columns)

factor =  ['氡浓度(bq/m3)', '磁法(nT)', '铀含量(ug/g)', '钍含量(ug/g)', '钾含量(%)', '总放射性含量(Ur)', '钍/铀', '极化率', '视电阻率',
       'ZRn', 'Z△t', 'ZU', 'ZTh', 'ZK', 'Zur', 'ZTh/U', 'ZFs', 'Zρ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F']

label = ['是否有矿']

df['无标签数据'] = df['是否有矿'].isna()
df['是否有矿'] = df['是否有矿'].fillna(0)
X, y = df[factor].values, df[label].values

# 创建 StandardScaler 对象
scaler = StandardScaler()
# 对数据集进行标准化处理
X_standardized = scaler.fit_transform(X)

# 划分训练集和测试集
np.random.seed(76)
df['是否测试集'] =np.random.choice([0, 1], size=df.shape[0], p=[0.8, 0.2])
df['是否测试集'] = pd.read_excel('result/result.xlsx')['是否测试集']  #
X_train = X_standardized[df['是否测试集'].values == 0]
X_test = X_standardized[df['是否测试集'].values == 1]
y_train = y[df['是否测试集'].values == 0]
y_test = y[df['是否测试集'].values == 1]


# 生成新标签
# 使用 SMOTE 进行样本平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
smote_svm = SVC(kernel='rbf', C=10**2, probability=True, random_state=42)
smote_svm.fit(X_train_resampled, y_train_resampled)

df['是否有矿_新'] = smote_svm.predict(X_standardized)
df.loc[df['无标签数据'].values == False, '是否有矿_新'] = df.loc[df['无标签数据'].values == False, '是否有矿']

# 更新标签
y = df['是否有矿_新'].values
y_train = y[df['是否测试集'].values == 0]
y_test = y[df['是否测试集'].values == 1]

# 实际实验结果如下：
# 使用 SMOTE 进行样本平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 多元线性回归模型
linear_reg = make_pipeline(PolynomialFeatures(3), LogisticRegression(random_state=42))
linear_reg.fit(X_train, y_train)
linear_reg_score = linear_reg.score(X_test, y_test)
print(linear_reg_score)

# 随机森林模型
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
random_forest_score = random_forest.score(X_test, y_test)
print(random_forest_score)

# 神经网络模型
mlp = MLPClassifier(max_iter=10000, early_stopping=False, random_state=42)
mlp.fit(X_train, y_train)
mlp_score = mlp.score(X_test, y_test)
print(mlp_score)


# 支持向量机模型
svm = SVC(kernel='rbf', C=10**2, probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_score = svm.score(X_test, y_test)
print(svm_score)

# 重采样支持向量机模型
smote_svm = SVC(kernel='rbf', C=10**2, probability=True, random_state=42)
smote_svm.fit(X_train_resampled, y_train_resampled)
smote_svm_score = smote_svm.score(X_test, y_test)
print(smote_svm_score)




model_ls = [linear_reg, random_forest, mlp, svm, smote_svm]
model_name = ['Logistic Regression', 'Random Forest', 'Neural Network', 'Support Vector Machine', 'SMOTE-SVM']
model_name_short = ['LR', 'RF', 'NN', 'SVM', 'SMOTE-SVM']

# ROC 曲线对比
plt.figure(figsize=(16, 10))
for i, md in enumerate(model_ls):

    y_pred = md.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    y_pred_prob = md.predict_proba(X_test)[:, 1]
    AUC = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    print(f'{model_name_short[i]} 模型在测试集上的准确率、F1 分数、精确率、召回率、AUC 分别为：')
    print(f'准确率：{acc}')
    print(f'F1 分数：{f1}')
    print(f'精确率：{precision}')
    print(f'召回率：{recall}')
    print(f'AUC：{AUC}')
    print('\n')

    plt.plot(fpr, tpr, label=model_name[i]+' AUC({:.2f})'.format(AUC))

    df['成矿概率({})'.format(model_name_short[i])] = md.predict_proba(X_standardized)[:, 1]

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('ROC.png')
plt.show()

df.to_excel('result.xlsx', index=False)





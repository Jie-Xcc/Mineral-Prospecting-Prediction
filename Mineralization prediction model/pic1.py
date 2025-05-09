import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. 读取 Excel 数据
# 请根据实际文件路径替换 "your_file.xlsx"
file_path = "result/result.xlsx"
data = pd.read_excel(file_path)

# 2. 提取坐标和 F1~F5, F 指标
X = data["X"].values
Y = data["Y"].values

# 创建插值网格
xi = np.linspace(X.min(), X.max(), 200)
yi = np.linspace(Y.min(), Y.max(), 200)

# 重新获取F1至F5的数据，以及F的数据
zi_F1 = griddata((X, Y), data["F1"], (xi[None, :], yi[:, None]), method='cubic')
zi_F2 = griddata((X, Y), data["F2"], (xi[None, :], yi[:, None]), method='cubic')
zi_F3 = griddata((X, Y), data["F3"], (xi[None, :], yi[:, None]), method='cubic')
zi_F4 = griddata((X, Y), data["F4"], (xi[None, :], yi[:, None]), method='cubic')
zi_F5 = griddata((X, Y), data["F5"], (xi[None, :], yi[:, None]), method='cubic')
zi_F = griddata((X, Y), data["F"], (xi[None, :], yi[:, None]), method='cubic')

# 4. 设置阈值（可根据实际调整或使用统计值）
thresholds = {
    'F1': np.nanmean(data["F1"]),
    'F2': np.nanmean(data["F2"]),
    'F3': np.nanmean(data["F3"]),
    'F4': np.nanmean(data["F4"]),
    'F5': np.nanmean(data["F5"]),
    'F':  np.nanmean(data["F"])
}

# 使用等值线图来标记异常区域
plt.figure(figsize=(10, 8))
contour_F1 = plt.contour(xi, yi, zi_F1, levels=[thresholds['F1']], colors='blue', linestyles='dashed', linewidths=1)
contour_F2 = plt.contour(xi, yi, zi_F2, levels=[thresholds['F2']], colors='green', linestyles='dashed', linewidths=1)
contour_F3 = plt.contour(xi, yi, zi_F3, levels=[thresholds['F3']], colors='red', linestyles='dashed', linewidths=1)
contour_F4 = plt.contour(xi, yi, zi_F4, levels=[thresholds['F4']], colors='purple', linestyles='dashed', linewidths=1)
contour_F5 = plt.contour(xi, yi, zi_F5, levels=[thresholds['F5']], colors='yellow', linestyles='dashed', linewidths=1)
contour_F = plt.contour(xi, yi, zi_F, levels=[thresholds['F']], colors='black', linestyles='dashed', linewidths=2)

# 添加图例
h1, _ = contour_F1.legend_elements()
h2, _ = contour_F2.legend_elements()
h3, _ = contour_F3.legend_elements()
h4, _ = contour_F4.legend_elements()
h5, _ = contour_F5.legend_elements()
h6, _ = contour_F.legend_elements()

plt.legend([h1[0], h2[0], h3[0], h4[0], h5[0], h6[0]],
           ['F1', 'F2', 'F3', 'F4', 'F5', 'Composite Score F'])

plt.scatter(X, Y, c='white', s=10, marker='o', edgecolors='black')  # 样点位置
plt.title("Potential Ore-forming Regions")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()

plt.show()

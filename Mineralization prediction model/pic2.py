import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 我们可以尝试以下方法来更清晰地显示被多条异常阈值线圈定的区域：
# 1. 绘制每个因子得分的异常阈值线。
# 2. 根据每个点被多少条异常阈值线圈定来为每个点分配一个得分。
# 3. 使用颜色图来显示这些得分，从而清晰地显示被多条异常阈值线圈定的区域。


# 1. 读取 Excel 数据
file_path = "result/result.xlsx"
data = pd.read_excel(file_path)

# 2. 提取坐标和 F1~F5, F 指标
X = data["X"].values
Y = data["Y"].values

# 创建插值网格
xi = np.linspace(X.min(), X.max(), 200)
yi = np.linspace(Y.min(), Y.max(), 200)

# 3. 插值因子数据
zi_F1 = griddata((X, Y), data["F1"], (xi[None, :], yi[:, None]), method='cubic')
zi_F2 = griddata((X, Y), data["F2"], (xi[None, :], yi[:, None]), method='cubic')
zi_F3 = griddata((X, Y), data["F3"], (xi[None, :], yi[:, None]), method='cubic')
zi_F4 = griddata((X, Y), data["F4"], (xi[None, :], yi[:, None]), method='cubic')
zi_F5 = griddata((X, Y), data["F5"], (xi[None, :], yi[:, None]), method='cubic')
zi_F  = griddata((X, Y), data["F"],  (xi[None, :], yi[:, None]), method='cubic')

# 4. 阈值设定（可替换为固定值或百分位数）
thresholds = {
    'F1': np.nanmean(data["F1"]),
    'F2': np.nanmean(data["F2"]),
    'F3': np.nanmean(data["F3"]),
    'F4': np.nanmean(data["F4"]),
    'F5': np.nanmean(data["F5"]),
    'F':  np.nanmean(data["F"])
}

# 计算每个点被多少条异常阈值线圈定的得分
scores = np.zeros_like(zi_F)
scores[zi_F1 > thresholds['F1']] += 1
scores[zi_F2 > thresholds['F2']] += 1
scores[zi_F3 > thresholds['F3']] += 1
scores[zi_F4 > thresholds['F4']] += 1
scores[zi_F5 > thresholds['F5']] += 1
scores[zi_F > thresholds['F']] += 1

# 使用颜色图来显示得分
plt.figure(figsize=(10, 8))
plt.imshow(scores, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='Reds', aspect='auto')
cbar = plt.colorbar(label="Number of Thresholds Exceeded")

# 再次绘制异常阈值线
contour_F1 = plt.contour(xi, yi, zi_F1, levels=[thresholds['F1']], colors='blue', linestyles='dashed', linewidths=1)
contour_F2 = plt.contour(xi, yi, zi_F2, levels=[thresholds['F2']], colors='green', linestyles='dashed', linewidths=1)
contour_F3 = plt.contour(xi, yi, zi_F3, levels=[thresholds['F3']], colors='red', linestyles='dashed', linewidths=1)
contour_F4 = plt.contour(xi, yi, zi_F4, levels=[thresholds['F4']], colors='purple', linestyles='dashed', linewidths=1)
contour_F5 = plt.contour(xi, yi, zi_F5, levels=[thresholds['F5']], colors='yellow', linestyles='dashed', linewidths=1)
contour_F = plt.contour(xi, yi, zi_F, levels=[thresholds['F']], colors='black', linestyles='dashed', linewidths=2)

plt.scatter(X, Y, c='white', s=10, marker='o', edgecolors='black')  # 样点位置
plt.title("Potential Ore-forming Regions Highlighted")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()

plt.show()

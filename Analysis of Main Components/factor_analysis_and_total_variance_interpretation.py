import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: 数据读取与标准化
# -----------------------------
file_path = '矿区预测资料/因子计算.xlsx'
df = pd.read_excel(file_path)
df.fillna(df.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)  # Z-score 标准化

# -----------------------------
# Step 2: 主成分分析（PCA）
# -----------------------------
pca = PCA(n_components=5)  # 提取前5个成分
pca.fit(X_scaled)

# 获取初始特征值等信息
explained_variance = pca.explained_variance_  # 初始特征值
explained_variance_ratio = pca.explained_variance_ratio_ * 100
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 构建载荷矩阵（Feature x Component）
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# -----------------------------
# Step 3: Kaiser 正态化函数
# -----------------------------
def kaiser_normalize(loadings):
    """对载荷矩阵的每一行做单位向量归一化"""
    norms = np.linalg.norm(loadings, axis=1)
    norms[norms == 0] = 1  # 防止除零错误
    normalized_loadings = loadings / norms[:, np.newaxis]
    return normalized_loadings, norms

# -----------------------------
# Step 4: 带 Kaiser 正态化的 Varimax 旋转函数（限制6次迭代）
# -----------------------------
def varimax_rotation_kaiser(F, q=6, gamma=1.0, tol=1e-6):
    """
    Perform Kaiser-normalized Varimax Rotation of the loading matrix F.

    Parameters:
    - F: np.array, shape (n_features, n_factors), loading matrix
    - q: int, optional (max number of iterations)
    - gamma: float, optional (default=1.0 for Varimax)
    - tol: float, optional (convergence tolerance)

    Returns:
    - L: rotated and re-scaled loading matrix
    """
    from numpy.linalg import svd

    # Step 1: Kaiser 正态化
    F_normalized, norms = kaiser_normalize(F)

    p, k = F_normalized.shape
    R = np.eye(k)
    d = 0

    # Step 2: Varimax 旋转（最多 q 次迭代）
    for _ in range(q):  # 控制最多迭代次数为 6 次
        d_old = d
        Lambda = F_normalized @ R

        # 手动检查 SVD 分解细节
        G = F_normalized.T @ (Lambda ** 3 - (gamma / k) * Lambda * np.diag(np.dot(Lambda.T, Lambda)))
        u, s, vh = svd(G)

        R = u @ vh
        d = np.sum(s ** 2)  # 使用 s 的平方和作为目标函数

        if d < d_old * (1 + tol):
            break

    rotated_normalized = F_normalized @ R
    rotated_original_scale = rotated_normalized * norms[:, np.newaxis]

    return rotated_original_scale

# -----------------------------
# Step 5: 应用旋转并计算旋转载荷平方和
# -----------------------------
rotated_loadings = varimax_rotation_kaiser(loadings, q=6)

# 计算旋转载荷平方和
rotation_loadings_squared_sum = np.sum(rotated_loadings ** 2, axis=0)
rotation_variance_ratio = (rotation_loadings_squared_sum / rotation_loadings_squared_sum.sum()) * 100
rotation_cumulative_variance = np.cumsum(rotation_variance_ratio)

# -----------------------------
# Step 6: 构建结果 DataFrame
# -----------------------------
result_data = {
    "成分": [f"F{i + 1}" for i in range(5)],
    "初始特征值-总计": explained_variance,
    "初始特征值-方差百分比": explained_variance_ratio,
    "初始特征值-累积%": cumulative_variance_ratio,
    "提取载荷平方和-总计": explained_variance,  # PCA 中提取载荷平方和 = 特征值
    "提取载荷平方和-方差百分比": explained_variance_ratio,
    "提取载荷平方和-累积%": cumulative_variance_ratio,
    "旋转载荷平方和-总计": rotation_loadings_squared_sum,
    "旋转载荷平方和-方差百分比": rotation_variance_ratio,
    "旋转载荷平方和-累积%": rotation_cumulative_variance,
}

# 创建 DataFrame 并设置多级表头
df_result = pd.DataFrame(result_data)

df_result = df_result[
    ["成分",
     "初始特征值-总计", "初始特征值-方差百分比", "初始特征值-累积%",
     "提取载荷平方和-总计", "提取载荷平方和-方差百分比", "提取载荷平方和-累积%",
     "旋转载荷平方和-总计", "旋转载荷平方和-方差百分比", "旋转载荷平方和-累积%"]
]

df_result.columns = pd.MultiIndex.from_tuples([
    ('', '成分'),
    ('初始特征值', '总计'), ('初始特征值', '方差百分比'), ('初始特征值', '累积%'),
    ('提取载荷平方和', '总计'), ('提取载荷平方和', '方差百分比'), ('提取载荷平方和', '累积%'),
    ('旋转载荷平方和', '总计'), ('旋转载荷平方和', '方差百分比'), ('旋转载荷平方和', '累积%')
])

# 输出到控制台 & Excel 文件
print(df_result)
df_result.to_excel('pca_with_varimax_kaiser_rotated_corrected.xlsx', index=True)

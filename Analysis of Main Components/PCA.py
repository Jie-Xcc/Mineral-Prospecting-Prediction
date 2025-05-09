import numpy as np


class PCA:

    def __init__(self, n_components):  # n_components表示想要有多少个主成分
        """初始化PCA"""
        assert n_components >= 1, \
            "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

    def fit(self, X, eta=0.01, n_iters=1e4):  # 用户需要传入X
        """根据训练集X获得n_components个主成分"""
        assert X.shape[1] >= self.n_components, \
            "n_components must be greater than the feature number of X"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            """目标函数"""
            return np.sum((X.dot(w)) ** 2) / len(X)

        def df(w, X):
            """目标函数的梯度"""
            return X.T.dot(X.dot(w)) * 2 / len(X)

        def directionize(w):
            """单位化向量"""
            return w / np.linalg.norm(w)

        def first_componet(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            """梯度上升法求解给定X的第一主成分"""
            w = directionize(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = directionize(w)  # 注意：每次求一个单位方向
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1

            return w

        X_pca = demean(X)
        # components_相当于W_k矩阵：W(k,n)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        res = []
        # 通过n次循环拿到n个主成分
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_componet(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        return self

    def transform(self, X):
        """将给定的X,映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的低维特征空间X转换回高维空间中"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

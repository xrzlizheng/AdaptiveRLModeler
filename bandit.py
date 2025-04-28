####################Multi-Armed Bandit Method##############################
# 导入必要的库
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 CSV 文件中读取数据集
data = pd.read_csv('rein_data_binary.csv')
# 提取特征
X = data.drop('label', axis=1)
# 提取标签
y = data['label']

# ---------------------------------------------# 2) 计算评估指标 (AUC, KS)# ---------------------------------------------# 计算 KS 分数的函数
def calc_ks_score(y_true, y_prob):
    # 创建一个包含真实标签和预测概率的 DataFrame，并按预测概率降序排序
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    # 计算正样本的累积数量
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    # 计算负样本的累积数量
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    # 计算正样本的总数
    total_pos = data['y_true'].sum()
    # 计算负样本的总数
    total_neg = (data['y_true'] == 0).sum()
    # 计算正样本的累积比率
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    # 计算负样本的累积比率
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    # 计算 KS 值
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()

# ---------------------------------------------# 3) 定义 PyTorch 深度神经网络模型# ---------------------------------------------# 定义 DNN 模型类
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        # 调用父类的构造函数
        super(DNNModel, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_dim, 16)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(16, 8)
        # 定义输出层
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        # 第一个全连接层后使用 ReLU 激活函数
        x = F.relu(self.fc1(x))
        # 第二个全连接层后使用 ReLU 激活函数
        x = F.relu(self.fc2(x))
        # 输出层使用 Sigmoid 激活函数
        x = torch.sigmoid(self.out(x))
        return x

# 训练并评估 PyTorch DNN 模型的函数
def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    # 初始化 DNN 模型
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    # 定义优化器，使用 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义损失函数，使用二元交叉熵损失函数
    criterion = nn.BCELoss()

    # 将训练数据转换为 PyTorch 张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    # 将训练标签转换为 PyTorch 张量
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    # 将验证数据转换为 PyTorch 张量
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    # 计算数据集的大小
    dataset_size = len(X_train_t)
    # 计算批次数量
    n_batches = (dataset_size // batch_size) + 1

    # 开始训练循环
    for epoch in range(epochs):
        # 生成随机索引，用于打乱数据
        indices = torch.randperm(dataset_size)
        # 打乱训练数据
        X_train_t = X_train_t[indices]
        # 打乱训练标签
        y_train_t = y_train_t[indices]

        # 开始批次训练循环
        for i in range(n_batches):
            # 计算当前批次的起始索引
            start_idx = i * batch_size
            # 计算当前批次的结束索引
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            # 提取当前批次的训练数据
            x_batch = X_train_t[start_idx:end_idx]
            # 提取当前批次的训练标签
            y_batch = y_train_t[start_idx:end_idx]

            # 前向传播，得到模型预测结果
            preds = model(x_batch)
            # 计算损失
            loss = criterion(preds, y_batch)

            # 清空优化器的梯度
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

    # 关闭梯度计算，进行验证
    with torch.no_grad():
        # 得到验证集的预测结果
        val_preds = model(X_val_t).cpu().numpy().ravel()

    # 计算验证集的 AUC 分数
    auc = roc_auc_score(y_val, val_preds)
    # 计算验证集的 KS 分数
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds

# ---------------------------------------------# 4) 辅助函数：训练并评估各种模型# ---------------------------------------------# 训练并评估不同模型的函数
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    if model_name == 'xgb':
        # 初始化 XGBoost 分类器
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # 训练模型
        model.fit(X_train, y_train)
        # 得到验证集的预测概率
        y_prob = model.predict_proba(X_val)[:, 1]
        # 计算验证集的 AUC 分数
        auc = roc_auc_score(y_val, y_prob)
        # 计算验证集的 KS 分数
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        # 初始化 LightGBM 分类器
        model = LGBMClassifier()
        # 训练模型
        model.fit(X_train, y_train)
        # 得到验证集的预测概率
        y_prob = model.predict_proba(X_val)[:, 1]
        # 计算验证集的 AUC 分数
        auc = roc_auc_score(y_val, y_prob)
        # 计算验证集的 KS 分数
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        # 初始化随机森林分类器
        model = RandomForestClassifier()
        # 训练模型
        model.fit(X_train, y_train)
        # 得到验证集的预测概率
        y_prob = model.predict_proba(X_val)[:, 1]
        # 计算验证集的 AUC 分数
        auc = roc_auc_score(y_val, y_prob)
        # 计算验证集的 KS 分数
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        # 调用 train_eval_pytorch_dnn 函数训练并评估 DNN 模型
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        # 如果模型名称未知，抛出异常
        raise ValueError(f"Unknown model name: {model_name}")

# Continue with the rest of the code unchanged...# ---------------------------------------------# 5) 加权融合预测结果# ---------------------------------------------# 加权融合预测概率的函数
def blend_predictions(probs_list, weights=None):
    if weights is None:
        # 如果没有提供权重，使用平均权重
        weights = [1.0 / len(probs_list)] * len(probs_list)
    # 初始化最终预测概率
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        # 加权累加预测概率
        final_prob += w * p
    return final_prob

# 评估动作的函数
def evaluate_action(action, X_train, X_val, y_train, y_val, device='cpu'):
    """
    action: 整数，范围从 0 到 4，分别对应 (xgb=0, lgbm=1, rf=2, dnn=3, blend=4)
    返回值:
      reward = (auc + ks) - penalty
      auc, ks
    """
    model_names = ['xgb', 'lgbm', 'rf', 'dnn']
    if action < 4:
        # 选择对应的模型
        chosen_model = model_names[action]
        _, auc_val, ks_val, _ = train_eval_model(chosen_model, X_train, y_train, X_val, y_val, device=device)
        # 如果选择的是 DNN 模型，添加惩罚项
        penalty = 0.05 if chosen_model == 'dnn' else 0.0
        # 计算奖励
        reward = (auc_val + ks_val) - penalty
        return reward, auc_val, ks_val
    else:
        # 融合模型预测结果
        probs_list = []
        for m in model_names:
            _, auc_m, ks_m, p = train_eval_model(m, X_train, y_train, X_val, y_val, device=device)
            probs_list.append(p)
        # 融合预测概率
        final_prob = blend_predictions(probs_list)
        # 计算融合后的 AUC 分数
        auc_blend = roc_auc_score(y_val, final_prob)
        # 计算融合后的 KS 分数
        ks_blend = calc_ks_score(y_val, final_prob)
        # 计算奖励，添加融合惩罚项
        reward = (auc_blend + ks_blend) - 0.1
        return reward, auc_blend, ks_blend

# ---------------------------------------------# 6) 简单的多臂老虎机方法# ---------------------------------------------# 多臂老虎机模型选择函数
def multi_armed_bandit_model_selection(
    n_episodes=50,
    n_actions=5,
    epsilon=0.06,
    device='cpu'
):
    """
    有 5 个动作 (xgb=0, lgbm=1, rf=2, dnn=3, blend=4)。
    对于每个 'episode':
      1) 使用选定的种子生成数据集 (X,y)
      2) 划分为训练集和验证集
      3) 使用 ε-greedy 策略选择一个动作
      4) 评估所选动作，得到奖励
      5) 更新该动作的平均奖励 (Q)
    """
    # 初始化动作价值数组
    Q = np.zeros(n_actions, dtype=np.float32)
    # 初始化每个动作的选择次数数组
    counts = np.zeros(n_actions, dtype=int)

    # 用于存储每次选择动作时的原始 AUC、KS 和奖励
    action_auc_records = [[] for _ in range(n_actions)]
    action_ks_records = [[] for _ in range(n_actions)]
    action_reward_records = [[] for _ in range(n_actions)]

    # 存储动作历史记录
    action_history = []
    # 存储奖励历史记录
    reward_history = []

    # 开始训练循环
    for episode in range(n_episodes):
        # 生成随机种子
        seed = 1000 + episode
        # 提取特征
        X = data.drop('label', axis=1)  # Features
        # 提取标签
        y = data['label']  # Labels

        # 划分数据集为训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123)

        # ε-greedy 动作选择
        if np.random.rand() < epsilon:
            # 探索：随机选择一个动作
            action = np.random.randint(n_actions)
        else:
            # 利用：选择动作价值最大的动作
            action = np.argmax(Q)

        # 评估所选动作，得到奖励、AUC 和 KS
        reward, auc_val, ks_val = evaluate_action(
            action, X_train, X_val, y_train, y_val, device=device
        )

        # 更新动作价值 (增量平均)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        # 存储详细信息
        action_history.append(action)
        reward_history.append(reward)
        action_auc_records[action].append(auc_val)
        action_ks_records[action].append(ks_val)
        action_reward_records[action].append(reward)

        print(f"Episode {episode+1}/{n_episodes}, "
              f"Action={action}, Reward={reward:.4f}, Updated Q={Q}")

    return Q, action_history, reward_history, action_auc_records, action_ks_records, action_reward_records

# ---------------------------------------------# 7) 运行多臂老虎机算法，然后解释结果# ---------------------------------------------# 运行多臂老虎机算法的函数
def run_bandit():
    # 检查是否有可用的 GPU，选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device={device}")

    # 定义训练轮数
    n_episodes = 50
    # 定义动作数量
    n_actions = 5
    # 定义探索率
    epsilon = 0.05

    (
        Q,
        actions,
        rewards,
        auc_records,
        ks_records,
        reward_records
    ) = multi_armed_bandit_model_selection(
        n_episodes=n_episodes,
        n_actions=n_actions,
        epsilon=epsilon,
        device=device
    )

    # 找到最优动作的索引
    best_action = np.argmax(Q)
    # 定义模型名称列表
    model_names = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]

    print("\n========================================")
    print("Interpreting Your Current Results")
    print("========================================\n")

    print("Final Q-values:", Q)
    print(f"Best action index: {best_action}")
    print(f"Best action is: {model_names[best_action]} with estimated Q = {Q[best_action]:.4f}\n")

    print("Detailed AUC/KS/Reward by action:")
    print("--------------------------------------------------")
    for a in range(n_actions):
        if len(auc_records[a]) > 0:
            # 计算平均 AUC
            avg_auc = np.mean(auc_records[a])
            # 计算平均 KS
            avg_ks = np.mean(ks_records[a])
            # 计算平均奖励
            avg_reward = np.mean(reward_records[a])
            print(f"Action {a} ({model_names[a]}): chosen {len(auc_records[a])} times")
            print(f"  Mean AUC = {avg_auc:.4f}, Mean KS = {avg_ks:.4f}, Mean Reward = {avg_reward:.4f}\n")
        else:
            print(f"Action {a} ({model_names[a]}): chosen 0 times\n")


if __name__ == "__main__":
    run_bandit()





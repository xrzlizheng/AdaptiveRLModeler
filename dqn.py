###########深度 Q 网络 (DQN)###################
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

# Gymnasium 强化学习环境库
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3 强化学习库
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# 用于回调函数
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------
# 1) 从 CSV 文件中读取数据
# ---------------------------------------------
data = pd.read_csv('rein_data_binary.csv')
X = data.drop('label', axis=1)  # 特征
y = data['label']  # 标签

# ---------------------------------------------
# 2) 评估指标 (AUC, KS)
# ---------------------------------------------
def calc_ks_score(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    total_pos = data['y_true'].sum()
    total_neg = (data['y_true'] == 0).sum()
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()

# ---------------------------------------------
# 3) PyTorch 深度神经网络
# ---------------------------------------------
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds

# ---------------------------------------------
# 4) 训练和评估辅助函数
# ---------------------------------------------
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    if model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def blend_predictions(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob

# ---------------------------------------------
# 5) 单步环境
# ---------------------------------------------
class ModelSelectionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, X, y, device='cpu'):
        super().__init__()
        self.device = device

        # 训练集/验证集划分
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=123
        )

        means = X.mean().values
        vars_ = X.var().values
        self.state = np.concatenate([means, vars_])  # 观测值

        # 5 个离散动作
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state),),
            dtype=np.float32
        )
        self.terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated = False
        return self.state.astype(np.float32), {}

    def step(self, action):
        if self.terminated:
            return self.state.astype(np.float32), 0.0, True, False, {}

        model_names = ['xgb', 'lgbm', 'rf', 'dnn']
        if action < 4:
            chosen_model = model_names[action]
            _, auc_v, ks_v, _ = train_eval_model(
                chosen_model, 
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                device=self.device
            )
            penalty = 0.05 if chosen_model == 'dnn' else 0.0
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": chosen_model,
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }
        else:
            # Blend
            probs_list = []
            for m in model_names:
                _, auc_m, ks_m, prob_m = train_eval_model(
                    m,
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    device=self.device
                )
                probs_list.append(prob_m)
            final_prob = blend_predictions(probs_list)
            auc_v = roc_auc_score(self.y_val, final_prob)
            ks_v = calc_ks_score(self.y_val, final_prob)
            penalty = 0.1
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": "blend",
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }

        self.terminated = True
        return self.state.astype(np.float32), reward, True, False, info

# ---------------------------------------------
# 7) 强化学习训练和执行
# ---------------------------------------------
def run_rl_model_selection_pytorch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # 创建单步 Gymnasium 环境
    env = ModelSelectionEnv(X, y, device=device)

    # 使用 DummyVecEnv 包装环境
    def make_env():
        return env
    vec_env = DummyVecEnv([make_env])

    # 创建回调函数
    callback = BanditSummaryCallback()

    # 创建 DQN 模型
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./rl_tensorboard/"
    )

    # 使用回调函数进行训练
    model.learn(total_timesteps=2000, callback=callback)

    # 评估最终策略（单步）
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    final_reward = rewards[0]
    action_map = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]
    print("\n======================================")
    print(f"Final chosen action => {action[0]} ({action_map[action[0]]})")
    print(f"Final step reward => (AUC + KS - penalty) = {final_reward:.4f}")
    print("======================================\n")

if __name__ == "__main__":
    run_rl_model_selection_pytorch()

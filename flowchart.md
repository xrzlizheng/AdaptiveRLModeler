# 强化学习算法流程图对比

## 目录
- [并排对比](#对比流程图)
- [独立流程图](#独立流程图)

## 对比流程图
```mermaid
flowchart TD
    subgraph DQN算法流程
        A1[初始化Q网络] --> B1[收集环境状态]
        B1 --> C1{选择动作}
        C1 -->|ε-greedy| D1[执行动作]
        D1 --> E1[存储经验]
        E1 --> F1[样本训练]
        F1 --> G1[更新目标网络]
    end
    
    subgraph Bandit算法流程
        A2[初始化模型权重] --> B2[计算置信区间]
        B2 --> C2{选择最优臂}
        C2 -->|UCB策略| D2[获得反馈]
        D2 --> E2[更新模型参数]
    end
```

## 独立流程图
### DQN独立流程
```mermaid
flowchart TD
    A[初始化Replay Buffer] --> B[观测当前状态]
    B --> C{ε策略选择}
    C -->|探索| D[随机动作]
    C -->|利用| E[Q网络预测]
    D --> F[执行动作]
    E --> F
    F --> G[存储transition]
    G --> H[抽样训练]
    H --> I[更新目标网络]
```

### Bandit独立流程
```mermaid
flowchart TD
    A[初始化各臂统计值] --> B[遍历候选臂]
    B --> C[计算置信上界]
    C --> D{选择最大UCB}
    D --> E[执行选择]
    E --> F[更新统计值]
```
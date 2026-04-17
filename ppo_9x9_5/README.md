# PPO Curriculum

独立的 PPO 训练目录，支持：
- `5x5_4`
- `7x7_5`
- `9x9_5`

包含：
- `gomoku_ppo.py`：单文件 PPO 训练器
- `train_ppo_5x5_4.sh`
- `train_ppo_7x7_5.sh`
- `train_ppo_9x9_5.sh`
- `train_ppo_curriculum.sh`

默认配置：
- `channels=128`
- `conv_layers=8`

直接跑完整课程：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/ppo_9x9_5
DEVICE=cuda ./train_ppo_curriculum.sh
```

单阶段示例：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/ppo_9x9_5
DEVICE=cuda ./train_ppo_5x5_4.sh
DEVICE=cuda INIT_CHECKPOINT=./gomoku_ppo_7x7_5_last.pt ./train_ppo_9x9_5.sh
```

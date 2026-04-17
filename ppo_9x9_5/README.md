# PPO 9x9 Connect-5

独立的 `9x9_5` PPO 训练目录。

包含：
- `gomoku_ppo.py`：单文件 PPO 训练器
- `train_ppo_9x9_5.sh`：一键训练脚本

默认配置：
- `board_size=9`
- `win_length=5`
- `channels=128`
- `conv_layers=8`

直接训练：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/ppo_9x9_5
DEVICE=cuda ./train_ppo_9x9_5.sh
```

Mac / MPS：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/ppo_9x9_5
DEVICE=mps ./train_ppo_9x9_5.sh
```

快速试跑：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/ppo_9x9_5
DEVICE=cuda ITERATIONS=200 GAMES_PER_ITER=16 EVAL_EVERY=5 ./train_ppo_9x9_5.sh
```

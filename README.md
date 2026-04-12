# Minimal Gomoku Policy Gradient

这是一个学习向的最简五子棋策略梯度示例，核心特点：

- 一个文件：`gomoku_pg.py`
- 可配置棋盘大小：例如 `5x5`、`15x15`
- 可配置连珠数：例如 `4` 连珠、`5` 连珠
- 使用 `torch` 和精简版 `actor-critic` policy gradient
- 同一个策略同时扮演先手和后手，自博弈训练

## 核心思路

状态编码是 3 个平面：

1. 当前行动方自己的棋子
2. 对手的棋子
3. 合法落点

策略网络是一个很小的全卷积网络，输出每个格子的 logits。非法位置会被 mask 掉，然后对合法位置做采样。

训练时：

1. 用当前策略自博弈完整下一局
2. 每一步保存 `log_prob(action)`
3. 终局后给每一步一个回报
   当前步所属玩家最终赢了就是 `+1`
   输了就是 `-1`
   平局就是 `0`
4. 策略头用 advantage 做 policy gradient，价值头预测回报，降低方差
5. 训练时随机旋转/翻转棋盘，提升样本效率

## 先做小棋盘验证

建议先验证：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py train \
  --board-size 5 \
  --win-length 4 \
  --episodes 5000 \
  --batch-size 32 \
  --eval-every 300 \
  --eval-games 40 \
  --checkpoint gomoku_5x5_4.pt
```

评估：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py eval \
  --board-size 5 \
  --win-length 4 \
  --checkpoint gomoku_5x5_4.pt \
  --agent mcts \
  --mcts-sims 120 \
  --games 100
```

图形界面对弈验证：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py gui \
  --checkpoint gomoku_5x5_4.pt \
  --agent mcts \
  --mcts-sims 120 \
  --human-first
```

操作：

- 鼠标左键落子
- `R` 重新开始
- `Esc` 退出

如果还没装 `pygame`：

```bash
~/miniconda3/bin/conda run -n lerobot python -m pip install pygame
```

人机对弈：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py play \
  --board-size 5 \
  --win-length 4 \
  --checkpoint gomoku_5x5_4.pt \
  --agent mcts \
  --mcts-sims 120 \
  --human-first
```

## 切换到标准五子棋

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py train \
  --board-size 15 \
  --win-length 5 \
  --episodes 20000 \
  --batch-size 32 \
  --eval-every 1000 \
  --eval-games 40 \
  --checkpoint gomoku_15x15_5.pt
```

注意：代码可以直接切棋盘大小，但模型参数需要重新训练，不能指望 `5x5 + 4 连珠` 学到的策略直接适用于 `15x15 + 5 连珠`。

## 怎么验证算法

最直接的验证顺序：

1. 先训练 `5x5 + 4 连珠`
2. 用 `eval` 看对随机策略胜率是否明显高于 50%
3. 用 `gui` 人工对弈，观察它是否会优先补成四连、阻挡你的四连
4. 再切到 `15x15 + 5 连珠` 重新训练

如果你只是想验证实现有没有大错，先看小棋盘最有效，因为训练快，策略错误会更明显。

## 为什么你会很容易赢

如果你之前用的是最原始的终局奖励 `REINFORCE`，很容易出现这几个问题：

- 终局奖励太稀疏，前面大量落子几乎收不到有效学习信号
- 方差很大，训练出来的策略不稳定
- `15x15` 动作空间太大，从零自博弈非常慢

这版已经改成更稳的 `actor-critic`。即便如此，标准五子棋从零训练仍然不可能靠几百局就变强。

## 推理时 MCTS

现在 `eval`、`play`、`gui` 都支持：

- `--agent policy`：直接让策略网络落子
- `--agent mcts`：让策略网络和值网络先做 MCTS 搜索，再落子

建议人机测试默认用 `mcts`，通常会比直接落子强一截。

例如：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py gui \
  --checkpoint gomoku_15x15_5.pt \
  --agent mcts \
  --mcts-sims 120 \
  --human-first
```

如果你觉得慢，可以先把 `--mcts-sims` 降到 `32` 或 `64`。

## 更现实的训练方式

建议这样做：

1. 先训 `5x5 + 4 连珠`
2. 再用小棋盘权重热启动更大的棋盘
3. 最后再训 `15x15 + 5 连珠`

例如：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py train \
  --board-size 7 \
  --win-length 5 \
  --episodes 5000 \
  --init-checkpoint gomoku_5x5_4.pt \
  --checkpoint gomoku_7x7_5.pt
```

再继续：

```bash
~/miniconda3/bin/conda run -n lerobot python gomoku_pg.py train \
  --board-size 15 \
  --win-length 5 \
  --episodes 20000 \
  --init-checkpoint gomoku_7x7_5.pt \
  --checkpoint gomoku_15x15_5.pt
```

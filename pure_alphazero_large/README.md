# Pure AlphaZero Large

这是一套独立的大模型课程训练目录。

默认网络结构：
- `channels=256`
- `conv_layers=10`

默认课程：
- `5x5_4`
- `7x7_5`
- `9x9_5`
- `11x11_5`
- `13x13_5`
- `15x15_5`

所有 checkpoint 默认保存在当前目录，文件名统一为：
- `gomoku_az_large_5x5_4.pt`
- `gomoku_az_large_7x7_5.pt`
- `gomoku_az_large_9x9_5.pt`
- `gomoku_az_large_11x11_5.pt`
- `gomoku_az_large_13x13_5.pt`
- `gomoku_az_large_15x15_5.pt`

直接跑完整课程：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/pure_alphazero_large
DEVICE=cuda ./train_az_curriculum.sh
```

单阶段示例：

```bash
cd /Users/liuchu/codes/gomoku-ai-code/pure_alphazero_large
DEVICE=cuda ./train_az_5x5_4.sh
DEVICE=cuda INIT_CHECKPOINT=./gomoku_az_large_9x9_5_last.pt ./train_az_11x11_5.sh
```

from board import Board
from ai_webui import WebUIAI

# 创建一个测试棋盘
board = Board()

# 在中心位置下一个黑子
board.set(9, 9, 1)

# 创建AI实例
ai = WebUIAI(player=2, difficulty='hard')

# 测试AI走棋
print("Testing AI move...")
best_move = ai.get_best_move(board)
print(f"AI's best move: {best_move}")

# 打印棋盘状态
print("Board state:")
for y in range(19):
    row = []
    for x in range(19):
        row.append(str(board.get(x, y)))
    print(' '.join(row))
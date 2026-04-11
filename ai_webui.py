import time
import math
import random
from board import Board

class WebUIAI:
    def __init__(self, player=2, difficulty='hard'):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.difficulty = difficulty
        self.BOARD_SIZE = 19
        self.WIN_COUNT = 6
        
        # 方向数组：右、下、右下、左下
        self.DX = [1, 0, 1, -1]
        self.DY = [0, 1, 1, 1]
    
    def get_best_move(self, board):
        # 寻找最佳走法
        moves = self.find_best_moves(board)
        if moves:
            return (moves[0]['x'], moves[0]['y'])
        else:
            # 尝试随机选择一个空位
            size = board.size
            for x in range(size):
                for y in range(size):
                    if board.get(x, y) == 0:
                        return (x, y)
            return None
    
    def find_best_moves(self, board):
        moves = []
        size = board.size
        
        # 1. 检查是否能赢
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    # 直接修改board.board数组
                    original_value = board.board[x][y]
                    board.board[x][y] = self.player
                    if self.check_win(board, x, y, self.player):
                        moves.append({'x': x, 'y': y, 'score': 100000})
                    # 恢复原始值
                    board.board[x][y] = original_value
        if moves:
            return moves
        
        # 2. 阻止对手获胜
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    # 直接修改board.board数组
                    original_value = board.board[x][y]
                    board.board[x][y] = self.opponent
                    if self.check_win(board, x, y, self.opponent):
                        moves.append({'x': x, 'y': y, 'score': 90000})
                    # 恢复原始值
                    board.board[x][y] = original_value
        if moves:
            return moves
        
        # 3. 评估所有空位
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    score = self.evaluate_position(board, x, y)
                    if score > 0:
                        moves.append({'x': x, 'y': y, 'score': score})
        
        # 按分数排序
        moves.sort(key=lambda x: x['score'], reverse=True)
        
        # 根据难度调整选择策略
        best_moves = moves
        if self.difficulty == 'easy' and len(moves) > 3:
            # 简单模式：随机选择前3个中的一个
            random_index = random.randint(0, 2)
            best_moves = [moves[random_index]]
        elif self.difficulty == 'medium' and len(moves) > 2:
            # 中等模式：随机选择前2个中的一个
            random_index = random.randint(0, 1)
            best_moves = [moves[random_index]]
        
        # 如果没有好的走法，随机选择
        if not best_moves:
            empty_cells = []
            for x in range(size):
                for y in range(size):
                    if board.get(x, y) == 0:
                        empty_cells.append({'x': x, 'y': y, 'score': 0})
            if empty_cells:
                random_index = random.randint(0, len(empty_cells) - 1)
                return [empty_cells[random_index]]
        
        return best_moves
    

    
    def evaluate_position(self, board, x, y):
        score = 0
        size = board.size
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for (dx, dy) in directions:
            # 评估AI的得分
            ai_count = 0
            ai_open = 0
            
            for dir in [-1, 1]:
                nx = x + dx * dir
                ny = y + dy * dir
                count = 0
                open = True
                
                while self.is_valid(nx, ny, size) and count < 5:
                    if board.get(nx, ny) == self.player:
                        count += 1
                    elif board.get(nx, ny) == 0:
                        break
                    else:
                        open = False
                        break
                    nx += dx * dir
                    ny += dy * dir
                
                ai_count += count
                if open:
                    ai_open += 1
            
            # 评估对手的威胁
            opponent_count = 0
            opponent_open = 0
            
            for dir in [-1, 1]:
                nx = x + dx * dir
                ny = y + dy * dir
                count = 0
                open = True
                
                while self.is_valid(nx, ny, size) and count < 5:
                    if board.get(nx, ny) == self.opponent:
                        count += 1
                    elif board.get(nx, ny) == 0:
                        break
                    else:
                        open = False
                        break
                    nx += dx * dir
                    ny += dy * dir
                
                opponent_count += count
                if open:
                    opponent_open += 1
            
            # 根据难度调整计分
            if self.difficulty == 'hard':
                # 困难模式：更重视进攻
                if ai_count >= 4:
                    score += 12000
                elif ai_count == 3:
                    score += 1500
                elif ai_count == 2:
                    score += 150
                elif ai_count == 1:
                    score += 20
                
                if opponent_count >= 4:
                    score += 8000
                elif opponent_count == 3:
                    score += 800
                elif opponent_count == 2:
                    score += 100
            elif self.difficulty == 'medium':
                # 中等模式：平衡进攻和防守
                if ai_count >= 4:
                    score += 10000
                elif ai_count == 3:
                    score += 1000
                elif ai_count == 2:
                    score += 100
                elif ai_count == 1:
                    score += 10
                
                if opponent_count >= 4:
                    score += 6000
                elif opponent_count == 3:
                    score += 600
                elif opponent_count == 2:
                    score += 60
            else:
                # 简单模式：更重视防守
                if ai_count >= 4:
                    score += 8000
                elif ai_count == 3:
                    score += 800
                elif ai_count == 2:
                    score += 80
                elif ai_count == 1:
                    score += 8
                
                if opponent_count >= 4:
                    score += 10000
                elif opponent_count == 3:
                    score += 1000
                elif opponent_count == 2:
                    score += 100
        
        # 优先选择靠近已有棋子的位置
        near_pieces = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if self.is_valid(nx, ny, board.size) and board.get(nx, ny) != 0:
                    near_pieces += 1
        score += near_pieces * 5
        
        # 中心位置加分
        center_x = board.size // 2
        center_y = board.size // 2
        dist_to_center = abs(x - center_x) + abs(y - center_y)
        score += max(0, 20 - dist_to_center)
        
        return score
    

    
    def check_win(self, board, x, y, player):
        size = board.size
        
        for dir in range(4):
            dx = self.DX[dir]
            dy = self.DY[dir]
            count = 1
            
            # 正方向
            nx = x + dx
            ny = y + dy
            while self.is_valid(nx, ny, size) and board.get(nx, ny) == player:
                count += 1
                nx += dx
                ny += dy
            
            # 反方向
            nx = x - dx
            ny = y - dy
            while self.is_valid(nx, ny, size) and board.get(nx, ny) == player:
                count += 1
                nx -= dx
                ny -= dy
            
            if count >= self.WIN_COUNT:
                return True
        
        return False
    
    def is_valid(self, x, y, size):
        return x >= 0 and x < size and y >= 0 and y < size
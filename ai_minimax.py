import time
import math
import random
from board import Board

class MinimaxAI:
    def __init__(self, player=2, difficulty='hard'):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.difficulty = difficulty
        self.BOARD_SIZE = 19
        self.WIN_COUNT = 6
        
        # 方向数组：右、下、右下、左下
        self.DX = [1, 0, 1, -1]
        self.DY = [0, 1, 1, 1]
        
        # 根据难度设置搜索深度
        self.depth_map = {
            'easy': 2,
            'medium': 3,
            'hard': 4
        }
        self.max_depth = self.depth_map.get(difficulty, 4)
    
    def get_best_move(self, board):
        # 寻找最佳走法
        best_score = -math.inf
        best_move = None
        
        # 生成候选走法
        moves = self.generate_moves(board)
        if not moves:
            # 尝试随机选择一个空位
            size = board.size
            for x in range(size):
                for y in range(size):
                    if board.get(x, y) == 0:
                        return (x, y)
            return None
        
        # 对每个候选走法进行评估
        for move in moves:
            x, y = move['x'], move['y']
            
            # 临时落子
            original_value = board.board[x][y]
            board.board[x][y] = self.player
            
            # 使用Minimax算法评估
            score = self.minimax(board, self.max_depth - 1, -math.inf, math.inf, False)
            
            # 恢复棋盘
            board.board[x][y] = original_value
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # 检查游戏是否结束
        winner = self.check_winner(board)
        if winner == self.player:
            return 100000 + depth  # 加分以优先选择更快的胜利
        elif winner == self.opponent:
            return -100000 - depth  # 减分以优先避免更快的失败
        elif depth == 0 or self.is_board_full(board):
            return self.evaluate(board)
        
        if is_maximizing:
            max_score = -math.inf
            moves = self.generate_moves(board)
            for move in moves:
                x, y = move['x'], move['y']
                
                # 临时落子
                original_value = board.board[x][y]
                board.board[x][y] = self.player
                
                # 递归评估
                score = self.minimax(board, depth - 1, alpha, beta, False)
                
                # 恢复棋盘
                board.board[x][y] = original_value
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Alpha-Beta剪枝
            return max_score
        else:
            min_score = math.inf
            moves = self.generate_moves(board)
            for move in moves:
                x, y = move['x'], move['y']
                
                # 临时落子
                original_value = board.board[x][y]
                board.board[x][y] = self.opponent
                
                # 递归评估
                score = self.minimax(board, depth - 1, alpha, beta, True)
                
                # 恢复棋盘
                board.board[x][y] = original_value
                
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha-Beta剪枝
            return min_score
    
    def generate_moves(self, board):
        """生成候选走法"""
        moves = []
        size = board.size
        
        # 1. 检查是否能赢
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    # 临时落子
                    original_value = board.board[x][y]
                    board.board[x][y] = self.player
                    if self.check_win(board, x, y, self.player):
                        moves.append({'x': x, 'y': y, 'score': 100000})
                    # 恢复棋盘
                    board.board[x][y] = original_value
        if moves:
            return moves
        
        # 2. 阻止对手获胜
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    # 临时落子
                    original_value = board.board[x][y]
                    board.board[x][y] = self.opponent
                    if self.check_win(board, x, y, self.opponent):
                        moves.append({'x': x, 'y': y, 'score': 90000})
                    # 恢复棋盘
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
        
        # 按分数排序，只保留前10个最佳走法以提高搜索效率
        moves.sort(key=lambda x: x['score'], reverse=True)
        return moves[:10] if len(moves) > 10 else moves
    
    def evaluate(self, board):
        """评估整个棋盘"""
        score = 0
        size = board.size
        
        # 评估每个空位
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    score += self.evaluate_position(board, x, y)
        
        return score
    
    def evaluate_position(self, board, x, y):
        """评估单个位置的价值"""
        score = 0
        size = board.size
        
        for dir in range(4):
            dx = self.DX[dir]
            dy = self.DY[dir]
            
            # 评估AI的得分
            ai_pattern = self.evaluate_pattern(board, x, y, self.player, dx, dy)
            score += ai_pattern
            
            # 评估对手的威胁
            opponent_pattern = self.evaluate_pattern(board, x, y, self.opponent, dx, dy)
            score -= opponent_pattern
        
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
        score += near_pieces * 10
        
        # 中心位置加分
        center_x = board.size // 2
        center_y = board.size // 2
        dist_to_center = abs(x - center_x) + abs(y - center_y)
        score += max(0, 30 - dist_to_center * 2)
        
        return score
    
    def evaluate_pattern(self, board, x, y, player, dx, dy):
        """评估指定方向的棋型"""
        size = board.size
        
        # 计算正方向的连续棋子数和开口
        forward_count = 0
        forward_open = False
        nx, ny = x + dx, y + dy
        while self.is_valid(nx, ny, size) and board.get(nx, ny) == player:
            forward_count += 1
            nx += dx
            ny += dy
        if self.is_valid(nx, ny, size) and board.get(nx, ny) == 0:
            forward_open = True
        
        # 计算反方向的连续棋子数和开口
        backward_count = 0
        backward_open = False
        nx, ny = x - dx, y - dy
        while self.is_valid(nx, ny, size) and board.get(nx, ny) == player:
            backward_count += 1
            nx -= dx
            ny -= dy
        if self.is_valid(nx, ny, size) and board.get(nx, ny) == 0:
            backward_open = True
        
        # 总连续棋子数
        total_count = forward_count + backward_count
        
        # 开口数量
        open_count = 0
        if forward_open:
            open_count += 1
        if backward_open:
            open_count += 1
        
        # 根据棋型评分
        if total_count >= 5:
            return 100000  # 即将获胜
        elif total_count == 4:
            if open_count == 2:
                return 10000  # 活四
            elif open_count == 1:
                return 5000   # 冲四
        elif total_count == 3:
            if open_count == 2:
                return 1000   # 活三
            elif open_count == 1:
                return 500    # 冲三
        elif total_count == 2:
            if open_count == 2:
                return 100    # 活二
            elif open_count == 1:
                return 50     # 冲二
        elif total_count == 1:
            if open_count == 2:
                return 20     # 活一
        
        return 0
    
    def check_winner(self, board):
        """检查当前棋盘的获胜者"""
        size = board.size
        
        # 检查所有已落子的位置
        for x in range(size):
            for y in range(size):
                player = board.get(x, y)
                if player != 0 and self.check_win(board, x, y, player):
                    return player
        
        return 0  # 没有获胜者
    
    def check_win(self, board, x, y, player):
        """检查指定位置的玩家是否获胜"""
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
    
    def is_board_full(self, board):
        """检查棋盘是否已满"""
        size = board.size
        for x in range(size):
            for y in range(size):
                if board.get(x, y) == 0:
                    return False
        return True
    
    def is_valid(self, x, y, size):
        """检查坐标是否有效"""
        return x >= 0 and x < size and y >= 0 and y < size

import time
import math
from board import Board

class AI:
    def __init__(self, player=2, max_depth=5, time_limit=5.0):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.transposition_table = {}
        self.history_heuristic = {}
        self.killer_moves = {}
    
    def get_best_move(self, board):
        start_time = time.time()
        best_move = None
        best_score = -math.inf
        
        # 迭代加深搜索
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break
            
            current_best_move, current_best_score = self.iterative_deepening(board, depth, start_time)
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_move = current_best_move
        
        return best_move
    
    def iterative_deepening(self, board, depth, start_time):
        best_move = None
        best_score = -math.inf
        
        # 生成所有合法走法
        moves = self.generate_moves(board)
        
        # 按历史启发值排序走法
        moves.sort(key=lambda move: self.history_heuristic.get(move, 0), reverse=True)
        
        for move in moves:
            if time.time() - start_time > self.time_limit:
                break
            
            # 复制棋盘并落子
            new_board = self.copy_board(board)
            new_board.set(move[0], move[1], self.player)
            
            # 计算分数
            score = -self.alpha_beta(new_board, depth-1, -math.inf, math.inf, self.opponent, start_time)
            
            if score > best_score:
                best_score = score
                best_move = move
                
                # 更新历史启发值
                self.history_heuristic[move] = self.history_heuristic.get(move, 0) + depth * depth
        
        return best_move, best_score
    
    def alpha_beta(self, board, depth, alpha, beta, current_player, start_time):
        # 检查时间限制
        if time.time() - start_time > self.time_limit:
            return 0
        
        # 检查是否达到终止条件
        if depth == 0 or self.is_game_over(board):
            return self.evaluate(board)
        
        # 检查置换表
        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'exact':
                    return entry['score']
                elif entry['flag'] == 'lower_bound':
                    alpha = max(alpha, entry['score'])
                elif entry['flag'] == 'upper_bound':
                    beta = min(beta, entry['score'])
                
                if alpha >= beta:
                    return entry['score']
        
        # 生成所有合法走法
        moves = self.generate_moves(board)
        
        # 按历史启发值排序走法
        moves.sort(key=lambda move: self.history_heuristic.get(move, 0), reverse=True)
        
        best_score = -math.inf if current_player == self.player else math.inf
        best_move = None
        
        for move in moves:
            # 复制棋盘并落子
            new_board = self.copy_board(board)
            new_board.set(move[0], move[1], current_player)
            
            # 递归搜索
            if current_player == self.player:
                score = -self.alpha_beta(new_board, depth-1, -beta, -alpha, self.opponent, start_time)
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
            else:
                score = -self.alpha_beta(new_board, depth-1, -beta, -alpha, self.player, start_time)
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, score)
            
            # 剪枝
            if alpha >= beta:
                # 更新杀手启发
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].append(move)
                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop(0)
                break
        
        # 更新置换表
        flag = 'exact'
        if best_score <= alpha:
            flag = 'upper_bound'
        elif best_score >= beta:
            flag = 'lower_bound'
        
        self.transposition_table[board_hash] = {
            'score': best_score,
            'depth': depth,
            'flag': flag,
            'best_move': best_move
        }
        
        return best_score
    
    def generate_moves(self, board):
        moves = []
        size = board.size
        
        # 优先考虑最近落子的周围位置
        recent_positions = []
        if board.history:
            # 只考虑最近5步的位置
            for i in range(max(0, len(board.history) - 5), len(board.history)):
                x, y, _ = board.history[i]
                recent_positions.append((x, y))
        else:
            # 初始位置，考虑中心区域
            center = size // 2
            for i in range(center-2, center+3):
                for j in range(center-2, center+3):
                    if 0 <= i < size and 0 <= j < size:
                        recent_positions.append((i, j))
        
        # 从最近位置生成走法
        for (x, y) in recent_positions:
            # 检查周围1-2层的位置
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = x + di, y + dj
                    if 0 <= ni < size and 0 <= nj < size and board.get(ni, nj) == 0:
                        if (ni, nj) not in moves:
                            moves.append((ni, nj))
        
        # 如果没有周围位置，返回中心位置
        if not moves:
            center = size // 2
            moves.append((center, center))
        
        return moves
    
    def evaluate(self, board):
        # 评估函数
        score = 0
        size = board.size
        
        # 检查胜负
        for i in range(size):
            for j in range(size):
                if board.get(i, j) == self.player:
                    if self.check_win(board, i, j, self.player):
                        return 1000000
                elif board.get(i, j) == self.opponent:
                    if self.check_win(board, i, j, self.opponent):
                        return -1000000
        
        # 只评估有棋子的位置及其周围
        evaluated_positions = set()
        for i in range(size):
            for j in range(size):
                if board.get(i, j) != 0:
                    # 评估当前位置
                    if (i, j) not in evaluated_positions:
                        if board.get(i, j) == self.player:
                            score += self.evaluate_position(board, i, j, self.player)
                        else:
                            score -= self.evaluate_position(board, i, j, self.opponent)
                        evaluated_positions.add((i, j))
                    
                    # 评估周围位置
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in evaluated_positions:
                                if board.get(ni, nj) == self.player:
                                    score += self.evaluate_position(board, ni, nj, self.player) * 0.5
                                elif board.get(ni, nj) == self.opponent:
                                    score -= self.evaluate_position(board, ni, nj, self.opponent) * 0.5
                                evaluated_positions.add((ni, nj))
        
        # 评估中心控制（只考虑有棋子的位置）
        center = size // 2
        for i in range(size):
            for j in range(size):
                if board.get(i, j) != 0:
                    distance = abs(i - center) + abs(j - center)
                    if board.get(i, j) == self.player:
                        score += (10 - distance) * 5
                    else:
                        score -= (10 - distance) * 5
        
        return score
    
    def evaluate_position(self, board, x, y, player):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # 计算当前方向的连续棋子数
            count = 1
            blocked = [False, False]
            
            # 向正方向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < board.size and 0 <= ny < board.size:
                if board.get(nx, ny) == player:
                    count += 1
                    nx, ny = nx + dx, ny + dy
                else:
                    blocked[0] = board.get(nx, ny) != 0
                    break
            
            # 向反方向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < board.size and 0 <= ny < board.size:
                if board.get(nx, ny) == player:
                    count += 1
                    nx, ny = nx - dx, ny - dy
                else:
                    blocked[1] = board.get(nx, ny) != 0
                    break
            
            # 根据棋型计算分数
            if count >= 6:
                score += 1000000
            elif count == 5:
                if not any(blocked):
                    score += 100000
                elif blocked[0] and blocked[1]:
                    score += 1000
                else:
                    score += 10000
            elif count == 4:
                if not any(blocked):
                    score += 10000
                elif blocked[0] and blocked[1]:
                    score += 100
                else:
                    score += 1000
            elif count == 3:
                if not any(blocked):
                    score += 1000
                elif blocked[0] and blocked[1]:
                    score += 10
                else:
                    score += 100
            elif count == 2:
                if not any(blocked):
                    score += 100
                elif blocked[0] and blocked[1]:
                    score += 1
                else:
                    score += 10
        
        return score
    
    def check_win(self, board, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 向正方向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
                count += 1
                nx, ny = nx + dx, ny + dy
            
            # 向反方向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
                count += 1
                nx, ny = nx - dx, ny - dy
            
            if count >= 6:
                return True
        
        return False
    
    def is_game_over(self, board):
        # 检查是否有一方获胜
        for i in range(board.size):
            for j in range(board.size):
                if board.get(i, j) != 0:
                    if self.check_win(board, i, j, board.get(i, j)):
                        return True
        
        # 检查是否平局
        return board.is_full()
    
    def copy_board(self, board):
        new_board = Board(board.size)
        for i in range(board.size):
            for j in range(board.size):
                new_board.board[i][j] = board.board[i][j]
        new_board.history = board.history.copy()
        return new_board
    
    def hash_board(self, board):
        # 简单的棋盘哈希函数
        hash_value = 0
        for i in range(board.size):
            for j in range(board.size):
                hash_value = hash_value * 3 + board.board[i][j]
        return hash_value
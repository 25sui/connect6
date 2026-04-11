import time
import math
from board import Board

class AdvancedAI:
    def __init__(self, player=2, max_depth=8, time_limit=5.0, difficulty='hard'):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.difficulty = difficulty
        self.transposition_table = {}
        self.history_heuristic = {}
        self.killer_moves = {}
        
        # 棋型评分
        self.SCORE_SIX = 100000000      # 连六（必胜）
        self.SCORE_LIVE_FIVE = 50000000  # 活五
        self.SCORE_DEAD_FIVE = 10000000  # 死五
        self.SCORE_LIVE_FOUR = 5000000   # 活四
        self.SCORE_DEAD_FOUR = 500000    # 死四
        self.SCORE_LIVE_THREE = 100000   # 活三
        self.SCORE_DEAD_THREE = 10000    # 死三
        self.SCORE_LIVE_TWO = 1000       # 活二
        self.SCORE_DEAD_TWO = 100        # 死二
        self.SCORE_LIVE_ONE = 10         # 活一
        
        # 方向数组：右、下、右下、左下
        self.DX = [1, 0, 1, -1]
        self.DY = [0, 1, 1, 1]
    
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
        considered = set()
        for (x, y) in recent_positions:
            # 检查周围2格范围内的空位
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    ni, nj = x + dx, y + dy
                    if 0 <= ni < size and 0 <= nj < size and board.get(ni, nj) == 0 and (ni, nj) not in considered:
                        considered.add((ni, nj))
                        moves.append((ni, nj))
        
        # 如果没有周围位置，返回中心位置
        if not moves:
            center = size // 2
            moves.append((center, center))
        
        # 评估并排序走法
        scored_moves = []
        for move in moves:
            score = self.evaluate_position(board, move[0], move[1], self.player)
            # 考虑防守价值
            score += self.evaluate_position(board, move[0], move[1], self.opponent) * 0.8
            # 位置价值
            center = size // 2
            dist_to_center = abs(move[0] - center) + abs(move[1] - center)
            score += max(0, 20 - dist_to_center) * 5
            # 靠近已有棋子的位置加分
            near_pieces = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx = move[0] + dx
                    ny = move[1] + dy
                    if 0 <= nx < size and 0 <= ny < size and board.get(nx, ny) != 0:
                        near_pieces += 1
            score += near_pieces * 10
            scored_moves.append((move, score))
        
        # 按分数排序
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # 根据难度调整返回的走法数量
        if self.difficulty == 'easy':
            # 简单模式：减少搜索深度，返回较少走法
            return [move for move, _ in scored_moves[:10]]
        elif self.difficulty == 'medium':
            # 中等模式：返回中等数量走法
            return [move for move, _ in scored_moves[:15]]
        else:
            # 困难模式：返回较多走法
            return [move for move, _ in scored_moves[:20]]
    
    def evaluate(self, board):
        # 评估函数
        score = 0
        size = board.size
        
        # 检查胜负
        for i in range(size):
            for j in range(size):
                if board.get(i, j) == self.player:
                    if self.check_win(board, i, j, self.player):
                        return self.SCORE_SIX
                elif board.get(i, j) == self.opponent:
                    if self.check_win(board, i, j, self.opponent):
                        return -self.SCORE_SIX
        
        # 1. 检查是否能直接获胜
        for i in range(size):
            for j in range(size):
                if board.get(i, j) == 0:
                    # 模拟下子
                    board.set(i, j, self.player)
                    if self.check_win(board, i, j, self.player):
                        board.set(i, j, 0)
                        return self.SCORE_LIVE_FIVE * 2  # 直接获胜的价值
                    board.set(i, j, 0)
        
        # 2. 检查是否需要阻止对手获胜
        for i in range(size):
            for j in range(size):
                if board.get(i, j) == 0:
                    # 模拟对手下子
                    board.set(i, j, self.opponent)
                    if self.check_win(board, i, j, self.opponent):
                        board.set(i, j, 0)
                        return self.SCORE_LIVE_FIVE  # 必须阻止对手获胜
                    board.set(i, j, 0)
        
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
                        score += (20 - distance) * 15
                    else:
                        score -= (20 - distance) * 15
        
        # 棋盘阶段评估
        move_count = len(board.history)
        if move_count < 8:
            # 开局阶段，位置价值更重要
            score = int(score * 1.2)
        elif move_count > 40:
            # 中局阶段，进攻价值更重要
            score = int(score * 1.3)
        elif move_count > 60:
            # 残局阶段，必须全力以赴
            score = int(score * 1.5)
        
        # 棋子密度评估
        piece_count = 0
        for i in range(size):
            for j in range(size):
                if board.get(i, j) != 0:
                    piece_count += 1
        
        # 密集局面，进攻价值提升
        if piece_count > 40:
            score = int(score * 1.2)
        
        # 空间优势评估
        my_mobility = 0
        opponent_mobility = 0
        for i in range(size):
            for j in range(size):
                if board.get(i, j) == 0:
                    # 计算周围棋子数量
                    my_around = 0
                    opponent_around = 0
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx = i + dx
                            ny = j + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                if board.get(nx, ny) == self.player:
                                    my_around += 1
                                elif board.get(nx, ny) == self.opponent:
                                    opponent_around += 1
                    if my_around > opponent_around:
                        my_mobility += 1
                    elif opponent_around > my_around:
                        opponent_mobility += 1
        
        # 机动性优势
        score += (my_mobility - opponent_mobility) * 100
        
        # 基于难度的评估调整
        if self.difficulty == 'easy':
            # 简单模式：降低整体评估值，使AI更保守
            score = int(score * 0.7)
        elif self.difficulty == 'medium':
            # 中等模式：轻微降低评估值
            score = int(score * 0.85)
        
        return score
    
    def evaluate_position(self, board, x, y, player):
        score = 0
        
        for dir in range(4):
            # 只计算从该位置开始的棋型，避免重复计算
            nx = x - self.DX[dir]
            ny = y - self.DY[dir]
            if 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
                continue  # 不是起点，跳过
            
            pattern_info = self.recognize_pattern(board, x, y, dir, player)
            score += pattern_info['score']
            
            # 额外的战术价值评估
            if pattern_info['type'] == 'LIVE_FOUR':
                score += 80000  # 活四的价值更高
            elif pattern_info['type'] == 'DEAD_FOUR':
                score += 40000  # 死四也有较高价值
            elif pattern_info['type'] == 'LIVE_THREE':
                score += 20000  # 活三的价值
            elif pattern_info['type'] == 'DEAD_THREE':
                score += 10000  # 死三的价值
            elif pattern_info['type'] == 'LIVE_TWO':
                score += 5000  # 活二的价值
        
        # 基于难度的评分调整
        if self.difficulty == 'easy':
            # 简单模式：更重视防守，降低进攻价值
            score = int(score * 0.8)
        elif self.difficulty == 'medium':
            # 中等模式：平衡进攻和防守
            score = int(score * 0.9)
        
        return score
    
    def recognize_pattern(self, board, x, y, dir, player):
        if board.get(x, y) != player:
            return {'type': 'NONE', 'length': 0, 'open': False, 'score': 0}
        
        count = 1
        left_open = False
        right_open = False
        left_block = False
        right_block = False
        
        # 正方向
        nx = x + self.DX[dir]
        ny = y + self.DY[dir]
        while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
            count += 1
            nx += self.DX[dir]
            ny += self.DY[dir]
        
        if 0 <= nx < board.size and 0 <= ny < board.size:
            if board.get(nx, ny) == 0:
                right_open = True
            else:
                right_block = True
        
        # 反方向
        nx = x - self.DX[dir]
        ny = y - self.DY[dir]
        while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
            count += 1
            nx -= self.DX[dir]
            ny -= self.DY[dir]
        
        if 0 <= nx < board.size and 0 <= ny < board.size:
            if board.get(nx, ny) == 0:
                left_open = True
            else:
                left_block = True
        
        is_live = left_open and right_open
        pattern_type = 'NONE'
        score = 0
        
        if count >= 6:
            pattern_type = 'SIX'
            score = self.SCORE_SIX
        elif count == 5:
            pattern_type = 'LIVE_FIVE' if is_live else 'DEAD_FIVE'
            score = self.SCORE_LIVE_FIVE if is_live else self.SCORE_DEAD_FIVE
        elif count == 4:
            pattern_type = 'LIVE_FOUR' if is_live else 'DEAD_FOUR'
            score = self.SCORE_LIVE_FOUR if is_live else self.SCORE_DEAD_FOUR
        elif count == 3:
            pattern_type = 'LIVE_THREE' if is_live else 'DEAD_THREE'
            score = self.SCORE_LIVE_THREE if is_live else self.SCORE_DEAD_THREE
        elif count == 2:
            pattern_type = 'LIVE_TWO' if is_live else 'DEAD_TWO'
            score = self.SCORE_LIVE_TWO if is_live else self.SCORE_DEAD_TWO
        else:
            pattern_type = 'LIVE_ONE'
            score = self.SCORE_LIVE_ONE
        
        # 增强的棋型评估
        if pattern_type == 'LIVE_THREE':
            # 检查是否有潜在的活四
            has_potential = False
            temp_x, temp_y = x, y
            for i in range(count):
                tx = temp_x + self.DX[dir] * (count)
                ty = temp_y + self.DY[dir] * (count)
                if 0 <= tx < board.size and 0 <= ty < board.size and board.get(tx, ty) == 0:
                    has_potential = True
                    break
                temp_x += self.DX[dir]
                temp_y += self.DY[dir]
            if has_potential:
                score += 10000  # 潜在活四的额外价值
        elif pattern_type == 'DEAD_THREE':
            # 检查是否有潜在的死四
            has_potential = False
            temp_x, temp_y = x, y
            for i in range(count):
                tx = temp_x + self.DX[dir] * (count)
                ty = temp_y + self.DY[dir] * (count)
                if 0 <= tx < board.size and 0 <= ty < board.size and board.get(tx, ty) == 0:
                    has_potential = True
                    break
                temp_x += self.DX[dir]
                temp_y += self.DY[dir]
            if has_potential:
                score += 5000  # 潜在死四的额外价值
        elif pattern_type == 'LIVE_TWO':
            # 检查是否有潜在的活三
            has_potential = False
            temp_x, temp_y = x, y
            for i in range(count):
                tx = temp_x + self.DX[dir] * (count)
                ty = temp_y + self.DY[dir] * (count)
                if 0 <= tx < board.size and 0 <= ty < board.size and board.get(tx, ty) == 0:
                    # 检查是否形成活三
                    can_form_live_three = True
                    temp_tx, temp_ty = tx, ty
                    for j in range(2):
                        temp_tx += self.DX[dir]
                        temp_ty += self.DY[dir]
                        if not (0 <= temp_tx < board.size and 0 <= temp_ty < board.size and board.get(temp_tx, temp_ty) == 0):
                            can_form_live_three = False
                            break
                    if can_form_live_three:
                        has_potential = True
                        break
                temp_x += self.DX[dir]
                temp_y += self.DY[dir]
            if has_potential:
                score += 3000  # 潜在活三的额外价值
        
        # 检查连续多个棋型的组合价值
        combo_score = 0
        temp_count = 1
        temp_x, temp_y = x, y
        while True:
            temp_x += self.DX[dir]
            temp_y += self.DY[dir]
            if not (0 <= temp_x < board.size and 0 <= temp_y < board.size and board.get(temp_x, temp_y) == player):
                break
            temp_count += 1
        
        if temp_count >= 4:
            combo_score += 20000  # 长连的额外价值
        elif temp_count == 3:
            combo_score += 10000  # 三连的额外价值
        
        score += combo_score
        
        return {'type': pattern_type, 'length': count, 'open': is_live, 'score': score}
    
    def check_win(self, board, x, y, player):
        for dir in range(4):
            count = 1
            
            # 正方向
            nx = x + self.DX[dir]
            ny = y + self.DY[dir]
            while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
                count += 1
                nx += self.DX[dir]
                ny += self.DY[dir]
            
            # 反方向
            nx = x - self.DX[dir]
            ny = y - self.DY[dir]
            while 0 <= nx < board.size and 0 <= ny < board.size and board.get(nx, ny) == player:
                count += 1
                nx -= self.DX[dir]
                ny -= self.DY[dir]
            
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
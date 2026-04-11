from board import Board

class Game:
    def __init__(self, size=19):
        self.board = Board(size)
        self.current_player = 1  # 1 for black, 2 for white
        self.move_count = 0
        self.game_over = False
        self.winner = 0
        self.move_history = []  # 历史记录
    
    def reset(self):
        self.board.reset()
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = 0
        self.move_history = []
    
    def make_move(self, x, y):
        if self.game_over:
            return False
        
        if self.board.set(x, y, self.current_player):
            # 记录移动历史
            self.move_history.append({
                'x': x,
                'y': y,
                'player': self.current_player
            })
            
            self.move_count += 1
            
            if self.check_win(x, y, self.current_player):
                self.game_over = True
                self.winner = self.current_player
                return True
            
            if self.board.is_full():
                self.game_over = True
                self.winner = 0  # draw
                return True
            
            # 切换玩家，黑方第一手下一子，之后双方各下两子
            # 严格按照规则：
            # 1. 黑方第一手下1子
            # 2. 之后每方每回合下2子
            if self.move_count == 1:
                # 黑方已经下了1子，第一回合结束，切换到白方
                self.current_player = 2
            elif self.move_count >= 3 and self.move_count % 2 == 1:
                # 当 move_count=3,5,7... 时，一方已经下了2子，回合结束，切换玩家
                self.current_player = 1 if self.current_player == 2 else 2
            
            return True
        return False
    
    def jump_to_move(self, index):
        """跳转到指定步数（从0开始）"""
        if index < 0 or index >= len(self.move_history):
            return False
        
        # 重置棋盘
        self.board.reset()
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = 0
        
        # 重新应用移动到指定步数
        for i in range(index + 1):
            move = self.move_history[i]
            self.board.set(move['x'], move['y'], move['player'])
            self.move_count += 1
            
            # 检查获胜
            if self.check_win(move['x'], move['y'], move['player']):
                self.game_over = True
                self.winner = move['player']
            
            # 切换玩家
            if self.move_count == 1:
                self.current_player = 2
            elif self.move_count >= 3 and self.move_count % 2 == 1:
                self.current_player = 1 if self.current_player == 2 else 2
        
        # 如果没有结束，设置正确的当前玩家
        if not self.game_over:
            if self.move_count == 1:
                self.current_player = 2
            elif self.move_count == 0:
                self.current_player = 1
            else:
                if self.move_count % 2 == 0:
                    self.current_player = 1
                else:
                    self.current_player = 2
        
        return True
    
    def check_win(self, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 向正方向检查
            nx, ny = x + dx, y + dy
            while self.board.get(nx, ny) == player:
                count += 1
                nx, ny = nx + dx, ny + dy
            
            # 向反方向检查
            nx, ny = x - dx, y - dy
            while self.board.get(nx, ny) == player:
                count += 1
                nx, ny = nx - dx, ny - dy
            
            if count >= 6:
                return True
        
        return False
    
    def undo(self):
        if self.board.undo():
            self.move_count -= 1
            
            if self.move_count == 0:
                self.current_player = 1
            elif self.move_count == 1:
                self.current_player = 1
            elif self.move_count == 2:
                # move_count=2: 白方下1子，应该是白方(2)的回合
                self.current_player = 2
            else:
                # 与 make_move 方法的逻辑一致
                # move_count=3: 白方下2子，回合结束，切换到黑方
                # move_count=4: 黑方下1子
                # move_count=5: 黑方下2子，回合结束，切换到白方
                if self.move_count % 2 == 1:
                    self.current_player = 1 if self.current_player == 2 else 2
                else:
                    # 保持当前玩家
                    pass
            
            self.game_over = False
            self.winner = 0
            return True
        return False
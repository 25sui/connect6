class Board:
    def __init__(self, size=19):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.history = []
    
    def reset(self):
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.history = []
    
    def get(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.board[x][y]
        return -1
    
    def set(self, x, y, player):
        if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
            self.board[x][y] = player
            self.history.append((x, y, player))
            return True
        return False
    
    def undo(self):
        if self.history:
            x, y, _ = self.history.pop()
            self.board[x][y] = 0
            return True
        return False
    
    def is_full(self):
        for row in self.board:
            if 0 in row:
                return False
        return True
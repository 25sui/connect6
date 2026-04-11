import tkinter as tk
from tkinter import messagebox, filedialog
import json
from datetime import datetime
from game import Game
from ai_minimax import MinimaxAI
from ai_webui import WebUIAI

class Connect6UI:
    def __init__(self, root):
        self.root = root
        self.root.title("六子棋")
        self.root.resizable(False, False)
        
        self.game = Game()
        self.cell_size = 30
        self.board_size = self.game.board.size
        self.canvas_width = self.cell_size * (self.board_size + 1)
        self.canvas_height = self.cell_size * (self.board_size + 1)
        
        # 游戏模式：0-人人对战，1-人机对战
        self.game_mode = 0
        self.ai = None
        
        # 创建主框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)
        
        # 创建棋盘画布
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="#E6B422")
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # 创建控制面板
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, padx=10)
        
        # 创建按钮
        self.new_game_btn = tk.Button(self.control_frame, text="新游戏", command=self.new_game, width=10)
        self.new_game_btn.pack(pady=5)
        
        self.undo_btn = tk.Button(self.control_frame, text="悔棋", command=self.undo, width=10)
        self.undo_btn.pack(pady=5)
        
        self.rules_btn = tk.Button(self.control_frame, text="游戏规则", command=self.show_rules, width=10)
        self.rules_btn.pack(pady=5)
        
        # 创建游戏模式选择按钮
        self.mode_var = tk.StringVar()
        self.mode_var.set("人人对战")
        self.mode_frame = tk.Frame(self.control_frame)
        self.mode_frame.pack(pady=5)
        
        self.pvp_btn = tk.Radiobutton(self.mode_frame, text="人人对战", variable=self.mode_var, value="人人对战", command=self.change_mode)
        self.pvp_btn.pack(side=tk.LEFT)
        
        self.pve_btn = tk.Radiobutton(self.mode_frame, text="人机对战", variable=self.mode_var, value="人机对战", command=self.change_mode)
        self.pve_btn.pack(side=tk.LEFT)
        
        # 创建AI类型选择按钮
        self.ai_type_var = tk.StringVar()
        self.ai_type_var.set("MinimaxAI")
        self.ai_type_frame = tk.Frame(self.control_frame)
        self.ai_type_frame.pack(pady=5)
        
        self.minimax_btn = tk.Radiobutton(self.ai_type_frame, text="MinimaxAI", variable=self.ai_type_var, value="MinimaxAI", command=self.change_ai_type)
        self.minimax_btn.pack(side=tk.LEFT)
        
        self.webui_btn = tk.Radiobutton(self.ai_type_frame, text="WebUIAI", variable=self.ai_type_var, value="WebUIAI", command=self.change_ai_type)
        self.webui_btn.pack(side=tk.LEFT)
        
        # 创建AI难度选择按钮
        self.difficulty_var = tk.StringVar()
        self.difficulty_var.set("困难")
        self.difficulty_frame = tk.Frame(self.control_frame)
        self.difficulty_frame.pack(pady=5)
        
        self.easy_btn = tk.Radiobutton(self.difficulty_frame, text="简单", variable=self.difficulty_var, value="简单", command=self.change_difficulty)
        self.easy_btn.pack(side=tk.LEFT)
        
        self.medium_btn = tk.Radiobutton(self.difficulty_frame, text="中等", variable=self.difficulty_var, value="中等", command=self.change_difficulty)
        self.medium_btn.pack(side=tk.LEFT)
        
        self.hard_btn = tk.Radiobutton(self.difficulty_frame, text="困难", variable=self.difficulty_var, value="困难", command=self.change_difficulty)
        self.hard_btn.pack(side=tk.LEFT)
        
        # 创建先手选择按钮
        self.first_player_var = tk.StringVar()
        self.first_player_var.set("人类先手")
        self.first_player_frame = tk.Frame(self.control_frame)
        self.first_player_frame.pack(pady=5)
        
        self.human_first_btn = tk.Radiobutton(self.first_player_frame, text="人类先手", variable=self.first_player_var, value="人类先手", command=self.change_first_player)
        self.human_first_btn.pack(side=tk.LEFT)
        
        self.ai_first_btn = tk.Radiobutton(self.first_player_frame, text="AI先手", variable=self.first_player_var, value="AI先手", command=self.change_first_player)
        self.ai_first_btn.pack(side=tk.LEFT)
        
        # 创建AI先走按钮
        self.ai_first_move_btn = tk.Button(self.control_frame, text="AI先走", command=self.ai_first_move, width=10)
        self.ai_first_move_btn.pack(pady=5)
        self.ai_first_move_btn.config(state=tk.DISABLED)
        
        # 创建状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("黑方先行")
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var, font=("Arial", 12), width=15)
        self.status_label.pack(pady=10)
        
        # 创建待下棋子数标签
        self.pending_var = tk.StringVar()
        self.pending_label = tk.Label(self.control_frame, textvariable=self.pending_var, font=("Arial", 12), width=15)
        self.pending_label.pack(pady=10)
        
        # 创建导出和导入棋谱按钮
        self.export_import_frame = tk.Frame(self.control_frame)
        self.export_import_frame.pack(pady=5)
        
        self.export_btn = tk.Button(self.export_import_frame, text="导出棋谱", command=self.export_game, width=10)
        self.export_btn.pack(side=tk.LEFT, padx=2)
        
        self.import_btn = tk.Button(self.export_import_frame, text="导入棋谱", command=self.import_game, width=10)
        self.import_btn.pack(side=tk.LEFT, padx=2)
        
        # 创建计时区域
        self.timer_frame = tk.Frame(self.control_frame)
        self.timer_frame.pack(pady=10)
        
        self.timer_label = tk.Label(self.timer_frame, text="时间限制", font=("Arial", 12, "bold"))
        self.timer_label.pack()
        
        self.timer_display_frame = tk.Frame(self.timer_frame)
        self.timer_display_frame.pack()
        
        self.black_timer_var = tk.StringVar()
        self.black_timer_var.set("10:00")
        self.black_timer_label = tk.Label(self.timer_display_frame, text="黑方:", font=("Arial", 12))
        self.black_timer_label.pack(side=tk.LEFT, padx=5)
        self.black_timer_display = tk.Label(self.timer_display_frame, textvariable=self.black_timer_var, font=("Arial", 12, "bold"))
        self.black_timer_display.pack(side=tk.LEFT, padx=5)
        
        self.white_timer_var = tk.StringVar()
        self.white_timer_var.set("10:00")
        self.white_timer_label = tk.Label(self.timer_display_frame, text="白方:", font=("Arial", 12))
        self.white_timer_label.pack(side=tk.LEFT, padx=5)
        self.white_timer_display = tk.Label(self.timer_display_frame, textvariable=self.white_timer_var, font=("Arial", 12, "bold"))
        self.white_timer_display.pack(side=tk.LEFT, padx=5)
        
        # 初始化计时器变量
        self.black_time = 600  # 黑方时间（秒）
        self.white_time = 600  # 白方时间（秒）
        self.timer_interval = None  # 计时器间隔
        self.is_timer_running = False  # 计时器是否运行
        
        # 创建历史记录区域
        self.history_label = tk.Label(self.control_frame, text="历史记录", font=("Arial", 12, "bold"))
        self.history_label.pack(pady=(10, 0))
        
        self.history_frame = tk.Frame(self.control_frame)
        self.history_frame.pack(pady=5)
        
        self.history_scrollbar = tk.Scrollbar(self.history_frame)
        self.history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(self.history_frame, yscrollcommand=self.history_scrollbar.set, height=10, width=25)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        self.history_scrollbar.config(command=self.history_listbox.yview)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 绘制棋盘
        self.draw_board()
        
        # 记录当前玩家需要下的棋子数
        self.pending_moves = 0
        self.update_pending_moves()
        
        # 初始化AI
        if self.ai_type_var.get() == "MinimaxAI":
            self.ai = MinimaxAI(player=2, difficulty='hard')
        else:
            self.ai = WebUIAI(player=2, difficulty='hard')
    
    def draw_board(self):
        # 绘制棋盘网格
        for i in range(self.board_size):
            x = self.cell_size * (i + 1)
            self.canvas.create_line(x, self.cell_size, x, self.canvas_height - self.cell_size)
            self.canvas.create_line(self.cell_size, x, self.canvas_width - self.cell_size, x)
        
        # 绘制星位点
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for x, y in star_points:
            pos_x = self.cell_size * (x + 1)
            pos_y = self.cell_size * (y + 1)
            self.canvas.create_oval(pos_x - 4, pos_y - 4, pos_x + 4, pos_y + 4, fill="black")
    
    def draw_pieces(self):
        # 清空画布上的棋子
        self.canvas.delete("piece")
        
        # 绘制所有棋子
        for x in range(self.board_size):
            for y in range(self.board_size):
                piece = self.game.board.get(x, y)
                if piece != 0:
                    pos_x = self.cell_size * (x + 1)
                    pos_y = self.cell_size * (y + 1)
                    color = "black" if piece == 1 else "white"
                    outline = "black" if piece == 2 else "black"
                    self.canvas.create_oval(pos_x - 12, pos_y - 12, pos_x + 12, pos_y + 12, 
                                           fill=color, outline=outline, tags="piece")
    
    def on_click(self, event):
        # 计算点击位置对应的棋盘坐标
        x = int((event.x - self.cell_size / 2) // self.cell_size)
        y = int((event.y - self.cell_size / 2) // self.cell_size)
        
        # 检查坐标是否在棋盘范围内
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            # 检查是否是人类的回合
            if self.game_mode == 1:
                if self.first_player_var.get() == "人类先手":
                    # 人类是黑方(1)
                    if self.game.current_player != 1:
                        return
                else:
                    # 人类是白方(2)
                    if self.game.current_player != 2:
                        return
            
            # 尝试落子
            if self.game.make_move(x, y):
                # 停止当前计时器
                self.stop_timer()
                
                # 绘制棋子
                self.draw_pieces()
                
                # 更新历史记录
                self.update_history()
                
                # 更新状态
                self.update_status()
                
                # 检查游戏是否结束
                if self.game.game_over:
                    if self.game.winner == 1:
                        messagebox.showinfo("游戏结束", "黑方获胜！")
                    elif self.game.winner == 2:
                        messagebox.showinfo("游戏结束", "白方获胜！")
                    else:
                        messagebox.showinfo("游戏结束", "平局！")
                else:
                    # 更新待下棋子数
                    self.update_pending_moves()
                    
                    # 开始下一个玩家的计时器
                    self.start_timer()
                    
                    # 检查是否需要AI落子
                    if self.game_mode == 1:
                        ai_player = 1 if self.first_player_var.get() == "AI先手" else 2
                        if self.game.current_player == ai_player:
                            self.ai_move()
    
    def new_game(self):
        self.game.reset()
        self.canvas.delete("piece")
        if self.game_mode == 0:
            self.status_var.set("黑方先行")
            self.pending_var.set("黑方还需下1子")
        else:
            # 根据先手选择设置
            if self.first_player_var.get() == "人类先手":
                self.status_var.set("黑方先行(人机对战)")
                self.pending_var.set("黑方还需下1子")
                # 人类是黑方，AI是白方
                ai_player = 2
            else:
                self.status_var.set("AI先行(人机对战)")
                self.pending_var.set("AI还需下1子")
                # AI是黑方，人类是白方
                ai_player = 1
            
        # 重置AI状态
        if self.ai:
            difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}
            difficulty = difficulty_map.get(self.difficulty_var.get(), "hard")
            # 根据先手选择设置AI的玩家ID
            ai_player = 1 if self.first_player_var.get() == "AI先手" else 2
            # 根据选择的AI类型初始化
            if self.ai_type_var.get() == "MinimaxAI":
                self.ai = MinimaxAI(player=ai_player, difficulty=difficulty)
            else:
                self.ai = WebUIAI(player=ai_player, difficulty=difficulty)
        
        # 更新AI先走按钮状态
        self.update_ai_first_move_btn_state()
    
    def undo(self):
        if self.game.undo():
            self.draw_pieces()
            self.update_status()
            self.update_pending_moves()
    
    def update_status(self):
        if self.game_mode == 1:
            if self.first_player_var.get() == "人类先手":
                # 人类是黑方(1)，AI是白方(2)
                if self.game.current_player == 1:
                    self.status_var.set("黑方下棋")
                else:
                    self.status_var.set("AI下棋")
            else:
                # AI是黑方(1)，人类是白方(2)
                if self.game.current_player == 1:
                    self.status_var.set("AI下棋")
                else:
                    self.status_var.set("白方下棋")
        else:
            if self.game.current_player == 1:
                self.status_var.set("黑方下棋")
            else:
                self.status_var.set("白方下棋")
    
    def update_pending_moves(self):
        # 计算当前玩家还需要下多少棋子
        if self.game.move_count == 0:
            self.pending_moves = 1  # 黑方第一手只下1子
        elif self.game.move_count == 1:
            # 黑方已下1子，白方开始下2子
            self.pending_moves = 2
        elif self.game.move_count % 2 == 0:
            # 偶数步数（2,4,6...）：当前玩家已下1子，还需下1子
            self.pending_moves = 1
        else:
            # 奇数步数（3,5,7...）：当前玩家已下2子，下一个玩家需要下2子
            self.pending_moves = 2
        
        # 更新待下棋子数标签
        if self.game_mode == 1:
            if self.first_player_var.get() == "人类先手":
                # 人类是黑方(1)，AI是白方(2)
                if self.game.current_player == 1:
                    player = "黑方"
                else:
                    player = "AI"
            else:
                # AI是黑方(1)，人类是白方(2)
                if self.game.current_player == 1:
                    player = "AI"
                else:
                    player = "白方"
        else:
            if self.game.current_player == 1:
                player = "黑方"
            else:
                player = "白方"
        self.pending_var.set(f"{player}还需下{self.pending_moves}子")
    
    def show_rules(self):
        rules = "六子棋规则：\n\n"
        rules += "1. 黑方先行，第一手只下1子\n"
        rules += "2. 之后黑白双方轮流各下2子\n"
        rules += "3. 先连成6子或以上者获胜\n"
        rules += "4. 棋子可以横向、纵向或斜向连接\n"
        rules += "5. 棋盘为19×19，与围棋棋盘相同"
        messagebox.showinfo("游戏规则", rules)
    
    def change_mode(self):
        if self.mode_var.get() == "人人对战":
            self.game_mode = 0
            self.status_var.set("黑方先行")
        else:
            self.game_mode = 1
            self.status_var.set("黑方先行(人机对战)")
        self.new_game()
        # 更新AI先走按钮状态
        self.update_ai_first_move_btn_state()
    
    def change_difficulty(self):
        # 只有在人机对战模式下才需要更新AI
        if self.game_mode == 1:
            self.new_game()
    
    def change_first_player(self):
        # 只有在人机对战模式下才需要更新
        if self.game_mode == 1:
            self.new_game()
        # 更新AI先走按钮状态
        self.update_ai_first_move_btn_state()
    
    def change_ai_type(self):
        # 只有在人机对战模式下才需要更新
        if self.game_mode == 1:
            self.new_game()
    
    def update_ai_first_move_btn_state(self):
        # 只有在人机对战模式且选择AI先手时启用按钮
        if self.game_mode == 1 and self.first_player_var.get() == "AI先手" and self.game.move_count == 0:
            self.ai_first_move_btn.config(state=tk.NORMAL)
        else:
            self.ai_first_move_btn.config(state=tk.DISABLED)
    
    def ai_first_move(self):
        # AI先手走第一子
        if self.game_mode == 1 and self.first_player_var.get() == "AI先手" and self.game.move_count == 0:
            self.ai_move()
    
    def ai_move(self):
        # AI思考中
        self.status_var.set("AI思考中...")
        self.root.update()
        
        # 获取AI的最佳落子
        best_move = self.ai.get_best_move(self.game.board)
        
        if best_move:
            # 尝试落子
            if self.game.make_move(best_move[0], best_move[1]):
                # 绘制棋子
                self.draw_pieces()
                
                # 更新状态
                self.update_status()
                
                # 检查游戏是否结束
                if self.game.game_over:
                    if self.game.winner == 1:
                        messagebox.showinfo("游戏结束", "黑方获胜！")
                    elif self.game.winner == 2:
                        messagebox.showinfo("游戏结束", "AI获胜！")
                    else:
                        messagebox.showinfo("游戏结束", "平局！")
                else:
                    # 更新待下棋子数
                    self.update_pending_moves()
                    # 更新AI先走按钮状态
                    self.update_ai_first_move_btn_state()
                    
                    # 检查是否需要AI继续落子
                    # 六子棋规则：黑方第一手只下1子，之后每方每回合下2子
                    ai_player = 1 if self.first_player_var.get() == "AI先手" else 2
                    is_first_move = self.game.move_count == 1
                    
                    # 如果是AI的回合，且不是第一手（第一手只下1子），则继续下第二子
                    if self.game_mode == 1 and self.game.current_player == ai_player and not is_first_move:
                        # 延迟一下，让玩家能看到第一子
                        self.root.after(500, self.ai_move)
    
    def update_history(self):
        """更新历史记录显示"""
        self.history_listbox.delete(0, tk.END)
        for i, move in enumerate(self.game.move_history):
            player_text = "黑方" if move['player'] == 1 else "白方"
            self.history_listbox.insert(tk.END, f"第{i+1}步: {player_text} ({move['x']}, {move['y']})")
    
    def on_history_select(self, event):
        """点击历史记录跳转到指定步数"""
        if not self.history_listbox.curselection():
            return
        
        index = self.history_listbox.curselection()[0]
        if self.game.jump_to_move(index):
            self.draw_pieces()
            self.update_status()
            self.update_pending_moves()
            self.update_ai_first_move_btn_state()
    
    def export_game(self):
        """导出棋谱为JSON文件"""
        if not self.game.move_history:
            messagebox.showinfo("提示", "当前没有可导出的棋谱！")
            return
        
        game_data = {
            'boardSize': self.game.board.size,
            'moveHistory': self.game.move_history,
            'currentPlayer': self.game.current_player,
            'gameOver': self.game.game_over,
            'winner': self.game.winner,
            'totalMoves': self.game.move_count,
            'timestamp': datetime.now().isoformat()
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"connect6_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(game_data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", "棋谱已成功导出！")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：{str(e)}")
    
    def import_game(self):
        """从JSON文件导入棋谱"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                
                if 'moveHistory' not in game_data or not isinstance(game_data['moveHistory'], list):
                    raise ValueError("无效的棋谱格式")
                
                # 重置游戏
                self.game.reset()
                
                # 应用棋谱
                self.game.move_history = game_data['moveHistory']
                for move in self.game.move_history:
                    self.game.board.set(move['x'], move['y'], move['player'])
                    self.game.move_count += 1
                    
                    # 检查获胜
                    if self.game.check_win(move['x'], move['y'], move['player']):
                        self.game.game_over = True
                        self.game.winner = move['player']
                    
                    # 切换玩家
                    if self.game.move_count == 1:
                        self.game.current_player = 2
                    elif self.game.move_count >= 3 and self.game.move_count % 2 == 1:
                        self.game.current_player = 1 if self.game.current_player == 2 else 2
                
                # 如果没有结束，设置正确的当前玩家
                if not self.game.game_over:
                    if self.game.move_count == 1:
                        self.game.current_player = 2
                    elif self.game.move_count == 0:
                        self.game.current_player = 1
                    else:
                        if self.game.move_count % 2 == 0:
                            self.game.current_player = 1
                        else:
                            self.game.current_player = 2
                
                # 更新UI
                self.draw_pieces()
                self.update_history()
                self.update_status()
                self.update_pending_moves()
                self.update_ai_first_move_btn_state()
                
                messagebox.showinfo("成功", "棋谱已成功导入！")
            except Exception as e:
                messagebox.showerror("错误", f"导入失败：{str(e)}")
    
    def new_game(self):
        self.game.reset()
        self.canvas.delete("piece")
        self.history_listbox.delete(0, tk.END)
        if self.game_mode == 0:
            self.status_var.set("黑方先行")
            self.pending_var.set("黑方还需下1子")
        else:
            # 根据先手选择设置
            if self.first_player_var.get() == "人类先手":
                self.status_var.set("黑方先行(人机对战)")
                self.pending_var.set("黑方还需下1子")
                # 人类是黑方，AI是白方
                ai_player = 2
            else:
                self.status_var.set("AI先行(人机对战)")
                self.pending_var.set("AI还需下1子")
                # AI是黑方，人类是白方
                ai_player = 1
            
        # 重置AI状态
        if self.ai:
            difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}
            difficulty = difficulty_map.get(self.difficulty_var.get(), "hard")
            # 根据先手选择设置AI的玩家ID
            ai_player = 1 if self.first_player_var.get() == "AI先手" else 2
            # 根据选择的AI类型初始化
            if self.ai_type_var.get() == "MinimaxAI":
                self.ai = MinimaxAI(player=ai_player, difficulty=difficulty)
            else:
                self.ai = WebUIAI(player=ai_player, difficulty=difficulty)
        
        # 重置计时器
        self.reset_timer()
        # 开始当前玩家的计时器
        self.start_timer()
        
        # 更新AI先走按钮状态
        self.update_ai_first_move_btn_state()
    
    def undo(self):
        if self.game.undo():
            self.draw_pieces()
            self.update_history()
            self.update_status()
            self.update_pending_moves()
            # 重置计时器
            self.reset_timer()
            # 开始当前玩家的计时器
            self.start_timer()
    
    def reset_timer(self):
        """重置计时器"""
        # 停止当前计时器
        self.stop_timer()
        
        # 重置时间
        self.black_time = 600
        self.white_time = 600
        
        # 更新计时器显示
        self.update_timer_display()
    
    def start_timer(self):
        """开始计时器"""
        if self.is_timer_running:
            return
        
        self.is_timer_running = True
        self.timer_interval = self.root.after(1000, self.timer_tick)
    
    def stop_timer(self):
        """停止计时器"""
        if self.timer_interval:
            self.root.after_cancel(self.timer_interval)
            self.timer_interval = None
        self.is_timer_running = False
    
    def timer_tick(self):
        """计时器 tick"""
        if self.game.game_over:
            self.stop_timer()
            return
        
        if self.game.current_player == 1:
            self.black_time -= 1
            if self.black_time <= 0:
                self.black_time = 0
                self.stop_timer()
                self.handle_timeout(1)
        else:
            self.white_time -= 1
            if self.white_time <= 0:
                self.white_time = 0
                self.stop_timer()
                self.handle_timeout(2)
        
        self.update_timer_display()
        self.timer_interval = self.root.after(1000, self.timer_tick)
    
    def update_timer_display(self):
        """更新计时器显示"""
        self.black_timer_var.set(self.format_time(self.black_time))
        self.white_timer_var.set(self.format_time(self.white_time))
        
        # 添加警告效果
        if self.black_time < 10:
            self.black_timer_display.config(fg="red")
        else:
            self.black_timer_display.config(fg="black")
        
        if self.white_time < 10:
            self.white_timer_display.config(fg="red")
        else:
            self.white_timer_display.config(fg="black")
    
    def format_time(self, seconds):
        """格式化时间"""
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def handle_timeout(self, player):
        """处理超时"""
        self.game.game_over = True
        self.game.winner = 2 if player == 1 else 1
        self.update_status()
        messagebox.showinfo("游戏结束", f"{'黑方' if player == 1 else '白方'}超时，对方获胜！")
    
    def ai_move(self):
        # AI思考中
        self.status_var.set("AI思考中...")
        self.root.update()
        
        # 获取AI的最佳落子
        best_move = self.ai.get_best_move(self.game.board)
        
        if best_move:
            # 尝试落子
            if self.game.make_move(best_move[0], best_move[1]):
                # 停止当前计时器
                self.stop_timer()
                
                # 绘制棋子
                self.draw_pieces()
                
                # 更新历史记录
                self.update_history()
                
                # 更新状态
                self.update_status()
                
                # 检查游戏是否结束
                if self.game.game_over:
                    if self.game.winner == 1:
                        messagebox.showinfo("游戏结束", "黑方获胜！")
                    elif self.game.winner == 2:
                        messagebox.showinfo("游戏结束", "AI获胜！")
                    else:
                        messagebox.showinfo("游戏结束", "平局！")
                else:
                    # 更新待下棋子数
                    self.update_pending_moves()
                    # 更新AI先走按钮状态
                    self.update_ai_first_move_btn_state()
                    
                    # 开始下一个玩家的计时器
                    self.start_timer()
                    
                    # 检查是否需要AI继续落子
                    # 六子棋规则：黑方第一手只下1子，之后每方每回合下2子
                    ai_player = 1 if self.first_player_var.get() == "AI先手" else 2
                    is_first_move = self.game.move_count == 1
                    
                    # 如果是AI的回合，且不是第一手（第一手只下1子），则继续下第二子
                    if self.game_mode == 1 and self.game.current_player == ai_player and not is_first_move:
                        # 延迟一下，让玩家能看到第一子
                        self.root.after(500, self.ai_move)

if __name__ == "__main__":
    root = tk.Tk()
    app = Connect6UI(root)
    root.mainloop()
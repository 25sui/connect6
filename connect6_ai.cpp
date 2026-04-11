/**
 * Connect6 (六子棋) AI - 竞赛级别实现
 * 算法: Minimax + Alpha-Beta剪枝 + 迭代加深 + 启发式评估
 * 作者: AI Assistant
 * 版本: 1.0
 */

#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#ifdef AI_SERVER
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif
using namespace std;

// ==================== 常量定义 ====================
const int BOARD_SIZE = 19;           // 棋盘大小
const int MAX_DEPTH = 10;            // 最大搜索深度
const int TIME_LIMIT_MS = 4500;      // 时间限制(毫秒)，留500ms缓冲
const int ITERATIVE_DEEPENING_STEP = 1;  // 迭代加深步长

// 棋型评分 (从高到低)
const int SCORE_SIX = 100000000;     // 连六 (必胜)
const int SCORE_LIVE_FIVE = 50000000; // 活五
const int SCORE_DEAD_FIVE = 10000000; // 死五
const int SCORE_LIVE_FOUR = 5000000;  // 活四
const int SCORE_DEAD_FOUR = 500000;   // 死四
const int SCORE_LIVE_THREE = 100000;  // 活三
const int SCORE_DEAD_THREE = 10000;   // 死三
const int SCORE_LIVE_TWO = 1000;      // 活二
const int SCORE_DEAD_TWO = 100;       // 死二
const int SCORE_LIVE_ONE = 10;        // 活一

// 方向数组: 右, 下, 右下, 左下
const int DX[4] = {1, 0, 1, -1};
const int DY[4] = {0, 1, 1, 1};

// 棋子类型
enum Piece {
    EMPTY = 0,
    BLACK = 1,
    WHITE = 2
};

// 位置结构
struct Position {
    int x, y;
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Position& other) const {
        return !(*this == other);
    }
};

// 走法结构
struct Move {
    Position p1, p2;  // 六子棋每步下两子
    int score;
    Move() : p1(-1, -1), p2(-1, -1), score(0) {}
    Move(Position p1, Position p2, int score = 0) : p1(p1), p2(p2), score(score) {}
};

// ==================== 棋盘类 ====================
class Board {
public:
    int board[BOARD_SIZE][BOARD_SIZE];
    int moveCount;  // 已走步数
    
    Board() {
        memset(board, 0, sizeof(board));
        moveCount = 0;
    }
    
    Board(const Board& other) {
        memcpy(board, other.board, sizeof(board));
        moveCount = other.moveCount;
    }
    
    // 检查位置是否在棋盘内
    inline bool isValid(int x, int y) const {
        return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
    }
    
    // 检查位置是否为空
    inline bool isEmpty(int x, int y) const {
        return isValid(x, y) && board[x][y] == EMPTY;
    }
    
    // 落子
    inline void place(int x, int y, Piece piece) {
        board[x][y] = piece;
        moveCount++;
    }
    
    // 移除棋子
    inline void remove(int x, int y) {
        board[x][y] = EMPTY;
        moveCount--;
    }
    
    // 获取棋子
    inline int get(int x, int y) const {
        if (isValid(x, y)) return board[x][y];
        return -1;
    }
    
    // 检查是否获胜
    bool checkWin(int x, int y, Piece piece) const {
        for (int dir = 0; dir < 4; dir++) {
            int count = 1;
            
            // 正方向
            int nx = x + DX[dir], ny = y + DY[dir];
            while (isValid(nx, ny) && board[nx][ny] == piece) {
                count++;
                nx += DX[dir];
                ny += DY[dir];
            }
            
            // 反方向
            nx = x - DX[dir];
            ny = y - DY[dir];
            while (isValid(nx, ny) && board[nx][ny] == piece) {
                count++;
                nx -= DX[dir];
                ny -= DY[dir];
            }
            
            if (count >= 6) return true;
        }
        return false;
    }
    
    // 检查游戏是否结束 (有六连)
    bool isGameOver() const {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != EMPTY) {
                    if (checkWin(i, j, (Piece)board[i][j])) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    // 获取获胜方
    Piece getWinner() const {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != EMPTY) {
                    if (checkWin(i, j, (Piece)board[i][j])) {
                        return (Piece)board[i][j];
                    }
                }
            }
        }
        return EMPTY;
    }
    
    // 清空棋盘
    void clear() {
        memset(board, 0, sizeof(board));
        moveCount = 0;
    }
    
    // 复制棋盘
    void copy(const Board& other) {
        memcpy(board, other.board, sizeof(board));
        moveCount = other.moveCount;
    }
};

// ==================== 评估函数类 ====================
class Evaluator {
public:
    // 棋型缓存
    int patternCache[4][BOARD_SIZE][BOARD_SIZE][3];  // [方向][x][y][棋子类型]
    bool cacheValid[4][BOARD_SIZE][BOARD_SIZE][3];
    
    Evaluator() {
        memset(cacheValid, 0, sizeof(cacheValid));
    }
    
    void clearCache() {
        memset(cacheValid, 0, sizeof(cacheValid));
    }
    
    // 获取某方向的棋型
    // 返回: (连续数, 是否活)
    pair<int, bool> getPattern(const Board& board, int x, int y, int dir, Piece piece) {
        if (!board.isValid(x, y) || board.get(x, y) != piece) {
            return {0, false};
        }
        
        int count = 1;
        bool leftOpen = false, rightOpen = false;
        
        // 正方向
        int nx = x + DX[dir], ny = y + DY[dir];
        while (board.isValid(nx, ny) && board.get(nx, ny) == piece) {
            count++;
            nx += DX[dir];
            ny += DY[dir];
        }
        // 检查正方向是否被堵住
        if (board.isValid(nx, ny) && board.get(nx, ny) == EMPTY) {
            rightOpen = true;
        }
        
        // 反方向
        nx = x - DX[dir];
        ny = y - DY[dir];
        while (board.isValid(nx, ny) && board.get(nx, ny) == piece) {
            count++;
            nx -= DX[dir];
            ny -= DY[dir];
        }
        // 检查反方向是否被堵住
        if (board.isValid(nx, ny) && board.get(nx, ny) == EMPTY) {
            leftOpen = true;
        }
        
        bool isLive = leftOpen && rightOpen;
        return {count, isLive};
    }
    
    // 棋型类型定义
enum PatternType {
    PATTERN_NONE,
    PATTERN_LIVE_ONE,
    PATTERN_DEAD_ONE,
    PATTERN_LIVE_TWO,
    PATTERN_DEAD_TWO,
    PATTERN_LIVE_THREE,
    PATTERN_DEAD_THREE,
    PATTERN_LIVE_FOUR,
    PATTERN_DEAD_FOUR,
    PATTERN_LIVE_FIVE,
    PATTERN_DEAD_FIVE,
    PATTERN_SIX
};

// 棋型信息结构
struct PatternInfo {
    PatternType type;
    int length;
    bool open;
    int score;
};

// 识别棋型
PatternInfo recognizePattern(const Board& board, int x, int y, int dir, Piece piece) {
    if (!board.isValid(x, y) || board.get(x, y) != piece) {
        return {PATTERN_NONE, 0, false, 0};
    }
    
    int count = 1;
    bool leftOpen = false, rightOpen = false;
    bool leftBlock = false, rightBlock = false;
    
    // 正方向
    int nx = x + DX[dir], ny = y + DY[dir];
    while (board.isValid(nx, ny) && board.get(nx, ny) == piece) {
        count++;
        nx += DX[dir];
        ny += DY[dir];
    }
    
    if (board.isValid(nx, ny)) {
        if (board.get(nx, ny) == EMPTY) {
            rightOpen = true;
        } else {
            rightBlock = true;
        }
    }
    
    // 反方向
    nx = x - DX[dir];
    ny = y - DY[dir];
    while (board.isValid(nx, ny) && board.get(nx, ny) == piece) {
        count++;
        nx -= DX[dir];
        ny -= DY[dir];
    }
    
    if (board.isValid(nx, ny)) {
        if (board.get(nx, ny) == EMPTY) {
            leftOpen = true;
        } else {
            leftBlock = true;
        }
    }
    
    bool isLive = leftOpen && rightOpen;
    PatternType type;
    int score;
    
    if (count >= 6) {
        type = PATTERN_SIX;
        score = SCORE_SIX;
    } else if (count == 5) {
        type = isLive ? PATTERN_LIVE_FIVE : PATTERN_DEAD_FIVE;
        score = isLive ? SCORE_LIVE_FIVE : SCORE_DEAD_FIVE;
    } else if (count == 4) {
        type = isLive ? PATTERN_LIVE_FOUR : PATTERN_DEAD_FOUR;
        score = isLive ? SCORE_LIVE_FOUR : SCORE_DEAD_FOUR;
    } else if (count == 3) {
        type = isLive ? PATTERN_LIVE_THREE : PATTERN_DEAD_THREE;
        score = isLive ? SCORE_LIVE_THREE : SCORE_DEAD_THREE;
    } else if (count == 2) {
        type = isLive ? PATTERN_LIVE_TWO : PATTERN_DEAD_TWO;
        score = isLive ? SCORE_LIVE_TWO : SCORE_DEAD_TWO;
    } else {
        type = PATTERN_LIVE_ONE;
        score = SCORE_LIVE_ONE;
    }
    
    return {type, count, isLive, score};
}

    // 评估单个位置
    int evaluatePosition(const Board& board, int x, int y, Piece piece) {
        if (board.get(x, y) != piece) return 0;
        
        int score = 0;
        static const int centerX = BOARD_SIZE / 2;
        static const int centerY = BOARD_SIZE / 2;
        
        // 位置价值评估（一次性计算）
        int distToCenter = abs(x - centerX) + abs(y - centerY);
        int positionScore = max(0, 20 - distToCenter) * 15;
        if (x <= 2 || x >= BOARD_SIZE - 3 || y <= 2 || y >= BOARD_SIZE - 3) {
            positionScore = static_cast<int>(positionScore * 0.8);  // 边缘位置价值降低
        }
        
        for (int dir = 0; dir < 4; dir++) {
            // 只计算从该位置开始的棋型，避免重复计算
            int nx = x - DX[dir], ny = y - DY[dir];
            if (board.isValid(nx, ny) && board.get(nx, ny) == piece) {
                continue;  // 不是起点，跳过
            }
            
            PatternInfo info = recognizePatternEnhanced(board, x, y, dir, piece);
            score += info.score;
            
            // 额外的战术价值评估
            switch (info.type) {
                case PATTERN_LIVE_FOUR:
                    score += 80000;  // 活四的价值更高
                    break;
                case PATTERN_DEAD_FOUR:
                    score += 40000;  // 死四也有较高价值
                    break;
                case PATTERN_LIVE_THREE:
                    score += 20000;  // 活三的价值
                    break;
                case PATTERN_DEAD_THREE:
                    score += 10000;  // 死三的价值
                    break;
                case PATTERN_LIVE_TWO:
                    score += 5000;  // 活二的价值
                    break;
            }
            
            // 添加位置价值
            score += positionScore;
        }
        
        return score;
    }
    
    // 评估整个棋盘
    int evaluate(const Board& board, Piece myPiece) {
        Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
        
        int myScore = 0;
        int opponentScore = 0;
        
        // 检查是否有获胜方
        Piece winner = board.getWinner();
        if (winner == myPiece) return SCORE_SIX;
        if (winner == opponent) return -SCORE_SIX;
        
        // 评估每个位置
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == myPiece) {
                    myScore += evaluatePosition(board, i, j, myPiece);
                } else if (board.get(i, j) == opponent) {
                    opponentScore += evaluatePosition(board, i, j, opponent);
                }
            }
        }
        
        // 应用权重调整
        double attackWeight = 1.0;
        double defenseWeight = 1.0;
        double positionWeight = 1.0;
        myScore = static_cast<int>(myScore * attackWeight);
        opponentScore = static_cast<int>(opponentScore * defenseWeight);
        
        // 防守价值评估 - 优先防守对方的威胁
        int defenseMultiplier = 1;
        if (opponentScore > myScore * 1.2) {
            defenseMultiplier = 1.3;  // 对方威胁较大时，加强防守
        } else if (opponentScore > myScore * 1.5) {
            defenseMultiplier = 1.5;  // 对方威胁很大时，大幅加强防守
        }
        
        // 棋盘阶段评估
        int moveCount = board.moveCount;
        if (moveCount < 8) {
            // 开局阶段，位置价值更重要
            myScore = static_cast<int>(myScore * 1.2 * positionWeight);
        } else if (moveCount > 40) {
            // 中局阶段，进攻价值更重要
            myScore = static_cast<int>(myScore * 1.3 * attackWeight);
        } else if (moveCount > 60) {
            // 残局阶段，必须全力以赴
            myScore = static_cast<int>(myScore * 1.5 * attackWeight);
        }
        
        // 棋子密度评估
        int pieceCount = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    pieceCount++;
                }
            }
        }
        
        // 密集局面，进攻价值提升
        if (pieceCount > 40) {
            myScore = static_cast<int>(myScore * 1.2 * attackWeight);
        }
        
        // 空间优势评估
        int myMobility = 0;
        int opponentMobility = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == EMPTY) {
                    // 计算周围棋子数量
                    int myAround = 0, opponentAround = 0;
                    for (int dx = -2; dx <= 2; dx++) {
                        for (int dy = -2; dy <= 2; dy++) {
                            int nx = i + dx, ny = j + dy;
                            if (board.isValid(nx, ny)) {
                                if (board.get(nx, ny) == myPiece) {
                                    myAround++;
                                } else if (board.get(nx, ny) == opponent) {
                                    opponentAround++;
                                }
                            }
                        }
                    }
                    if (myAround > opponentAround) {
                        myMobility++;
                    } else if (opponentAround > myAround) {
                        opponentMobility++;
                    }
                }
            }
        }
        
        // 机动性优势
        myScore += static_cast<int>((myMobility - opponentMobility) * 100 * positionWeight);
        
        return myScore - opponentScore * defenseMultiplier;
    }
    
    // 增强的棋型识别
    PatternInfo recognizePatternEnhanced(const Board& board, int x, int y, int dir, Piece piece) {
        PatternInfo info = recognizePattern(board, x, y, dir, piece);
        
        // 增强的棋型评估
        if (info.type == PATTERN_LIVE_THREE) {
            // 检查是否有潜在的活四
            bool hasPotential = false;
            int nx = x, ny = y;
            for (int i = 0; i < info.length; i++) {
                int tx = nx + DX[dir] * (info.length);
                int ty = ny + DY[dir] * (info.length);
                if (board.isValid(tx, ty) && board.get(tx, ty) == EMPTY) {
                    hasPotential = true;
                    break;
                }
                nx += DX[dir];
                ny += DY[dir];
            }
            if (hasPotential) {
                info.score += 10000;  // 潜在活四的额外价值
            }
        } else if (info.type == PATTERN_DEAD_THREE) {
            // 检查是否有潜在的死四
            bool hasPotential = false;
            int nx = x, ny = y;
            for (int i = 0; i < info.length; i++) {
                int tx = nx + DX[dir] * (info.length);
                int ty = ny + DY[dir] * (info.length);
                if (board.isValid(tx, ty) && board.get(tx, ty) == EMPTY) {
                    hasPotential = true;
                    break;
                }
                nx += DX[dir];
                ny += DY[dir];
            }
            if (hasPotential) {
                info.score += 5000;  // 潜在死四的额外价值
            }
        } else if (info.type == PATTERN_LIVE_TWO) {
            // 检查是否有潜在的活三
            bool hasPotential = false;
            int nx = x, ny = y;
            for (int i = 0; i < info.length; i++) {
                int tx = nx + DX[dir] * (info.length);
                int ty = ny + DY[dir] * (info.length);
                if (board.isValid(tx, ty) && board.get(tx, ty) == EMPTY) {
                    // 检查是否形成活三
                    bool canFormLiveThree = true;
                    int tempX = tx, tempY = ty;
                    for (int j = 0; j < 2; j++) {
                        tempX += DX[dir];
                        tempY += DY[dir];
                        if (!board.isValid(tempX, tempY) || board.get(tempX, tempY) != EMPTY) {
                            canFormLiveThree = false;
                            break;
                        }
                    }
                    if (canFormLiveThree) {
                        hasPotential = true;
                        break;
                    }
                }
                nx += DX[dir];
                ny += DY[dir];
            }
            if (hasPotential) {
                info.score += 3000;  // 潜在活三的额外价值
            }
        }
        
        // 检查连续多个棋型的组合价值
        // 例如：两个活三的组合
        int comboScore = 0;
        int count = 1;
        int currentX = x, currentY = y;
        while (true) {
            currentX += DX[dir];
            currentY += DY[dir];
            if (!board.isValid(currentX, currentY) || board.get(currentX, currentY) != piece) {
                break;
            }
            count++;
        }
        
        if (count >= 4) {
            comboScore += 20000;  // 长连的额外价值
        } else if (count == 3) {
            comboScore += 10000;  // 三连的额外价值
        }
        
        info.score += comboScore;
        
        return info;
    }
    
    // 快速评估 - 只检查关键位置
    int evaluateFast(const Board& board, Piece myPiece, int lastX, int lastY) {
        Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
        
        // 检查最后落子位置是否获胜
        if (lastX >= 0 && board.checkWin(lastX, lastY, opponent)) {
            return -SCORE_SIX;
        }
        
        return evaluate(board, myPiece);
    }
};

// ==================== 机器学习评估器 ====================
class MLEvaluator {
private:
    // 神经网络权重
    double weights[128]; // 128个特征的权重
    
    // 特征提取
    vector<double> extractFeatures(const Board& board, Piece myPiece) {
        vector<double> features(128, 0.0);
        Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
        
        // 特征1-16: 我方各棋型数量
        int myPatterns[16] = {0}; // 0-3: 活一到活四, 4-7: 死一到死四, 8-11: 活五到死五, 12-15: 其他
        // 特征17-32: 对方各棋型数量
        int opponentPatterns[16] = {0};
        // 特征33-48: 位置价值
        double positionValue = 0.0;
        // 特征49-64: 机动性
        int myMobility = 0, opponentMobility = 0;
        // 特征65-80: 棋子密度和分布
        double pieceDensity = 0.0, centralControl = 0.0;
        // 特征81-96: 战术模式
        int myThreats = 0, opponentThreats = 0;
        // 特征97-112: 棋子连接性
        int myConnections = 0, opponentConnections = 0;
        // 特征113-128: 棋盘阶段和时间控制
        double gamePhase = 0.0, urgency = 0.0;
        
        // 提取棋型特征
        Evaluator evaluator;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == myPiece) {
                    for (int dir = 0; dir < 4; dir++) {
                        int nx = i - DX[dir], ny = j - DY[dir];
                        if (board.isValid(nx, ny) && board.get(nx, ny) == myPiece) {
                            continue;
                        }
                        auto info = evaluator.recognizePattern(board, i, j, dir, myPiece);
                        if (info.type >= 1 && info.type <= 12) {
                            myPatterns[info.type - 1]++;
                        }
                        // 检查威胁
                        if (info.type == Evaluator::PATTERN_LIVE_FOUR || info.type == Evaluator::PATTERN_LIVE_THREE) {
                            myThreats++;
                        }
                    }
                    // 检查连接性
                    int connections = 0;
                    for (int dir = 0; dir < 4; dir++) {
                        int nx = i + DX[dir], ny = j + DY[dir];
                        if (board.isValid(nx, ny) && board.get(nx, ny) == myPiece) {
                            connections++;
                        }
                    }
                    myConnections += connections;
                } else if (board.get(i, j) == opponent) {
                    for (int dir = 0; dir < 4; dir++) {
                        int nx = i - DX[dir], ny = j - DY[dir];
                        if (board.isValid(nx, ny) && board.get(nx, ny) == opponent) {
                            continue;
                        }
                        auto info = evaluator.recognizePattern(board, i, j, dir, opponent);
                        if (info.type >= 1 && info.type <= 12) {
                            opponentPatterns[info.type - 1]++;
                        }
                        // 检查威胁
                        if (info.type == Evaluator::PATTERN_LIVE_FOUR || info.type == Evaluator::PATTERN_LIVE_THREE) {
                            opponentThreats++;
                        }
                    }
                    // 检查连接性
                    int connections = 0;
                    for (int dir = 0; dir < 4; dir++) {
                        int nx = i + DX[dir], ny = j + DY[dir];
                        if (board.isValid(nx, ny) && board.get(nx, ny) == opponent) {
                            connections++;
                        }
                    }
                    opponentConnections += connections;
                }
            }
        }
        
        // 填充棋型特征
        for (int i = 0; i < 16; i++) {
            features[i] = myPatterns[i] / 10.0;
            features[i + 16] = opponentPatterns[i] / 10.0;
        }
        
        // 提取位置价值特征
        int centerX = BOARD_SIZE / 2;
        int centerY = BOARD_SIZE / 2;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == myPiece) {
                    int dist = abs(i - centerX) + abs(j - centerY);
                    positionValue += (20 - dist) / 20.0;
                    // 中心控制
                    if (dist <= 3) {
                        centralControl += 1.0;
                    }
                }
            }
        }
        features[32] = positionValue / (BOARD_SIZE * BOARD_SIZE);
        features[33] = centralControl / 25.0; // 中心区域最多25个位置
        
        // 提取机动性特征
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == EMPTY) {
                    int myAround = 0, opponentAround = 0;
                    for (int dx = -2; dx <= 2; dx++) {
                        for (int dy = -2; dy <= 2; dy++) {
                            int nx = i + dx, ny = j + dy;
                            if (board.isValid(nx, ny)) {
                                if (board.get(nx, ny) == myPiece) {
                                    myAround++;
                                } else if (board.get(nx, ny) == opponent) {
                                    opponentAround++;
                                }
                            }
                        }
                    }
                    if (myAround > opponentAround) {
                        myMobility++;
                    } else if (opponentAround > myAround) {
                        opponentMobility++;
                    }
                }
            }
        }
        features[48] = myMobility / 100.0;
        features[49] = opponentMobility / 100.0;
        
        // 提取棋子密度和分布特征
        int pieceCount = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    pieceCount++;
                }
            }
        }
        pieceDensity = static_cast<double>(pieceCount) / (BOARD_SIZE * BOARD_SIZE);
        features[64] = pieceDensity;
        
        // 提取战术模式特征
        features[80] = myThreats / 10.0;
        features[81] = opponentThreats / 10.0;
        
        // 提取棋子连接性特征
        features[96] = myConnections / 50.0;
        features[97] = opponentConnections / 50.0;
        
        // 提取棋盘阶段和紧急度特征
        int moveCount = board.moveCount;
        if (moveCount < 10) {
            gamePhase = 0.0; // 开局
        } else if (moveCount < 40) {
            gamePhase = 0.5; // 中局
        } else {
            gamePhase = 1.0; // 残局
        }
        features[112] = gamePhase;
        
        // 紧急度：根据局面评估计算
        int evaluation = evaluator.evaluate(board, myPiece);
        urgency = min(1.0, abs(evaluation) / 100000.0);
        features[113] = urgency;
        
        return features;
    }
    
    // 加载训练好的模型
    void loadModel() {
        // 这里使用预训练的权重
        // 实际应用中，应该从文件加载
        double defaultWeights[128] = {
            // 我方棋型
            1.0, 0.5, 0.3, 0.2,  // 活一到活四
            0.8, 0.4, 0.2, 0.1,  // 死一到死四
            1.5, 0.8, 0.5, 0.3,  // 活五到死五
            0.1, 0.1, 0.1, 0.1,  // 其他
            // 对方棋型
            -1.0, -0.5, -0.3, -0.2,  // 活一到活四
            -0.8, -0.4, -0.2, -0.1,  // 死一到死四
            -1.5, -0.8, -0.5, -0.3,  // 活五到死五
            -0.1, -0.1, -0.1, -0.1,  // 其他
            // 位置价值
            0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // 机动性
            0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // 棋子密度和分布
            0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // 战术模式
            0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // 棋子连接性
            0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // 棋盘阶段和紧急度
            0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
        memcpy(weights, defaultWeights, sizeof(weights));
    }
    
public:
    MLEvaluator() {
        loadModel();
    }
    
    // 评估整个棋盘
    int evaluate(const Board& board, Piece myPiece) {
        Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
        
        // 检查是否有获胜方
        Piece winner = board.getWinner();
        if (winner == myPiece) return 1000000;
        if (winner == opponent) return -1000000;
        
        // 提取特征
        vector<double> features = extractFeatures(board, myPiece);
        
        // 计算加权和
        double score = 0.0;
        for (int i = 0; i < 128; i++) {
            score += features[i] * weights[i];
        }
        
        // 转换为整数分数
        return static_cast<int>(score * 10000);
    }
};

// ==================== 走法生成器 ====================
class MoveGenerator {
public:
    // 获取候选走法位置 (只考虑已有棋子周围的位置)
    vector<Position> getCandidatePositions(const Board& board) {
        vector<Position> candidates;
        static bool considered[BOARD_SIZE][BOARD_SIZE];
        memset(considered, 0, sizeof(considered));
        
        // 如果棋盘为空，返回中心位置
        if (board.moveCount == 0) {
            int center = BOARD_SIZE / 2;
            candidates.push_back(Position(center, center));
            return candidates;
        }
        
        // 预分配内存
        candidates.reserve(50); // 预分配足够的空间
        
        // 遍历所有已有棋子，添加周围空位
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    // 检查周围2格范围内的空位
                    for (int dx = -2; dx <= 2; dx++) {
                        for (int dy = -2; dy <= 2; dy++) {
                            int nx = i + dx, ny = j + dy;
                            if (board.isEmpty(nx, ny) && !considered[nx][ny]) {
                                considered[nx][ny] = true;
                                candidates.emplace_back(nx, ny);
                            }
                        }
                    }
                }
            }
        }
        
        return candidates;
    }
    
    // 评估单个位置的分数 (用于排序)
    int evaluatePositionScore(const Board& board, int x, int y, Piece myPiece, Evaluator& evaluator) {
        Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
        int score = 0;
        
        // 模拟落子并评估
        Board temp = board;
        temp.place(x, y, myPiece);
        score += evaluator.evaluatePosition(temp, x, y, myPiece);
        temp.remove(x, y);
        
        // 考虑防守价值
        temp.place(x, y, opponent);
        score += evaluator.evaluatePosition(temp, x, y, opponent) * 0.8;
        
        // 位置价值
        int centerX = BOARD_SIZE / 2;
        int centerY = BOARD_SIZE / 2;
        int distToCenter = abs(x - centerX) + abs(y - centerY);
        score += max(0, 20 - distToCenter) * 5;
        
        return score;
    }
    
    // 生成所有可能的走法 (六子棋每步两子)
    vector<Move> generateMoves(const Board& board, Piece myPiece, Evaluator& evaluator, int maxMoves = 20) {
        vector<Move> moves;
        vector<Position> candidates = getCandidatePositions(board);
        
        if (candidates.empty()) {
            return moves;
        }
        
        // 评估每个候选位置
        vector<pair<Position, int>> scoredPositions;
        scoredPositions.reserve(candidates.size());
        for (const auto& pos : candidates) {
            int score = evaluatePositionScore(board, pos.x, pos.y, myPiece, evaluator);
            scoredPositions.emplace_back(pos, score);
        }
        
        // 按分数排序
        sort(scoredPositions.begin(), scoredPositions.end(), 
            [](const pair<Position, int>& a, const pair<Position, int>& b) { return a.second > b.second; });
        
        // 限制候选位置数量
        int numCandidates = min((int)scoredPositions.size(), maxMoves * 2);
        
        // 预分配内存
        int maxCombinations = numCandidates * (numCandidates - 1) / 2;
        moves.reserve(maxCombinations);
        
        // 生成所有两子组合
        for (int i = 0; i < numCandidates; i++) {
            for (int j = i + 1; j < numCandidates; j++) {
                Move move(scoredPositions[i].first, scoredPositions[j].first);
                
                // 评估组合价值
                Board temp = board;
                temp.place(move.p1.x, move.p1.y, myPiece);
                temp.place(move.p2.x, move.p2.y, myPiece);
                
                // 检查是否形成威胁
                bool isWin = temp.checkWin(move.p1.x, move.p1.y, myPiece) || 
                            temp.checkWin(move.p2.x, move.p2.y, myPiece);
                
                if (isWin) {
                    move.score = 1000000; // 获胜走法
                } else {
                    // 评估组合的整体价值
                    move.score = scoredPositions[i].second + scoredPositions[j].second;
                    
                    // 检查两个位置之间的协同效应
                    int dx = abs(move.p1.x - move.p2.x);
                    int dy = abs(move.p1.y - move.p2.y);
                    if (dx <= 2 && dy <= 2) {
                        move.score += 1000; // 邻近位置有协同效应
                    }
                }
                
                moves.push_back(move);
            }
        }
        
        // 按分数排序并限制数量
        sort(moves.begin(), moves.end(), 
            [](const Move& a, const Move& b) { return a.score > b.score; });
        
        if (moves.size() > maxMoves) {
            moves.resize(maxMoves);
        }
        
        return moves;
    }
    
    // 生成单步走法 (用于特殊开局)
    vector<Position> generateSingleMoves(const Board& board, Piece myPiece, Evaluator& evaluator, int maxMoves = 10) {
        vector<Position> candidates = getCandidatePositions(board);
        vector<pair<Position, int>> scoredPositions;
        scoredPositions.reserve(candidates.size());
        
        for (const auto& pos : candidates) {
            int score = evaluatePositionScore(board, pos.x, pos.y, myPiece, evaluator);
            scoredPositions.emplace_back(pos, score);
        }
        
        sort(scoredPositions.begin(), scoredPositions.end(), 
            [](const pair<Position, int>& a, const pair<Position, int>& b) { return a.second > b.second; });
        
        vector<Position> result;
        result.reserve(maxMoves);
        int count = min((int)scoredPositions.size(), maxMoves);
        for (int i = 0; i < count; i++) {
            result.push_back(scoredPositions[i].first);
        }
        
        return result;
    }
};

// ==================== 搜索算法类 ====================
class SearchEngine {
public:
    Board board;
    Piece myPiece;
    Piece opponentPiece;
    Evaluator evaluator;
    MLEvaluator mlEvaluator;
    MoveGenerator moveGenerator;
    bool useMLEvaluator = true; // 是否使用机器学习评估器
    
    // 搜索统计
    int nodesSearched;
    int cutoffs;
    
    // 时间控制
    chrono::steady_clock::time_point startTime;
    bool timeUp;
    int timeLimitMs;
    
    // 置换表
    struct TTEntry {
        uint64_t key;
        int depth;
        int score;
        int flag;  // 0=精确, 1=下界, 2=上界
        Move bestMove;
        uint32_t age;  // 用于替换策略
    };
    
    // 使用固定大小的数组作为置换表，提高访问速度
    static const int TT_SIZE = 1 << 20;  // 约100万项
    TTEntry* transpositionTable;
    uint32_t ttAge;  // 全局年龄计数器
    
    // 历史启发表
    int historyTable[BOARD_SIZE][BOARD_SIZE];
    
    // 杀手走法表
    Move killerMoves[MAX_DEPTH][2];
    
    // 并行搜索相关
    mutex searchMutex;
    condition_variable searchCV;
    vector<thread> searchThreads;
    bool parallelSearchEnabled;
    int threadCount;
    vector<Move> parallelMoves;
    vector<int> parallelScores;
    atomic<int> parallelIndex;
    atomic<bool> parallelSearchDone;
    atomic<int> completedTasks;
    
    // 线程池相关
    mutex taskMutex;
    condition_variable taskCV;
    queue<function<void()>> taskQueue;
    atomic<bool> poolRunning;
    atomic<int> activeTasks;
    
    // 初始化线程池
    void initThreadPool() {
        poolRunning = true;
        activeTasks = 0;
        for (int i = 0; i < threadCount; i++) {
            searchThreads.emplace_back([this]() {
                while (poolRunning) {
                    function<void()> task;
                    {
                        unique_lock<mutex> lock(taskMutex);
                        taskCV.wait(lock, [this]() { return !taskQueue.empty() || !poolRunning; });
                        if (!poolRunning && taskQueue.empty()) {
                            return;
                        }
                        if (taskQueue.empty()) {
                            continue;
                        }
                        task = move(taskQueue.front());
                        taskQueue.pop();
                        activeTasks++;
                    }
                    task();
                    activeTasks--;
                }
            });
        }
    }
    
    // 关闭线程池
    void shutdownThreadPool() {
        poolRunning = false;
        taskCV.notify_all();
        for (auto& t : searchThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        searchThreads.clear();
    }
    
    // 提交任务到线程池
    void submitTask(function<void()> task) {
        unique_lock<mutex> lock(taskMutex);
        taskQueue.push(move(task));
        taskCV.notify_one();
    }
    
    // 等待所有任务完成
    void waitForTasks() {
        while (activeTasks > 0 && !timeUp) {
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
    
    // 残局库
    struct OpeningEntry {
        vector<Position> moves;
        Move bestResponse;
    };
    unordered_map<string, OpeningEntry> openingBook;
    unordered_map<string, Move> endgameBook;
    
    // 对手建模系统
    class OpponentModel {
    private:
        // 对手风格特征
    public:
        // 风格分类
        enum StyleType {
            STYLE_AGGRESSIVE, // 进攻型
            STYLE_DEFENSIVE, // 防守型
            STYLE_BALANCED, // 平衡型
            STYLE_POSITIONAL, // 位置型
            STYLE_TACTICAL, // 战术型
            STYLE_UNKNOWN    // 未知
        };
        
        // 开局偏好
        enum OpeningPreference {
            OPENING_CENTER, // 中心开局
            OPENING_DIAGONAL, // 对角线开局
            OPENING_STAR, // 星位开局
            OPENING_EDGE, // 边缘开局
            OPENING_CORNER, // 角落开局
            OPENING_MIXED, // 混合型开局
            OPENING_UNKNOWN // 未知
        };
        
    private:
        int attackMoves; // 进攻性走法数量
        int defenseMoves; // 防守性走法数量
        int balancedMoves; // 平衡性走法数量
        int centerControl; // 中心控制程度
        int edgeControl; // 边缘控制程度
        int cornerControl; // 角落控制程度
        int riskTolerance; // 风险承受能力
        int patternRecognition; // 棋型识别能力
        int openingPreference; // 开局偏好
        int endgameSkill; // 残局技巧
        
        StyleType style;
        OpeningPreference openingPref;
        
    public:
        OpponentModel() {
            reset();
        }
        
        void reset() {
            attackMoves = 0;
            defenseMoves = 0;
            balancedMoves = 0;
            centerControl = 0;
            edgeControl = 0;
            cornerControl = 0;
            riskTolerance = 0;
            patternRecognition = 0;
            openingPreference = 0;
            endgameSkill = 0;
            style = STYLE_UNKNOWN;
            openingPref = OPENING_UNKNOWN;
        }
        
        // 分析对手走法
        void analyzeMove(const Board& board, const Move& move, Piece opponentColor) {
            // 分析走法的进攻性
            bool isAttack = isAttackMove(board, move, opponentColor);
            bool isDefense = isDefenseMove(board, move, opponentColor);
            
            if (isAttack && !isDefense) {
                attackMoves++;
                riskTolerance++;
            } else if (isDefense && !isAttack) {
                defenseMoves++;
            } else {
                balancedMoves++;
            }
            
            // 分析位置控制
            analyzePositionControl(move);
            
            // 分析棋型识别能力
            analyzePatternRecognition(board, move, opponentColor);
            
            // 分析开局偏好
            if (board.moveCount < 10) {
                analyzeOpeningPreference(move);
            }
            
            // 分析残局技巧
            if (board.moveCount > 40) {
                analyzeEndgameSkill(board, move, opponentColor);
            }
            
            // 更新风格分类
            updateStyle();
        }
        
        // 判断是否为进攻性走法
        bool isAttackMove(const Board& board, const Move& move, Piece opponentColor) {
            // 检查是否形成威胁性棋型
            Board tempBoard = board;
            tempBoard.place(move.p1.x, move.p1.y, opponentColor);
            tempBoard.place(move.p2.x, move.p2.y, opponentColor);
            
            Evaluator evaluator;
            
            // 检查是否形成活四、活三等进攻性棋型
            for (int dir = 0; dir < 4; dir++) {
                auto info1 = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, opponentColor);
                auto info2 = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, opponentColor);
                
                if (info1.type == Evaluator::PATTERN_LIVE_FOUR || info1.type == Evaluator::PATTERN_LIVE_THREE ||
                    info2.type == Evaluator::PATTERN_LIVE_FOUR || info2.type == Evaluator::PATTERN_LIVE_THREE) {
                    return true;
                }
            }
            
            return false;
        }
        
        // 判断是否为防守性走法
        bool isDefenseMove(const Board& board, const Move& move, Piece opponentColor) {
            Piece myColor = (opponentColor == BLACK) ? WHITE : BLACK;
            
            // 检查是否阻挡我方的威胁
            Board tempBoard = board;
            tempBoard.place(move.p1.x, move.p1.y, myColor);
            tempBoard.place(move.p2.x, move.p2.y, myColor);
            
            Evaluator evaluator;
            
            // 检查是否阻挡活四、活三等威胁
            for (int dir = 0; dir < 4; dir++) {
                auto info1 = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, myColor);
                auto info2 = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, myColor);
                
                if (info1.type == Evaluator::PATTERN_LIVE_FOUR || info1.type == Evaluator::PATTERN_LIVE_THREE ||
                    info2.type == Evaluator::PATTERN_LIVE_FOUR || info2.type == Evaluator::PATTERN_LIVE_THREE) {
                    return true;
                }
            }
            
            return false;
        }
        
        // 分析位置控制
        void analyzePositionControl(const Move& move) {
            int centerX = BOARD_SIZE / 2;
            int centerY = BOARD_SIZE / 2;
            
            // 分析第一个点
            int dist1 = abs(move.p1.x - centerX) + abs(move.p1.y - centerY);
            if (dist1 <= 3) {
                centerControl++;
            } else if (move.p1.x <= 2 || move.p1.x >= BOARD_SIZE - 3 || 
                       move.p1.y <= 2 || move.p1.y >= BOARD_SIZE - 3) {
                edgeControl++;
            } else if ((move.p1.x <= 1 || move.p1.x >= BOARD_SIZE - 2) && 
                       (move.p1.y <= 1 || move.p1.y >= BOARD_SIZE - 2)) {
                cornerControl++;
            }
            
            // 分析第二个点
            int dist2 = abs(move.p2.x - centerX) + abs(move.p2.y - centerY);
            if (dist2 <= 3) {
                centerControl++;
            } else if (move.p2.x <= 2 || move.p2.x >= BOARD_SIZE - 3 || 
                       move.p2.y <= 2 || move.p2.y >= BOARD_SIZE - 3) {
                edgeControl++;
            } else if ((move.p2.x <= 1 || move.p2.x >= BOARD_SIZE - 2) && 
                       (move.p2.y <= 1 || move.p2.y >= BOARD_SIZE - 2)) {
                cornerControl++;
            }
        }
        
        // 分析棋型识别能力
        void analyzePatternRecognition(const Board& board, const Move& move, Piece opponentColor) {
            Board tempBoard = board;
            tempBoard.place(move.p1.x, move.p1.y, opponentColor);
            tempBoard.place(move.p2.x, move.p2.y, opponentColor);
            
            Evaluator evaluator;
            int patternScore = 0;
            
            // 检查是否识别并利用高级棋型
            for (int dir = 0; dir < 4; dir++) {
                auto info1 = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, opponentColor);
                auto info2 = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, opponentColor);
                
                if (info1.type == Evaluator::PATTERN_LIVE_FOUR || info1.type == Evaluator::PATTERN_LIVE_THREE ||
                    info2.type == Evaluator::PATTERN_LIVE_FOUR || info2.type == Evaluator::PATTERN_LIVE_THREE) {
                    patternRecognition += 2;
                } else if (info1.type == Evaluator::PATTERN_DEAD_FOUR || info1.type == Evaluator::PATTERN_DEAD_THREE ||
                           info2.type == Evaluator::PATTERN_DEAD_FOUR || info2.type == Evaluator::PATTERN_DEAD_THREE) {
                    patternRecognition += 1;
                }
            }
        }
        
        // 分析开局偏好
        void analyzeOpeningPreference(const Move& move) {
            int centerX = BOARD_SIZE / 2;
            int centerY = BOARD_SIZE / 2;
            
            // 分析第一个点
            int dist1 = abs(move.p1.x - centerX) + abs(move.p1.y - centerY);
            // 分析第二个点
            int dist2 = abs(move.p2.x - centerX) + abs(move.p2.y - centerY);
            
            if (dist1 <= 2 && dist2 <= 2) {
                openingPreference += 5; // 中心开局
                openingPref = OPENING_CENTER;
            } else if ((abs(move.p1.x - move.p1.y) <= 1 && abs(move.p2.x - move.p2.y) <= 1) ||
                       (abs(move.p1.x + move.p1.y - 18) <= 1 && abs(move.p2.x + move.p2.y - 18) <= 1)) {
                openingPreference += 3; // 对角线开局
                openingPref = OPENING_DIAGONAL;
            } else if ((dist1 >= 6 && dist1 <= 9) && (dist2 >= 6 && dist2 <= 9)) {
                openingPreference += 3; // 星位开局
                openingPref = OPENING_STAR;
            } else if (move.p1.x <= 2 || move.p1.x >= 16 || move.p1.y <= 2 || move.p1.y >= 16 ||
                       move.p2.x <= 2 || move.p2.x >= 16 || move.p2.y <= 2 || move.p2.y >= 16) {
                openingPreference += 2; // 边缘开局
                openingPref = OPENING_EDGE;
            } else if ((move.p1.x <= 1 && move.p1.y <= 1) || (move.p1.x >= 17 && move.p1.y >= 17) ||
                       (move.p2.x <= 1 && move.p2.y <= 1) || (move.p2.x >= 17 && move.p2.y >= 17)) {
                openingPreference += 1; // 角落开局
                openingPref = OPENING_CORNER;
            } else {
                openingPreference += 2; // 混合型开局
                openingPref = OPENING_MIXED;
            }
        }
        
        // 分析残局技巧
        void analyzeEndgameSkill(const Board& board, const Move& move, Piece opponentColor) {
            Board tempBoard = board;
            tempBoard.place(move.p1.x, move.p1.y, opponentColor);
            tempBoard.place(move.p2.x, move.p2.y, opponentColor);
            
            Evaluator evaluator;
            int endgameScore = 0;
            
            // 检查是否在残局中做出正确的决策
            for (int dir = 0; dir < 4; dir++) {
                auto info1 = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, opponentColor);
                auto info2 = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, opponentColor);
                
                if (info1.type == Evaluator::PATTERN_SIX || info2.type == Evaluator::PATTERN_SIX) {
                    endgameSkill += 3; // 直接获胜
                } else if (info1.type == Evaluator::PATTERN_LIVE_FIVE || info2.type == Evaluator::PATTERN_LIVE_FIVE) {
                    endgameSkill += 2; // 活五
                } else if (info1.type == Evaluator::PATTERN_LIVE_FOUR || info2.type == Evaluator::PATTERN_LIVE_FOUR) {
                    endgameSkill += 2; // 活四
                } else if (info1.type == Evaluator::PATTERN_DEAD_FOUR || info2.type == Evaluator::PATTERN_DEAD_FOUR) {
                    endgameSkill += 1; // 死四
                }
            }
        }
        
        // 更新风格分类
        void updateStyle() {
            int totalMoves = attackMoves + defenseMoves + balancedMoves;
            if (totalMoves < 5) {
                style = STYLE_UNKNOWN;
                return;
            }
            
            double attackRatio = (double)attackMoves / totalMoves;
            double defenseRatio = (double)defenseMoves / totalMoves;
            double centerRatio = (double)centerControl / (centerControl + edgeControl + cornerControl);
            
            if (attackRatio > 0.6) {
                if (patternRecognition > totalMoves * 1.5) {
                    style = STYLE_TACTICAL; // 战术型
                } else {
                    style = STYLE_AGGRESSIVE; // 进攻型
                }
            } else if (defenseRatio > 0.6) {
                style = STYLE_DEFENSIVE; // 防守型
            } else if (centerRatio > 0.6) {
                style = STYLE_POSITIONAL; // 位置型
            } else {
                style = STYLE_BALANCED; // 平衡型
            }
        }
        
        // 获取对手风格
        StyleType getStyle() {
            return style;
        }
        
        // 获取开局偏好
        OpeningPreference getOpeningPreference() {
            return openingPref;
        }
        
        // 根据对手风格调整评估权重
        void adjustWeights(double& attackWeight, double& defenseWeight, double& positionWeight) {
            switch (style) {
                case STYLE_AGGRESSIVE:
                    // 对进攻型对手，加强防守
                    attackWeight = 0.8;
                    defenseWeight = 1.2;
                    positionWeight = 1.0;
                    break;
                case STYLE_DEFENSIVE:
                    // 对防守型对手，加强进攻
                    attackWeight = 1.2;
                    defenseWeight = 0.8;
                    positionWeight = 1.0;
                    break;
                case STYLE_BALANCED:
                    // 对平衡型对手，保持平衡
                    attackWeight = 1.0;
                    defenseWeight = 1.0;
                    positionWeight = 1.0;
                    break;
                case STYLE_POSITIONAL:
                    // 对位置型对手，加强位置控制
                    attackWeight = 0.9;
                    defenseWeight = 0.9;
                    positionWeight = 1.2;
                    break;
                case STYLE_TACTICAL:
                    // 对战术型对手，加强战术应对
                    attackWeight = 1.1;
                    defenseWeight = 1.1;
                    positionWeight = 0.9;
                    break;
                default:
                    // 未知风格，保持默认
                    attackWeight = 1.0;
                    defenseWeight = 1.0;
                    positionWeight = 1.0;
                    break;
            }
        }
        
        // 根据开局偏好调整策略
        void adjustOpeningStrategy(double& centerWeight, double& edgeWeight, double& cornerWeight) {
            switch (openingPref) {
                case OPENING_CENTER:
                    centerWeight = 1.2;
                    edgeWeight = 0.9;
                    cornerWeight = 0.8;
                    break;
                case OPENING_DIAGONAL:
                    centerWeight = 1.0;
                    edgeWeight = 1.0;
                    cornerWeight = 0.9;
                    break;
                case OPENING_STAR:
                    centerWeight = 0.9;
                    edgeWeight = 1.1;
                    cornerWeight = 0.9;
                    break;
                case OPENING_EDGE:
                    centerWeight = 0.8;
                    edgeWeight = 1.2;
                    cornerWeight = 1.0;
                    break;
                case OPENING_CORNER:
                    centerWeight = 0.8;
                    edgeWeight = 1.0;
                    cornerWeight = 1.2;
                    break;
                case OPENING_MIXED:
                    centerWeight = 1.0;
                    edgeWeight = 1.0;
                    cornerWeight = 1.0;
                    break;
                default:
                    centerWeight = 1.0;
                    edgeWeight = 1.0;
                    cornerWeight = 1.0;
                    break;
            }
        }
    };
    
    // 对手模型
    OpponentModel opponentModel;
    
    SearchEngine() {
        memset(historyTable, 0, sizeof(historyTable));
        timeLimitMs = TIME_LIMIT_MS;
        ttAge = 0;
        parallelSearchEnabled = false;
        threadCount = 0;
        parallelIndex = 0;
        parallelSearchDone = false;
        poolRunning = false;
        
        // 初始化置换表
        transpositionTable = new TTEntry[TT_SIZE];
        for (int i = 0; i < TT_SIZE; i++) {
            transpositionTable[i].key = 0;
            transpositionTable[i].depth = 0;
            transpositionTable[i].age = 0;
        }
        
        // 启用并行搜索，根据CPU核心数创建线程
        threadCount = thread::hardware_concurrency();
        if (threadCount > 1) {
            parallelSearchEnabled = true;
            // 最多使用4个线程，避免过度竞争
            threadCount = min(threadCount, 4);
            // 初始化线程池
            initThreadPool();
        }
        
        // 初始化开局库
        initOpeningBook();
        // 初始化残局库
        initEndgameBook();
        
        // 初始化对手模型
        opponentModel = OpponentModel();
    }
    
    ~SearchEngine() {
        if (poolRunning) {
            shutdownThreadPool();
        }
        delete[] transpositionTable;
    }
    
    void setTimeLimit(int ms) {
        timeLimitMs = ms;
    }
    
    void setEvaluatorType(bool useML) {
        useMLEvaluator = useML;
    }
    
    // Zobrist哈希
    uint64_t zobristTable[BOARD_SIZE][BOARD_SIZE][3];
    uint64_t zobristInitialized = false;
    
    void initZobrist() {
        if (zobristInitialized) return;
        mt19937_64 rng(123456789);
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                for (int k = 0; k < 3; k++) {
                    zobristTable[i][j][k] = rng();
                }
            }
        }
        zobristInitialized = true;
    }
    
    uint64_t computeHash(const Board& b) {
        uint64_t hash = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (b.get(i, j) != EMPTY) {
                    hash ^= zobristTable[i][j][b.get(i, j)];
                }
            }
        }
        return hash;
    }
    
    // 检查时间
    bool checkTime() {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - startTime).count();
        if (elapsed >= timeLimitMs) {
            timeUp = true;
            return true;
        }
        return false;
    }
    
    // 计算局面复杂度
    int calculateComplexity(const Board& board) {
        int complexity = 0;
        
        // 计算已有棋子数量
        int pieceCount = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    pieceCount++;
                }
            }
        }
        
        // 棋子越多，局面越复杂
        complexity += pieceCount * 10;
        
        // 计算活跃区域
        int activeCount = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    // 检查周围2格范围内的空位
                    for (int dx = -2; dx <= 2; dx++) {
                        for (int dy = -2; dy <= 2; dy++) {
                            int nx = i + dx, ny = j + dy;
                            if (board.isValid(nx, ny) && board.isEmpty(nx, ny)) {
                                activeCount++;
                            }
                        }
                    }
                }
            }
        }
        
        complexity += activeCount * 5;
        
        return complexity;
    }
    
    // 历史搜索时间记录
    vector<int> searchTimeHistory; // 记录最近的搜索时间
    
    // 动态时间分配
    int calculateTimeLimit(int remainingTime, int moveNumber, const Board& board) {
        // 基础时间分配
        int baseTime;
        
        // 根据剩余时间和回合数调整
        if (remainingTime > 45000) { // 剩余时间非常充足
            baseTime = remainingTime / (40 - moveNumber / 2); // 假设平均40回合
        } else if (remainingTime > 30000) { // 剩余时间充足
            baseTime = remainingTime / (35 - moveNumber / 2); // 假设平均35回合
        } else if (remainingTime > 15000) { // 剩余时间中等
            baseTime = remainingTime / (30 - moveNumber / 2); // 假设平均30回合
        } else if (remainingTime > 5000) { // 剩余时间紧张
            baseTime = remainingTime / (25 - moveNumber / 2); // 假设平均25回合
        } else { // 剩余时间危急
            baseTime = remainingTime / 10; // 假设还有10回合
        }
        
        int complexity = calculateComplexity(board);
        
        // 根据复杂度调整时间
        if (complexity > 800) {
            baseTime = min(static_cast<int>(baseTime * 2.5), static_cast<int>(remainingTime * 0.75)); // 非常复杂局面分配更多时间
        } else if (complexity > 600) {
            baseTime = min(static_cast<int>(baseTime * 2), static_cast<int>(remainingTime * 0.7)); // 复杂局面分配较多时间
        } else if (complexity > 400) {
            baseTime = min(static_cast<int>(baseTime * 1.5), static_cast<int>(remainingTime * 0.6)); // 中等复杂局面分配适量时间
        } else if (complexity < 200) {
            baseTime = max(baseTime / 2, 600); // 简单局面分配较少时间
        }
        
        // 根据局面评估调整时间
        int evaluation = useMLEvaluator ? mlEvaluator.evaluate(board, myPiece) : evaluator.evaluate(board, myPiece);
        if (abs(evaluation) > 80000) {
            // 优势或劣势非常明显，大幅减少思考时间
            baseTime = max(baseTime / 3, 600);
        } else if (abs(evaluation) > 50000) {
            // 优势或劣势明显，减少思考时间
            baseTime = max(baseTime / 2, 600);
        } else if (abs(evaluation) < 5000) {
            // 局面非常接近，需要更多思考时间
            baseTime = min(static_cast<int>(baseTime * 1.5), static_cast<int>(remainingTime * 0.7));
        } else if (abs(evaluation) < 10000) {
            // 局面接近，需要更多思考时间
            baseTime = min(static_cast<int>(baseTime * 1.3), static_cast<int>(remainingTime * 0.6));
        }
        
        // 根据对手风格调整时间
        double timeMultiplier = 1.0;
        switch (opponentModel.getStyle()) {
            case OpponentModel::STYLE_AGGRESSIVE:
                // 对进攻型对手，需要更多时间思考
                timeMultiplier = 1.2;
                break;
            case OpponentModel::STYLE_DEFENSIVE:
                // 对防守型对手，可以减少思考时间
                timeMultiplier = 0.8;
                break;
            default:
                timeMultiplier = 1.0;
                break;
        }
        baseTime = static_cast<int>(baseTime * timeMultiplier);
        
        // 根据历史搜索时间调整
        if (!searchTimeHistory.empty()) {
            // 计算平均搜索时间
            int avgTime = 0;
            for (int time : searchTimeHistory) {
                avgTime += time;
            }
            avgTime /= searchTimeHistory.size();
            
            // 根据历史时间调整当前时间分配
            if (avgTime > baseTime * 1.5) {
                // 历史搜索时间较长，增加当前时间分配
                baseTime = min(static_cast<int>(baseTime * 1.3), static_cast<int>(remainingTime * 0.8));
            } else if (avgTime < baseTime * 0.5) {
                // 历史搜索时间较短，减少当前时间分配
                baseTime = max(static_cast<int>(baseTime * 0.8), 600);
            }
        }
        
        // 确保时间限制在合理范围内
        int finalTimeLimit = min(max(baseTime, 600), static_cast<int>(remainingTime * 0.85)); // 最少0.6秒，最多剩余时间的85%
        
        return finalTimeLimit;
    }
    
    // 记录搜索时间
    void recordSearchTime(int timeUsed) {
        searchTimeHistory.push_back(timeUsed);
        // 只保留最近10次的搜索时间
        if (searchTimeHistory.size() > 10) {
            searchTimeHistory.erase(searchTimeHistory.begin());
        }
    }
    
    // 计算自适应搜索深度
    int calculateAdaptiveDepth(int timeLimit, const Board& board) {
        int complexity = calculateComplexity(board);
        int baseDepth = 4;
        
        // 根据时间限制调整深度
        if (timeLimit >= 5000) {
            baseDepth = 6;
        } else if (timeLimit >= 3000) {
            baseDepth = 5;
        } else if (timeLimit >= 1000) {
            baseDepth = 4;
        } else {
            baseDepth = 3;
        }
        
        // 根据复杂度调整深度
        if (complexity > 800) {
            // 非常复杂局面，减少深度
            baseDepth = max(baseDepth - 1, 3);
        } else if (complexity < 200) {
            // 简单局面，增加深度
            baseDepth = min(baseDepth + 1, 7);
        }
        
        // 根据局面评估调整深度
        int evaluation = useMLEvaluator ? mlEvaluator.evaluate(board, myPiece) : evaluator.evaluate(board, myPiece);
        if (abs(evaluation) < 5000) {
            // 局面接近，增加深度
            baseDepth = min(baseDepth + 1, 7);
        }
        
        return baseDepth;
    }
    
    // PVS（Principal Variation Search）
    int pvs(Board& board, int depth, int alpha, int beta, Piece currentPlayer, 
            int lastX, int lastY, Move& bestMove) {
        return negascout(board, depth, alpha, beta, currentPlayer, lastX, lastY, bestMove);
    }
    
    // NegaScout搜索算法
    int negascout(Board& board, int depth, int alpha, int beta, Piece currentPlayer, 
                 int lastX, int lastY, Move& bestMove) {
        nodesSearched++;
        
        // 检查时间
        if ((nodesSearched & 1023) == 0) {  // 每1024个节点检查一次
            if (checkTime()) {
                return 0;
            }
        }
        
        // 检查游戏结束
        if (lastX >= 0 && board.checkWin(lastX, lastY, currentPlayer == myPiece ? opponentPiece : myPiece)) {
            return (currentPlayer == myPiece) ? -SCORE_SIX : SCORE_SIX;
        }
        
        // 达到深度限制
        if (depth <= 0) {
            return useMLEvaluator ? mlEvaluator.evaluate(board, myPiece) : evaluator.evaluate(board, myPiece);
        }
        
        // 查找置换表
        uint64_t hashKey = computeHash(board);
        int ttIndex = hashKey % TT_SIZE;
        TTEntry& entry = transpositionTable[ttIndex];
        
        if (entry.key == hashKey && entry.depth >= depth) {
            // 置换表命中
            if (entry.flag == 0) {  // 精确值
                bestMove = entry.bestMove;
                return entry.score;
            } else if (entry.flag == 1 && entry.score >= beta) {  // 下界
                return entry.score;
            } else if (entry.flag == 2 && entry.score <= alpha) {  // 上界
                return entry.score;
            }
        }
        
        // 如果置换表中有最佳走法，优先考虑
        Move ttBestMove = entry.bestMove;
        
        // 生成走法
        vector<Move> moves = moveGenerator.generateMoves(board, currentPlayer, evaluator, 15);
        
        if (moves.empty()) {
            return useMLEvaluator ? mlEvaluator.evaluate(board, myPiece) : evaluator.evaluate(board, myPiece);
        }
        
        // 对走法排序 (使用历史启发和杀手走法)
        sortMoves(moves, depth);
        
        // 如果置换表中有最佳走法，将其移到最前面
        if (ttBestMove.p1.x >= 0) {
            for (size_t i = 0; i < moves.size(); i++) {
                if ((moves[i].p1.x == ttBestMove.p1.x && moves[i].p1.y == ttBestMove.p1.y &&
                     moves[i].p2.x == ttBestMove.p2.x && moves[i].p2.y == ttBestMove.p2.y) ||
                    (moves[i].p1.x == ttBestMove.p2.x && moves[i].p1.y == ttBestMove.p2.y &&
                     moves[i].p2.x == ttBestMove.p1.x && moves[i].p2.y == ttBestMove.p1.y)) {
                    // 找到置换表中的最佳走法，移到最前面
                    iter_swap(moves.begin(), moves.begin() + i);
                    break;
                }
            }
        }
        
        int bestScore = -INT_MAX;
        Move localBestMove;
        bool isFirstMove = true;
        
        for (size_t i = 0; i < moves.size() && !timeUp; i++) {
            const auto& move = moves[i];
            
            // 执行走法
            board.place(move.p1.x, move.p1.y, currentPlayer);
            board.place(move.p2.x, move.p2.y, currentPlayer);
            
            // 检查是否获胜
            bool isWin = board.checkWin(move.p1.x, move.p1.y, currentPlayer) || 
                        board.checkWin(move.p2.x, move.p2.y, currentPlayer);
            
            int score;
            if (isWin) {
                score = (currentPlayer == myPiece) ? SCORE_SIX : -SCORE_SIX;
            } else {
                // 递归搜索
                Move childBestMove;
                if (isFirstMove) {
                    // 主变例，使用完整Alpha-Beta
                    score = -negascout(board, depth - 1, -beta, -alpha, 
                        (currentPlayer == BLACK) ? WHITE : BLACK, move.p2.x, move.p2.y, childBestMove);
                } else {
                    // NegaScout：首先使用零窗搜索
                    score = -negascout(board, depth - 1, -alpha - 1, -alpha, 
                        (currentPlayer == BLACK) ? WHITE : BLACK, move.p2.x, move.p2.y, childBestMove);
                    
                    // 零窗失败，使用完整窗口重新搜索
                    if (score > alpha && score < beta) {
                        score = -negascout(board, depth - 1, -beta, -score, 
                            (currentPlayer == BLACK) ? WHITE : BLACK, move.p2.x, move.p2.y, childBestMove);
                    }
                }
            }
            
            // 撤销走法
            board.remove(move.p1.x, move.p1.y);
            board.remove(move.p2.x, move.p2.y);
            
            if (timeUp) return 0;
            
            if (score > bestScore) {
                bestScore = score;
                localBestMove = move;
                
                if (score > alpha) {
                    alpha = score;
                    
                    // Alpha-Beta剪枝
                    if (alpha >= beta) {
                        cutoffs++;
                        // 更新杀手走法
                        if (killerMoves[depth][0].p1.x != move.p1.x || 
                            killerMoves[depth][0].p1.y != move.p1.y ||
                            killerMoves[depth][0].p2.x != move.p2.x || 
                            killerMoves[depth][0].p2.y != move.p2.y) {
                            killerMoves[depth][1] = killerMoves[depth][0];
                            killerMoves[depth][0] = move;
                        }
                        // 更新历史表
                        historyTable[move.p1.x][move.p1.y] += depth * depth;
                        historyTable[move.p2.x][move.p2.y] += depth * depth;
                        break;
                    }
                }
            }
            
            isFirstMove = false;
        }
        
        bestMove = localBestMove;
        
        // 存储到置换表
        int flag = 0;
        if (bestScore >= beta) {
            flag = 1;  // 下界
        } else if (bestScore <= alpha) {
            flag = 2;  // 上界
        } else {
            flag = 0;  // 精确值
        }
        
        // 替换策略：如果当前条目为空，或者当前深度大于等于条目的深度，或者条目是旧的，则替换
        if (entry.key == 0 || depth >= entry.depth || entry.age < ttAge - 10) {
            entry.key = hashKey;
            entry.depth = depth;
            entry.score = bestScore;
            entry.flag = flag;
            entry.bestMove = bestMove;
            entry.age = ttAge;
        }
        
        return bestScore;
    }
    
    // Alpha-Beta搜索（保留为兼容函数）
    int alphaBeta(Board& board, int depth, int alpha, int beta, Piece currentPlayer, 
                  int lastX, int lastY, Move& bestMove) {
        return pvs(board, depth, alpha, beta, currentPlayer, lastX, lastY, bestMove);
    }
    
    // PNS（Proof-Number Search）节点结构
    struct PNSNode {
        Board board;
        Piece player;
        Move move; // 到达此节点的走法
        int proof; // 证明数
        int disproof; // 反驳数
        vector<PNSNode*> children;
        PNSNode* bestChild; // 最佳子节点
        bool solved; // 是否已解决
        bool winning; // 是否为获胜节点
        
        PNSNode(const Board& b, Piece p, const Move& m) : 
            board(b), player(p), move(m), 
            proof(1), disproof(1), bestChild(nullptr), 
            solved(false), winning(false) {}
    };
    
    // Proof-Number Search
    bool pns(PNSNode* node, Move& bestMove) {
        // 检查游戏结束
        if (node->board.getWinner() == myPiece) {
            node->proof = 0;
            node->disproof = INT_MAX;
            node->solved = true;
            node->winning = true;
            return true;
        }
        if (node->board.getWinner() == opponentPiece) {
            node->proof = INT_MAX;
            node->disproof = 0;
            node->solved = true;
            node->winning = false;
            return false;
        }
        
        // 检查时间
        if (checkTime()) {
            return false;
        }
        
        // 生成子节点
        if (node->children.empty()) {
            vector<Move> moves = moveGenerator.generateMoves(node->board, node->player, evaluator, 15);
            
            // 对走法进行排序，优先考虑有威胁的走法
            sort(moves.begin(), moves.end(), [this, &node](const Move& a, const Move& b) {
                Board tempA = node->board;
                tempA.place(a.p1.x, a.p1.y, node->player);
                tempA.place(a.p2.x, a.p2.y, node->player);
                
                Board tempB = node->board;
                tempB.place(b.p1.x, b.p1.y, node->player);
                tempB.place(b.p2.x, b.p2.y, node->player);
                
                int scoreA = evaluator.evaluate(tempA, myPiece);
                int scoreB = evaluator.evaluate(tempB, myPiece);
                
                return scoreA > scoreB;
            });
            
            for (const auto& move : moves) {
                Board tempBoard = node->board;
                tempBoard.place(move.p1.x, move.p1.y, node->player);
                tempBoard.place(move.p2.x, move.p2.y, node->player);
                node->children.push_back(new PNSNode(tempBoard, 
                    (node->player == BLACK) ? WHITE : BLACK, move));
            }
            
            if (node->children.empty()) {
                // 没有可走的位置，平局
                node->proof = INT_MAX;
                node->disproof = INT_MAX;
                node->solved = true;
                node->winning = false;
                return false;
            }
        }
        
        // 选择最有希望的子节点
        PNSNode* selectedChild = nullptr;
        int minValue = INT_MAX;
        
        for (auto child : node->children) {
            if (!child->solved) {
                int value = max(child->proof, child->disproof);
                if (value < minValue) {
                    minValue = value;
                    selectedChild = child;
                }
            }
        }
        
        if (!selectedChild) {
            // 所有子节点都已解决
            int minProof = INT_MAX;
            int sumDisproof = 0;
            bool hasWinningChild = false;
            
            for (auto child : node->children) {
                if (child->winning) {
                    hasWinningChild = true;
                    minProof = min(minProof, child->proof);
                } else {
                    sumDisproof += child->disproof;
                }
            }
            
            if (hasWinningChild) {
                node->proof = minProof;
                node->disproof = INT_MAX;
                node->solved = true;
                node->winning = true;
                
                // 找到最佳子节点
                for (auto child : node->children) {
                    if (child->winning && child->proof == minProof) {
                        node->bestChild = child;
                        break;
                    }
                }
                
                bestMove = node->bestChild->move;
                return true;
            } else {
                node->proof = INT_MAX;
                node->disproof = sumDisproof;
                node->solved = true;
                node->winning = false;
                return false;
            }
        }
        
        // 递归搜索子节点
        Move childBestMove;
        bool childWinning = pns(selectedChild, childBestMove);
        
        // 剪枝：如果当前节点已经找到获胜路径，提前返回
        if (childWinning && node->player == myPiece) {
            node->proof = 1;
            node->disproof = INT_MAX;
            node->solved = true;
            node->winning = true;
            node->bestChild = selectedChild;
            bestMove = selectedChild->move;
            return true;
        }
        
        // 更新当前节点的证明数和反驳数
        int minProof = INT_MAX;
        int sumDisproof = 0;
        bool hasWinningChild = false;
        
        for (auto child : node->children) {
            if (child->winning) {
                hasWinningChild = true;
                minProof = min(minProof, child->proof);
            } else {
                sumDisproof += child->disproof;
            }
        }
        
        if (hasWinningChild) {
            node->proof = minProof;
            node->disproof = INT_MAX;
            node->solved = true;
            node->winning = true;
            
            // 找到最佳子节点
            for (auto child : node->children) {
                if (child->winning && child->proof == minProof) {
                    node->bestChild = child;
                    break;
                }
            }
            
            bestMove = node->bestChild->move;
            return true;
        } else {
            node->proof = INT_MAX;
            node->disproof = sumDisproof;
            node->solved = true;
            node->winning = false;
            return false;
        }
    }
    
    // 使用PNS搜索最佳走法
    Move searchWithPNS(Board& board, Piece player) {
        Move bestMove;
        PNSNode* root = new PNSNode(board, player, Move());
        pns(root, bestMove);
        delete root;
        return bestMove;
    }
    
    // 对走法排序
    void sortMoves(vector<Move>& moves, int depth) {
        for (auto& move : moves) {
            // 历史启发分数
            move.score += historyTable[move.p1.x][move.p1.y] + historyTable[move.p2.x][move.p2.y];
            
            // 杀手走法加分
            if ((move.p1.x == killerMoves[depth][0].p1.x && move.p1.y == killerMoves[depth][0].p1.y &&
                 move.p2.x == killerMoves[depth][0].p2.x && move.p2.y == killerMoves[depth][0].p2.y) ||
                (move.p1.x == killerMoves[depth][0].p2.x && move.p1.y == killerMoves[depth][0].p2.y &&
                 move.p2.x == killerMoves[depth][0].p1.x && move.p2.y == killerMoves[depth][0].p1.y)) {
                move.score += 1000000;
            }
            if ((move.p1.x == killerMoves[depth][1].p1.x && move.p1.y == killerMoves[depth][1].p1.y &&
                 move.p2.x == killerMoves[depth][1].p2.x && move.p2.y == killerMoves[depth][1].p2.y) ||
                (move.p1.x == killerMoves[depth][1].p2.x && move.p1.y == killerMoves[depth][1].p2.y &&
                 move.p2.x == killerMoves[depth][1].p1.x && move.p2.y == killerMoves[depth][1].p1.y)) {
                move.score += 500000;
            }
            
            // 威胁走法加分
            // 检查是否形成活四或活三
            Board tempBoard;
            tempBoard.copy(board);
            tempBoard.place(move.p1.x, move.p1.y, myPiece);
            tempBoard.place(move.p2.x, move.p2.y, myPiece);
            
            // 检查是否获胜
            if (tempBoard.checkWin(move.p1.x, move.p1.y, myPiece) || 
                tempBoard.checkWin(move.p2.x, move.p2.y, myPiece)) {
                move.score += 2000000; // 获胜走法
            }
            
            // 检查是否形成活四
            for (int dir = 0; dir < 4; dir++) {
                Evaluator::PatternInfo info = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, myPiece);
                if (info.type == Evaluator::PATTERN_LIVE_FOUR) {
                    move.score += 1500000; // 活四走法
                    break;
                }
                info = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, myPiece);
                if (info.type == Evaluator::PATTERN_LIVE_FOUR) {
                    move.score += 1500000; // 活四走法
                    break;
                }
            }
            
            // 检查是否形成活三
            for (int dir = 0; dir < 4; dir++) {
                Evaluator::PatternInfo info = evaluator.recognizePatternEnhanced(tempBoard, move.p1.x, move.p1.y, dir, myPiece);
                if (info.type == Evaluator::PATTERN_LIVE_THREE) {
                    move.score += 1000000; // 活三走法
                    break;
                }
                info = evaluator.recognizePatternEnhanced(tempBoard, move.p2.x, move.p2.y, dir, myPiece);
                if (info.type == Evaluator::PATTERN_LIVE_THREE) {
                    move.score += 1000000; // 活三走法
                    break;
                }
            }
            
            // 防守性走法加分
            // 检查是否能阻挡对方的活四或活三
            Piece opponent = (myPiece == BLACK) ? WHITE : BLACK;
            Board tempBoardOpponent;
            tempBoardOpponent.copy(board);
            tempBoardOpponent.place(move.p1.x, move.p1.y, opponent);
            tempBoardOpponent.place(move.p2.x, move.p2.y, opponent);
            
            bool blocksThreat = false;
            for (int dir = 0; dir < 4; dir++) {
                Evaluator::PatternInfo info = evaluator.recognizePatternEnhanced(tempBoardOpponent, move.p1.x, move.p1.y, dir, opponent);
                if (info.type == Evaluator::PATTERN_LIVE_FOUR || info.type == Evaluator::PATTERN_LIVE_THREE) {
                    blocksThreat = true;
                    break;
                }
                info = evaluator.recognizePatternEnhanced(tempBoardOpponent, move.p2.x, move.p2.y, dir, opponent);
                if (info.type == Evaluator::PATTERN_LIVE_FOUR || info.type == Evaluator::PATTERN_LIVE_THREE) {
                    blocksThreat = true;
                    break;
                }
            }
            
            if (blocksThreat) {
                move.score += 800000; // 防守走法
            }
            
            // 位置价值加分
            int centerX = BOARD_SIZE / 2;
            int centerY = BOARD_SIZE / 2;
            int dist1 = abs(move.p1.x - centerX) + abs(move.p1.y - centerY);
            int dist2 = abs(move.p2.x - centerX) + abs(move.p2.y - centerY);
            move.score += (40 - dist1 - dist2) * 100; // 中心位置加分
        }
        
        sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return a.score > b.score;
        });
    }
    
    // 并行搜索线程函数
    void parallelSearchThread(Board board, Piece player, int depth) {
        while (true) {
            int index = -1;
            {
                unique_lock<mutex> lock(searchMutex);
                if (parallelIndex >= parallelMoves.size() || parallelSearchDone) {
                    return;
                }
                index = parallelIndex++;
            }
            
            // 检查时间
            if (checkTime()) {
                return;
            }
            
            Move move = parallelMoves[index];
            Board tempBoard = board;
            tempBoard.place(move.p1.x, move.p1.y, player);
            tempBoard.place(move.p2.x, move.p2.y, player);
            
            Move dummyMove;
            int score = -pvs(tempBoard, depth - 1, -INT_MAX, INT_MAX, 
                             (player == BLACK) ? WHITE : BLACK, move.p2.x, move.p2.y, dummyMove);
            
            {
                unique_lock<mutex> lock(searchMutex);
                parallelScores[index] = score;
            }
        }
    }
    
    // 初始化开局库
    void initOpeningBook() {
        // 常见的六子棋开局走法
        
        // 1. 中心开局系列
        vector<Position> centerOpening = {Position(9, 9)};
        openingBook["9,9"] = {centerOpening, Move(Position(8, 9), Position(10, 9))}; // 中心对称
        openingBook["9,9,8,9,10,9"] = {{Position(9,9), Position(8,9), Position(10,9)}, Move(Position(9,8), Position(9,10))}; // 十字形
        openingBook["9,9,9,8,9,10"] = {{Position(9,9), Position(9,8), Position(9,10)}, Move(Position(8,9), Position(10,9))}; // 垂直形
        openingBook["9,9,8,8,10,10"] = {{Position(9,9), Position(8,8), Position(10,10)}, Move(Position(8,10), Position(10,8))}; // 菱形
        openingBook["9,9,8,10,10,8"] = {{Position(9,9), Position(8,10), Position(10,8)}, Move(Position(8,8), Position(10,10))}; // 反菱形
        openingBook["9,9,8,9,9,8"] = {{Position(9,9), Position(8,9), Position(9,8)}, Move(Position(10,9), Position(9,10))}; // 小十字
        openingBook["9,9,8,9,10,8"] = {{Position(9,9), Position(8,9), Position(10,8)}, Move(Position(9,7), Position(9,10))}; // 斜十字
        openingBook["9,9,7,9,11,9"] = {{Position(9,9), Position(7,9), Position(11,9)}, Move(Position(8,9), Position(10,9))}; // 长水平
        openingBook["9,9,9,7,9,11"] = {{Position(9,9), Position(9,7), Position(9,11)}, Move(Position(9,8), Position(9,10))}; // 长垂直
        
        // 2. 对角线开局系列
        vector<Position> diagonalOpening = {Position(9, 9), Position(8, 8), Position(10, 10)};
        openingBook["9,9,8,8,10,10"] = {diagonalOpening, Move(Position(7, 7), Position(11, 11))}; // 对角线延伸
        openingBook["9,9,8,8,11,11"] = {{Position(9,9), Position(8,8), Position(11,11)}, Move(Position(7,7), Position(10,10))}; // 对角线变体
        openingBook["9,9,7,7,11,11"] = {{Position(9,9), Position(7,7), Position(11,11)}, Move(Position(6,6), Position(12,12))}; // 长对角线
        openingBook["9,9,8,8,7,7"] = {{Position(9,9), Position(8,8), Position(7,7)}, Move(Position(10,10), Position(11,11))}; // 单对角线
        openingBook["9,9,8,8,9,10"] = {{Position(9,9), Position(8,8), Position(9,10)}, Move(Position(10,8), Position(11,11))}; // 对角混合
        openingBook["9,9,8,8,10,8"] = {{Position(9,9), Position(8,8), Position(10,8)}, Move(Position(7,7), Position(9,10))}; // 对角边
        
        // 3. 星位开局系列
        vector<Position> starOpening = {Position(9, 9), Position(3, 3), Position(15, 15)};
        openingBook["9,9,3,3,15,15"] = {starOpening, Move(Position(3, 15), Position(15, 3))}; // 对角星位
        openingBook["9,9,3,15,15,3"] = {{Position(9,9), Position(3,15), Position(15,3)}, Move(Position(3,3), Position(15,15))}; // 反对角星位
        openingBook["9,9,3,9,15,9"] = {{Position(9,9), Position(3,9), Position(15,9)}, Move(Position(9,3), Position(9,15))}; // 水平星位
        openingBook["9,9,9,3,9,15"] = {{Position(9,9), Position(9,3), Position(9,15)}, Move(Position(3,9), Position(15,9))}; // 垂直星位
        openingBook["9,9,3,3,9,15"] = {{Position(9,9), Position(3,3), Position(9,15)}, Move(Position(15,15), Position(15,3))}; // 混合星位
        openingBook["9,9,3,15,9,3"] = {{Position(9,9), Position(3,15), Position(9,3)}, Move(Position(15,3), Position(15,15))}; // 交叉星位
        
        // 4. 边缘开局系列
        vector<Position> edgeOpening = {Position(9, 9), Position(0, 9), Position(18, 9)};
        openingBook["9,9,0,9,18,9"] = {edgeOpening, Move(Position(9,0), Position(9,18))}; // 水平边缘
        openingBook["9,9,9,0,9,18"] = {{Position(9,9), Position(9,0), Position(9,18)}, Move(Position(0,9), Position(18,9))}; // 垂直边缘
        openingBook["9,9,0,0,18,18"] = {{Position(9,9), Position(0,0), Position(18,18)}, Move(Position(0,18), Position(18,0))}; // 对角边缘
        openingBook["9,9,0,18,18,0"] = {{Position(9,9), Position(0,18), Position(18,0)}, Move(Position(0,0), Position(18,18))}; // 反对角边缘
        openingBook["9,9,0,9,9,0"] = {{Position(9,9), Position(0,9), Position(9,0)}, Move(Position(18,9), Position(9,18))}; // 角落边缘
        openingBook["9,9,18,9,9,18"] = {{Position(9,9), Position(18,9), Position(9,18)}, Move(Position(0,9), Position(9,0))}; // 对角边缘
        
        // 5. 角落开局系列
        vector<Position> cornerOpening = {Position(9, 9), Position(0, 0), Position(18, 18)};
        openingBook["9,9,0,0,18,18"] = {cornerOpening, Move(Position(0, 18), Position(18, 0))}; // 对角角落
        openingBook["9,9,0,18,18,0"] = {{Position(9,9), Position(0,18), Position(18,0)}, Move(Position(0,0), Position(18,18))}; // 反对角角落
        openingBook["9,9,0,0,0,18"] = {{Position(9,9), Position(0,0), Position(0,18)}, Move(Position(18,0), Position(18,18))}; // 左侧角落
        openingBook["9,9,18,0,18,18"] = {{Position(9,9), Position(18,0), Position(18,18)}, Move(Position(0,0), Position(0,18))}; // 右侧角落
        openingBook["9,9,0,0,1,1"] = {{Position(9,9), Position(0,0), Position(1,1)}, Move(Position(18,18), Position(17,17))}; // 小角落
        openingBook["9,9,18,18,17,17"] = {{Position(9,9), Position(18,18), Position(17,17)}, Move(Position(0,0), Position(1,1))}; // 大角落
        
        // 6. 混合型开局
        vector<Position> mixedOpening = {Position(9, 9), Position(3, 9), Position(15, 9)};
        openingBook["9,9,3,9,15,9"] = {mixedOpening, Move(Position(9,3), Position(9,15))}; // 十字星位
        openingBook["9,9,3,3,9,15"] = {{Position(9,9), Position(3,3), Position(9,15)}, Move(Position(15,15), Position(15,3))}; // 混合星位
        openingBook["9,9,3,15,15,15"] = {{Position(9,9), Position(3,15), Position(15,15)}, Move(Position(3,3), Position(15,3))}; // 底边混合
        openingBook["9,9,3,3,15,3"] = {{Position(9,9), Position(3,3), Position(15,3)}, Move(Position(3,15), Position(15,15))}; // 顶边混合
        openingBook["9,9,3,9,9,3"] = {{Position(9,9), Position(3,9), Position(9,3)}, Move(Position(15,9), Position(9,15))}; // 十字交叉
        openingBook["9,9,3,15,9,3"] = {{Position(9,9), Position(3,15), Position(9,3)}, Move(Position(15,3), Position(9,15))}; // 对角线交叉
        
        // 7. 紧凑开局
        vector<Position> compactOpening = {Position(9, 9), Position(8, 8), Position(8, 10)};
        openingBook["9,9,8,8,8,10"] = {compactOpening, Move(Position(10,8), Position(10,10))}; // 紧凑型
        openingBook["9,9,8,9,9,8"] = {{Position(9,9), Position(8,9), Position(9,8)}, Move(Position(10,9), Position(9,10))}; // 小十字
        openingBook["9,9,8,8,9,8"] = {{Position(9,9), Position(8,8), Position(9,8)}, Move(Position(10,10), Position(9,10))}; // 小菱形
        openingBook["9,9,8,10,9,8"] = {{Position(9,9), Position(8,10), Position(9,8)}, Move(Position(10,8), Position(9,10))}; // 小反菱形
        openingBook["9,9,8,9,7,9"] = {{Position(9,9), Position(8,9), Position(7,9)}, Move(Position(10,9), Position(11,9))}; // 紧凑水平
        openingBook["9,9,9,8,9,7"] = {{Position(9,9), Position(9,8), Position(9,7)}, Move(Position(9,10), Position(9,11))}; // 紧凑垂直
        
        // 8. 分散开局
        vector<Position> spreadOpening = {Position(9, 9), Position(2, 2), Position(16, 16)};
        openingBook["9,9,2,2,16,16"] = {spreadOpening, Move(Position(2, 16), Position(16, 2))}; // 分散型
        openingBook["9,9,2,16,16,2"] = {{Position(9,9), Position(2,16), Position(16,2)}, Move(Position(2,2), Position(16,16))}; // 反对角分散
        openingBook["9,9,2,9,16,9"] = {{Position(9,9), Position(2,9), Position(16,9)}, Move(Position(9,2), Position(9,16))}; // 水平分散
        openingBook["9,9,9,2,9,16"] = {{Position(9,9), Position(9,2), Position(9,16)}, Move(Position(2,9), Position(16,9))}; // 垂直分散
        openingBook["9,9,2,2,2,16"] = {{Position(9,9), Position(2,2), Position(2,16)}, Move(Position(16,2), Position(16,16))}; // 左侧分散
        openingBook["9,9,16,2,16,16"] = {{Position(9,9), Position(16,2), Position(16,16)}, Move(Position(2,2), Position(2,16))}; // 右侧分散
    }
    
    // 生成开局库键
    string generateOpeningKey(const Board& board) {
        string key;
        vector<Position> moves;
        
        // 收集所有已下的棋子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) != EMPTY) {
                    moves.push_back(Position(i, j));
                }
            }
        }
        
        // 按坐标排序
        sort(moves.begin(), moves.end(), [](const Position& a, const Position& b) {
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        
        // 生成键
        for (size_t i = 0; i < moves.size(); i++) {
            if (i > 0) key += ",";
            key += to_string(moves[i].x) + "," + to_string(moves[i].y);
        }
        
        return key;
    }
    
    // 检查开局库
    Move checkOpeningBook(const Board& board, Piece player) {
        string key = generateOpeningKey(board);
        auto it = openingBook.find(key);
        if (it != openingBook.end()) {
            return it->second.bestResponse;
        }
        return Move();
    }
    
    // 初始化残局库
    void initEndgameBook() {
        // 常见的六子棋残局走法
        
        // 1. 五连子进攻
        // 五连子 + 两个空格
        endgameBook["5,5,6,5,7,5,8,5,9,5,4,5,10,5"] = Move(Position(3, 5), Position(11, 5)); // 六连子获胜
        endgameBook["5,5,5,6,5,7,5,8,5,9,5,4,5,10"] = Move(Position(5, 3), Position(5, 11)); // 垂直五连子
        endgameBook["5,5,6,6,7,7,8,8,9,9,4,4,10,10"] = Move(Position(3, 3), Position(11, 11)); // 斜线五连子
        endgameBook["5,10,6,9,7,8,8,7,9,6,4,11,10,5"] = Move(Position(3, 12), Position(11, 4)); // 反斜线五连子
        endgameBook["6,5,7,5,8,5,9,5,10,5,5,5,11,5"] = Move(Position(4, 5), Position(12, 5)); // 长水平五连子
        endgameBook["5,6,5,7,5,8,5,9,5,10,5,5,5,11"] = Move(Position(5, 4), Position(5, 12)); // 长垂直五连子
        
        // 2. 活四进攻
        // 活四 + 两个空格
        endgameBook["4,4,5,4,6,4,7,4,3,4,8,4"] = Move(Position(2, 4), Position(9, 4)); // 形成六连子
        endgameBook["4,5,5,5,6,5,7,5,3,5,8,5"] = Move(Position(2, 5), Position(9, 5)); // 垂直活四
        endgameBook["4,4,5,5,6,6,7,7,3,3,8,8"] = Move(Position(2, 2), Position(9, 9)); // 斜线活四
        endgameBook["4,8,5,7,6,6,7,5,3,9,8,4"] = Move(Position(2, 10), Position(9, 3)); // 反斜线活四
        endgameBook["5,4,6,4,7,4,8,4,4,4,9,4"] = Move(Position(3, 4), Position(10, 4)); // 水平活四
        endgameBook["4,5,4,6,4,7,4,8,4,4,4,9"] = Move(Position(4, 3), Position(4, 10)); // 垂直活四
        
        // 3. 活三进攻
        // 活三 + 合适的位置
        endgameBook["5,6,6,6,7,6,4,6,8,6"] = Move(Position(3, 6), Position(9, 6)); // 形成活四
        endgameBook["5,7,6,7,7,7,4,7,8,7"] = Move(Position(3, 7), Position(9, 7)); // 垂直活三
        endgameBook["5,5,6,6,7,7,4,4,8,8"] = Move(Position(3, 3), Position(9, 9)); // 斜线活三
        endgameBook["5,9,6,8,7,7,4,10,8,6"] = Move(Position(3, 11), Position(9, 5)); // 反斜线活三
        endgameBook["6,6,7,6,8,6,5,6,9,6"] = Move(Position(4, 6), Position(10, 6)); // 水平活三
        endgameBook["6,5,6,6,6,7,6,4,6,8"] = Move(Position(6, 3), Position(6, 9)); // 垂直活三
        
        // 4. 防守残局
        // 对方活四，需要阻挡
        endgameBook["2,3,3,3,4,3,5,3,1,3,6,3"] = Move(Position(0, 3), Position(7, 3)); // 阻挡对方六连子
        endgameBook["2,4,3,4,4,4,5,4,1,4,6,4"] = Move(Position(0, 4), Position(7, 4)); // 阻挡垂直活四
        endgameBook["2,2,3,3,4,4,5,5,1,1,6,6"] = Move(Position(0, 0), Position(7, 7)); // 阻挡斜线活四
        endgameBook["2,6,3,5,4,4,5,3,1,7,6,2"] = Move(Position(0, 8), Position(7, 1)); // 阻挡反斜线活四
        endgameBook["3,3,4,3,5,3,6,3,2,3,7,3"] = Move(Position(1, 3), Position(8, 3)); // 阻挡长水平活四
        endgameBook["3,4,3,5,3,6,3,7,3,3,3,8"] = Move(Position(3, 2), Position(3, 9)); // 阻挡长垂直活四
        
        // 5. 双活三进攻
        endgameBook["4,5,5,5,6,5,4,6,6,6,3,5,7,5"] = Move(Position(3, 6), Position(7, 6)); // 水平双活三
        endgameBook["5,4,5,5,5,6,6,4,6,6,5,3,5,7"] = Move(Position(6, 3), Position(6, 7)); // 垂直双活三
        endgameBook["4,4,5,5,6,6,4,6,6,4,3,3,7,7"] = Move(Position(3, 7), Position(7, 3)); // 交叉双活三
        endgameBook["5,5,6,5,7,5,5,6,7,6,4,5,8,5"] = Move(Position(4, 6), Position(8, 6)); // 平行双活三
        endgameBook["5,5,5,6,5,7,6,5,6,7,4,5,6,4"] = Move(Position(7, 5), Position(7, 7)); // 垂直双活三
        
        // 6. 斜线进攻
        endgameBook["3,3,4,4,5,5,6,6,2,2,7,7"] = Move(Position(1, 1), Position(8, 8)); // 斜线六连子
        endgameBook["3,4,4,5,5,6,6,7,2,3,7,8"] = Move(Position(1, 2), Position(8, 9)); // 斜线活四
        endgameBook["4,3,5,4,6,5,7,6,3,2,8,7"] = Move(Position(2, 1), Position(9, 8)); // 斜线活三
        endgameBook["5,3,6,4,7,5,8,6,4,2,9,7"] = Move(Position(3, 1), Position(10, 8)); // 长斜线进攻
        
        // 7. 反斜线进攻
        endgameBook["3,7,4,6,5,5,6,4,2,8,7,3"] = Move(Position(1, 9), Position(8, 2)); // 反斜线六连子
        endgameBook["3,8,4,7,5,6,6,5,2,9,7,4"] = Move(Position(1, 10), Position(8, 3)); // 反斜线活四
        endgameBook["4,7,5,6,6,5,7,4,3,8,8,3"] = Move(Position(2, 9), Position(9, 2)); // 反斜线活三
        endgameBook["5,7,6,6,7,5,8,4,4,8,9,3"] = Move(Position(3, 9), Position(10, 2)); // 长反斜线进攻
        
        // 8. 边缘残局
        endgameBook["0,5,1,5,2,5,3,5,4,5,0,4,0,6"] = Move(Position(0, 3), Position(0, 7)); // 左侧边缘进攻
        endgameBook["18,5,17,5,16,5,15,5,14,5,18,4,18,6"] = Move(Position(18, 3), Position(18, 7)); // 右侧边缘进攻
        endgameBook["5,0,5,1,5,2,5,3,5,4,4,0,6,0"] = Move(Position(3, 0), Position(7, 0)); // 顶部边缘进攻
        endgameBook["5,18,5,17,5,16,5,15,5,14,4,18,6,18"] = Move(Position(3, 18), Position(7, 18)); // 底部边缘进攻
        endgameBook["0,0,1,1,2,2,3,3,4,4,0,1,1,0"] = Move(Position(5, 5), Position(0, 2)); // 角落边缘进攻
        endgameBook["18,18,17,17,16,16,15,15,14,14,18,17,17,18"] = Move(Position(13, 13), Position(18, 16)); // 对角角落进攻
        
        // 9. 复杂残局
        endgameBook["6,6,7,6,8,6,9,6,7,7,8,7,9,7"] = Move(Position(5, 6), Position(10, 6)); // 双行进攻
        endgameBook["7,5,7,6,7,7,7,8,8,5,8,6,8,7"] = Move(Position(7, 4), Position(7, 9)); // 双列进攻
        endgameBook["6,5,7,5,8,5,7,6,8,6,9,6,6,6"] = Move(Position(5, 5), Position(10, 5)); // 混合进攻
        endgameBook["6,6,7,6,8,6,6,7,7,7,8,7,9,7"] = Move(Position(5, 6), Position(9, 8)); // 交叉进攻
    }
    
    // 生成残局库键
    string generateEndgameKey(const Board& board) {
        string key;
        vector<Position> blackMoves;
        vector<Position> whiteMoves;
        
        // 收集所有已下的棋子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board.get(i, j) == BLACK) {
                    blackMoves.push_back(Position(i, j));
                } else if (board.get(i, j) == WHITE) {
                    whiteMoves.push_back(Position(i, j));
                }
            }
        }
        
        // 按坐标排序
        sort(blackMoves.begin(), blackMoves.end(), [](const Position& a, const Position& b) {
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        
        sort(whiteMoves.begin(), whiteMoves.end(), [](const Position& a, const Position& b) {
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        
        // 生成键：先黑后白
        for (const auto& pos : blackMoves) {
            if (!key.empty()) key += ",";
            key += to_string(pos.x) + "," + to_string(pos.y);
        }
        for (const auto& pos : whiteMoves) {
            key += "," + to_string(pos.x) + "," + to_string(pos.y);
        }
        
        return key;
    }
    
    // 检查残局库
    Move checkEndgameBook(const Board& board, Piece player) {
        string key = generateEndgameKey(board);
        auto it = endgameBook.find(key);
        if (it != endgameBook.end()) {
            return it->second;
        }
        return Move();
    }
    
    // 并行搜索函数
    Move parallelSearch(Board& board, Piece player, int depth) {
        if (!parallelSearchEnabled || board.moveCount == 0) {
            return iterativeDeepening(board, player);
        }
        
        // 生成候选走法
        vector<Move> moves = moveGenerator.generateMoves(board, player, evaluator, 15); // 增加候选走法数量
        if (moves.empty()) {
            return Move();
        }
        
        // 初始化并行搜索变量
        parallelMoves = moves;
        parallelScores.resize(moves.size(), 0);
        parallelIndex = 0;
        parallelSearchDone = false;
        completedTasks = 0;
        int totalTasks = moves.size();
        
        // 使用线程池执行任务
        for (int i = 0; i < totalTasks; i++) {
            submitTask([this, i, board, player, depth, totalTasks]() {
                Move move = parallelMoves[i];
                Board tempBoard = board;
                tempBoard.place(move.p1.x, move.p1.y, player);
                tempBoard.place(move.p2.x, move.p2.y, player);
                
                // 检查时间
                if (checkTime()) {
                    completedTasks++;
                    return;
                }
                
                Move dummyMove;
                int score = -pvs(tempBoard, depth - 1, -INT_MAX, INT_MAX, 
                                 (player == BLACK) ? WHITE : BLACK, move.p2.x, move.p2.y, dummyMove);
                
                {   
                    unique_lock<mutex> lock(searchMutex);
                    parallelScores[i] = score;
                }
                
                completedTasks++;
            });
        }
        
        // 等待所有任务完成
        waitForTasks();
        
        // 找出最佳走法
        int bestScore = -INT_MAX;
        int bestIndex = 0;
        for (int i = 0; i < parallelScores.size(); i++) {
            if (parallelScores[i] > bestScore) {
                bestScore = parallelScores[i];
                bestIndex = i;
            }
        }
        
        return parallelMoves[bestIndex];
    }
    
    // 混合并行搜索函数（结合深度和广度）
    Move hybridParallelSearch(Board& board, Piece player) {
        if (!parallelSearchEnabled || board.moveCount == 0) {
            return iterativeDeepening(board, player);
        }
        
        // 生成候选走法
        vector<Move> moves = moveGenerator.generateMoves(board, player, evaluator, 10);
        if (moves.empty()) {
            return Move();
        }
        
        // 对前几个最佳走法进行深度搜索
        int deepSearchCount = min(3, (int)moves.size());
        vector<int> deepScores(deepSearchCount, 0);
        
        // 对剩余走法进行浅度搜索
        int shallowSearchCount = moves.size() - deepSearchCount;
        vector<int> shallowScores(shallowSearchCount, 0);
        
        completedTasks = 0;
        int totalTasks = deepSearchCount + shallowSearchCount;
        
        // 提交深度搜索任务
        for (int i = 0; i < deepSearchCount; i++) {
            submitTask([this, i, &moves, board, player, &deepScores, totalTasks]() {
                Board tempBoard = board;
                tempBoard.place(moves[i].p1.x, moves[i].p1.y, player);
                tempBoard.place(moves[i].p2.x, moves[i].p2.y, player);
                
                Move dummyMove;
                int score = -pvs(tempBoard, 5, -INT_MAX, INT_MAX, 
                                 (player == BLACK) ? WHITE : BLACK, moves[i].p2.x, moves[i].p2.y, dummyMove);
                deepScores[i] = score;
                completedTasks++;
            });
        }
        
        // 提交浅度搜索任务
        for (int i = 0; i < shallowSearchCount; i++) {
            submitTask([this, i, &moves, board, player, deepSearchCount, &shallowScores, totalTasks]() {
                int index = deepSearchCount + i;
                Board tempBoard = board;
                tempBoard.place(moves[index].p1.x, moves[index].p1.y, player);
                tempBoard.place(moves[index].p2.x, moves[index].p2.y, player);
                
                Move dummyMove;
                int score = -pvs(tempBoard, 3, -INT_MAX, INT_MAX, 
                                 (player == BLACK) ? WHITE : BLACK, moves[index].p2.x, moves[index].p2.y, dummyMove);
                shallowScores[i] = score;
                completedTasks++;
            });
        }
        
        // 等待所有任务完成
        waitForTasks();
        
        // 合并分数
        vector<int> allScores;
        for (int score : deepScores) {
            allScores.push_back(score);
        }
        for (int score : shallowScores) {
            allScores.push_back(score);
        }
        
        // 找出最佳走法
        int bestScore = -INT_MAX;
        int bestIndex = 0;
        for (int i = 0; i < allScores.size(); i++) {
            if (allScores[i] > bestScore) {
                bestScore = allScores[i];
                bestIndex = i;
            }
        }
        
        return moves[bestIndex];
    }
    
    // MTD(f) 搜索算法
    int mtdf(Board& board, int depth, int f, Move& bestMove) {
        int g = f;
        int upperBound = INT_MAX;
        int lowerBound = -INT_MAX;
        
        while (lowerBound < upperBound) {
            int beta = (g == lowerBound) ? g + 1 : g;
            Move currentBestMove;
            int score = -pvs(board, depth, -beta, -g, myPiece, -1, -1, currentBestMove);
            
            if (timeUp) return g;
            
            g = score;
            
            if (g < beta) {
                upperBound = g;
            } else {
                lowerBound = g;
                bestMove = currentBestMove;
            }
        }
        
        return g;
    }
    
    // 迭代加深搜索
    Move iterativeDeepening(Board& board, Piece player) {
        initZobrist();
        startTime = chrono::steady_clock::now();
        timeUp = false;
        nodesSearched = 0;
        cutoffs = 0;
        
        myPiece = player;
        opponentPiece = (player == BLACK) ? WHITE : BLACK;
        
        Move bestMove;
        Move lastCompleteMove;
        bool foundMove = false;
        int lastScore = 0;
        
        // 清空历史表和杀手走法
        memset(historyTable, 0, sizeof(historyTable));
        memset(killerMoves, 0, sizeof(killerMoves));
        
        // 特殊开局处理
        if (board.moveCount == 0) {
            // 第一手，下在中心
            int center = BOARD_SIZE / 2;
            return Move(Position(center, center), Position(center, center));
        }
        
        // 计算当前局面复杂度
        int complexity = calculateComplexity(board);
        
        // 计算时间限制
        int remainingTime = timeLimitMs; // 假设剩余时间就是时间限制
        int moveNumber = board.moveCount / 2; // 每回合2子，所以回合数是总子数除以2
        int calculatedTimeLimit = calculateTimeLimit(remainingTime, moveNumber, board);
        
        // 计算自适应搜索深度
        int adaptiveDepth = calculateAdaptiveDepth(calculatedTimeLimit, board);
        int startDepth = max(2, adaptiveDepth - 1); // 从稍低的深度开始
        
        // 迭代加深
        for (int depth = startDepth; depth <= MAX_DEPTH && !timeUp; depth += ITERATIVE_DEEPENING_STEP) {
            // 每轮搜索增加年龄计数器
            ttAge++;
            
            auto depthStartTime = chrono::steady_clock::now();
            Move currentBestMove;
            int score = mtdf(board, depth, lastScore, currentBestMove);
            auto depthEndTime = chrono::steady_clock::now();
            int depthTime = chrono::duration_cast<chrono::milliseconds>(depthEndTime - depthStartTime).count();
            
            // 记录搜索时间
            recordSearchTime(depthTime);
            
            if (!timeUp) {
                bestMove = currentBestMove;
                lastCompleteMove = currentBestMove;
                lastScore = score;
                foundMove = true;
                
                auto now = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - startTime).count();
                
                // 如果已经找到必胜/必败，提前结束
                if (abs(score) >= SCORE_SIX / 2) {
                    break;
                }
                
                // 计算剩余时间比例
                double timeRatio = static_cast<double>(elapsed) / calculatedTimeLimit;
                
                // 根据复杂度、时间比例和局面评估调整搜索策略
                if (complexity > 800) {
                    // 非常复杂局面，更保守的时间估计
                    if (timeRatio > 0.5) {
                        break;
                    }
                } else if (complexity > 600) {
                    // 复杂局面，中等保守的时间估计
                    if (timeRatio > 0.6) {
                        break;
                    }
                } else if (complexity > 400) {
                    // 中等复杂局面，适度保守的时间估计
                    if (timeRatio > 0.7) {
                        break;
                    }
                } else {
                    // 简单局面，更激进的时间估计
                    if (timeRatio > 0.85) {
                        break;
                    }
                }
                
                // 根据局面评估调整
                if (abs(score) > 50000) {
                    // 优势或劣势明显，提前结束
                    if (timeRatio > 0.4) {
                        break;
                    }
                } else if (abs(score) > 30000) {
                    // 优势或劣势明显，提前结束
                    if (timeRatio > 0.5) {
                        break;
                    }
                } else if (abs(score) < 5000) {
                    // 局面非常接近，需要更多搜索
                    if (timeRatio < 0.8) {
                        // 继续搜索
                    } else {
                        break;
                    }
                }
            }
        }
        
        if (!foundMove) {
            // 如果没有完成任何搜索，使用简单启发式
            auto moves = moveGenerator.generateMoves(board, player, evaluator, 1);
            if (!moves.empty()) {
                bestMove = moves[0];
            }
        } else {
            bestMove = lastCompleteMove;
        }
        
        return bestMove;
    }
};

// ==================== 主程序 ====================
class Connect6AI {
public:
    Board board;
    SearchEngine engine;
    Piece myColor;
    
    Connect6AI() {
        myColor = BLACK;
    }
    
    // 初始化
    void init(Piece color) {
        myColor = color;
        board.clear();
    }
    
    // 对手落子
    void opponentMove(int x1, int y1, int x2, int y2) {
        Piece opponent = (myColor == BLACK) ? WHITE : BLACK;
        if (board.isEmpty(x1, y1)) {
            board.place(x1, y1, opponent);
        }
        if (x2 >= 0 && y2 >= 0 && board.isEmpty(x2, y2)) {
            board.place(x2, y2, opponent);
        }
    }
    
    // AI思考并返回走法
    Move think() {
        Move bestMove;
        
        try {
            // 检查开局库
            if (board.moveCount < 10) {
                Move openingMove = engine.checkOpeningBook(board, myColor);
                if (openingMove.p1.x >= 0) {
                    bestMove = openingMove;
                    // 执行走法
                    if (bestMove.p1.x >= 0 && board.isEmpty(bestMove.p1.x, bestMove.p1.y)) {
                        board.place(bestMove.p1.x, bestMove.p1.y, myColor);
                    }
                    if (bestMove.p2.x >= 0 && !(bestMove.p1.x == bestMove.p2.x && bestMove.p1.y == bestMove.p2.y) && board.isEmpty(bestMove.p2.x, bestMove.p2.y)) {
                        board.place(bestMove.p2.x, bestMove.p2.y, myColor);
                    }
                    return bestMove;
                }
            }
            
            // 检查残局库
            if (board.moveCount > 40) {
                Move endgameMove = engine.checkEndgameBook(board, myColor);
                if (endgameMove.p1.x >= 0) {
                    bestMove = endgameMove;
                    // 执行走法
                    if (bestMove.p1.x >= 0 && board.isEmpty(bestMove.p1.x, bestMove.p1.y)) {
                        board.place(bestMove.p1.x, bestMove.p1.y, myColor);
                    }
                    if (bestMove.p2.x >= 0 && !(bestMove.p1.x == bestMove.p2.x && bestMove.p1.y == bestMove.p2.y) && board.isEmpty(bestMove.p2.x, bestMove.p2.y)) {
                        board.place(bestMove.p2.x, bestMove.p2.y, myColor);
                    }
                    return bestMove;
                }
            }
            
            // 对于复杂局面使用混合并行搜索，简单局面使用传统搜索
            if (board.moveCount > 10) {
                // 复杂局面，使用混合并行搜索
                bestMove = engine.hybridParallelSearch(board, myColor);
            } else {
                // 简单局面，使用传统迭代加深搜索
                bestMove = engine.iterativeDeepening(board, myColor);
            }
            
            // 验证并执行走法
            if (bestMove.p1.x >= 0 && bestMove.p1.x < BOARD_SIZE && bestMove.p1.y >= 0 && bestMove.p1.y < BOARD_SIZE && board.isEmpty(bestMove.p1.x, bestMove.p1.y)) {
                board.place(bestMove.p1.x, bestMove.p1.y, myColor);
            }
            if (bestMove.p2.x >= 0 && bestMove.p2.x < BOARD_SIZE && bestMove.p2.y >= 0 && bestMove.p2.y < BOARD_SIZE && !(bestMove.p1.x == bestMove.p2.x && bestMove.p1.y == bestMove.p2.y) && board.isEmpty(bestMove.p2.x, bestMove.p2.y)) {
                board.place(bestMove.p2.x, bestMove.p2.y, myColor);
            }
            
            // 确保返回有效的走法
            if (bestMove.p1.x < 0 || bestMove.p1.x >= BOARD_SIZE || bestMove.p1.y < 0 || bestMove.p1.y >= BOARD_SIZE) {
                // 如果走法无效，生成一个简单的走法
                auto moves = engine.moveGenerator.generateMoves(board, myColor, engine.evaluator, 1);
                if (!moves.empty()) {
                    bestMove = moves[0];
                    if (bestMove.p1.x >= 0 && board.isEmpty(bestMove.p1.x, bestMove.p1.y)) {
                        board.place(bestMove.p1.x, bestMove.p1.y, myColor);
                    }
                    if (bestMove.p2.x >= 0 && !(bestMove.p1.x == bestMove.p2.x && bestMove.p1.y == bestMove.p2.y) && board.isEmpty(bestMove.p2.x, bestMove.p2.y)) {
                        board.place(bestMove.p2.x, bestMove.p2.y, myColor);
                    }
                }
            }
            
            return bestMove;
        } catch (const exception& e) {
            cerr << "Error in think(): " << e.what() << endl;
            // 发生错误时，返回一个安全的走法
            auto moves = engine.moveGenerator.generateMoves(board, myColor, engine.evaluator, 1);
            if (!moves.empty()) {
                bestMove = moves[0];
                if (bestMove.p1.x >= 0 && board.isEmpty(bestMove.p1.x, bestMove.p1.y)) {
                    board.place(bestMove.p1.x, bestMove.p1.y, myColor);
                }
                if (bestMove.p2.x >= 0 && !(bestMove.p1.x == bestMove.p2.x && bestMove.p1.y == bestMove.p2.y) && board.isEmpty(bestMove.p2.x, bestMove.p2.y)) {
                    board.place(bestMove.p2.x, bestMove.p2.y, myColor);
                }
            }
            return bestMove;
        }
    }
    
    // 分析对手走法
    void analyzeOpponentMove(const Move& move) {
        Piece opponentColor = (myColor == BLACK) ? WHITE : BLACK;
        engine.opponentModel.analyzeMove(board, move, opponentColor);
    }
    
    // 输出走法
    void outputMove(const Move& move) {
        cout << move.p1.x << " " << move.p1.y;
        if (move.p2.x >= 0 && !(move.p1.x == move.p2.x && move.p1.y == move.p2.y)) {
            cout << " " << move.p2.x << " " << move.p2.y;
        }
        cout << endl;
        cout.flush();
    }
};

// ==================== 输入输出处理 ====================
#ifdef AI_STANDALONE
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Connect6AI ai;
    string cmd;
    
    while (cin >> cmd) {
        if (cmd == "START") {
            int size;
            cin >> size;
            // 初始化棋盘
            ai.board.clear();
        }
        else if (cmd == "TURN") {
            // 轮到我方下棋
            Move move = ai.think();
            ai.outputMove(move);
        }
        else if (cmd == "BEGIN") {
            // 黑方第一手，只下一子
            int center = BOARD_SIZE / 2;
            ai.board.place(center, center, BLACK);
            ai.myColor = BLACK;
            cout << center << " " << center << endl;
            cout.flush();
        }
        else if (cmd == "BOARD") {
            // 读取棋盘状态
            ai.board.clear();
            string input;
            while (cin >> input && input != "DONE") {
                int x, y, color;
                x = stoi(input);
                cin >> y >> color;
                if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
                    ai.board.place(x, y, (Piece)color);
                }
            }
        }
        else if (cmd == "INFO") {
            // 读取额外信息
            string key;
            cin >> key;
            if (key == "timeout_turn") {
                int timeout;
                cin >> timeout;
                ai.engine.setTimeLimit(timeout - 500);  // 留500ms缓冲
            }
            else if (key == "max_memory") {
                int memory;
                cin >> memory;
            }
            else {
                string value;
                cin >> value;
            }
        }
        else if (cmd == "END") {
            break;
        }
        else if (cmd == "ABOUT") {
            cout << "name=\"Connect6AI\", author=\"AI\", version=\"1.0\"" << endl;
            cout.flush();
        }
        else {
            // 可能是对手的走法
            // 尝试解析为坐标
            try {
                int x1 = stoi(cmd);
                int y1, x2, y2;
                cin >> y1;
                
                // 检查是否还有第二子
                if (cin.peek() == '\n' || cin.peek() == EOF) {
                    // 只有一子 (开局情况)
                    ai.opponentMove(x1, y1, -1, -1);
                } else {
                    cin >> x2 >> y2;
                    ai.opponentMove(x1, y1, x2, y2);
                }
            } catch (...) {
                // 未知命令，忽略
            }
        }
    }
    
    return 0;
}
#endif

#ifdef AI_SERVER
// ==================== HTTP 服务器 ====================
class HttpServer {
private:
    SOCKET serverSocket;
    Connect6AI ai;
    
public:
    HttpServer() {
        // 初始化 Winsock
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            cerr << "WSAStartup failed." << endl;
            return;
        }
        
        // 创建服务器套接字
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == INVALID_SOCKET) {
            cerr << "socket failed." << endl;
            WSACleanup();
            return;
        }
        
        // 绑定地址
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(8080);
        
        if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            cerr << "bind failed." << endl;
            closesocket(serverSocket);
            WSACleanup();
            return;
        }
        
        // 开始监听
        if (listen(serverSocket, 5) == SOCKET_ERROR) {
            cerr << "listen failed." << endl;
            closesocket(serverSocket);
            WSACleanup();
            return;
        }
        
        cout << "Server started on port 8080" << endl;
    }
    
    ~HttpServer() {
        closesocket(serverSocket);
        WSACleanup();
    }
    
    void start() {
        while (true) {
            // 接受客户端连接
            SOCKET clientSocket = accept(serverSocket, NULL, NULL);
            if (clientSocket == INVALID_SOCKET) {
                cerr << "accept failed." << endl;
                continue;
            }
            
            // 处理请求
            handleRequest(clientSocket);
            
            // 关闭连接
            closesocket(clientSocket);
        }
    }
    
    void handleRequest(SOCKET clientSocket) {
        char buffer[4096];
        int bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);
        if (bytesRead == SOCKET_ERROR) {
            cerr << "recv failed." << endl;
            return;
        }
        
        // 解析请求
        string request(buffer, bytesRead);
        cout << "Request: " << request << endl;
        
        // 检查是否是 POST 请求
        if (request.find("POST /ai/move") != string::npos) {
            // 提取 JSON 数据
            size_t jsonStart = request.find("\r\n\r\n");
            if (jsonStart != string::npos) {
                string jsonData = request.substr(jsonStart + 4);
                processMoveRequest(clientSocket, jsonData);
            }
        } else {
            // 返回 404
            string response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot Found";
            send(clientSocket, response.c_str(), response.size(), 0);
        }
    }
    
    // 简单的JSON解析辅助函数
    int parsePlayer(const string& jsonData) {
        size_t playerStart = jsonData.find("player");
        if (playerStart == string::npos) return 2; // 默认白棋
        
        size_t colonPos = jsonData.find(":", playerStart);
        if (colonPos == string::npos) return 2;
        
        size_t valueStart = colonPos + 1;
        // 跳过空白字符
        while (valueStart < jsonData.size() && (jsonData[valueStart] == ' ' || jsonData[valueStart] == '\t' || jsonData[valueStart] == '\n')) {
            valueStart++;
        }
        
        size_t valueEnd = valueStart;
        while (valueEnd < jsonData.size() && (isdigit(jsonData[valueEnd]) || jsonData[valueEnd] == '-')) {
            valueEnd++;
        }
        
        string playerStr = jsonData.substr(valueStart, valueEnd - valueStart);
        try {
            return stoi(playerStr);
        } catch (...) {
            return 2;
        }
    }
    
    // 解析棋盘数据
    void parseBoard(const string& jsonData, vector<vector<int>>& board) {
        size_t boardStart = jsonData.find("board");
        if (boardStart == string::npos) return;
        
        size_t arrayStart = jsonData.find("[", boardStart);
        if (arrayStart == string::npos) return;
        
        size_t rowStart = arrayStart + 1;
        int row = 0;
        
        while (row < BOARD_SIZE && rowStart < jsonData.size()) {
            size_t rowEnd = jsonData.find("]", rowStart);
            if (rowEnd == string::npos) break;
            
            string rowStr = jsonData.substr(rowStart, rowEnd - rowStart);
            size_t colStart = 0;
            int col = 0;
            
            while (col < BOARD_SIZE && colStart < rowStr.size()) {
                size_t commaPos = rowStr.find(",", colStart);
                if (commaPos == string::npos) {
                    commaPos = rowStr.size();
                }
                
                string cellStr = rowStr.substr(colStart, commaPos - colStart);
                // 跳过空白字符
                size_t cellValueStart = 0;
                while (cellValueStart < cellStr.size() && (cellStr[cellValueStart] == ' ' || cellStr[cellValueStart] == '\t')) {
                    cellValueStart++;
                }
                size_t cellValueEnd = cellStr.size() - 1;
                while (cellValueEnd >= cellValueStart && (cellStr[cellValueEnd] == ' ' || cellStr[cellValueEnd] == '\t')) {
                    cellValueEnd--;
                }
                
                if (cellValueStart <= cellValueEnd) {
                    string valueStr = cellStr.substr(cellValueStart, cellValueEnd - cellValueStart + 1);
                    try {
                        board[row][col] = stoi(valueStr);
                    } catch (...) {
                        board[row][col] = 0;
                    }
                }
                
                colStart = commaPos + 1;
                col++;
            }
            
            rowStart = jsonData.find("[", rowEnd);
            if (rowStart == string::npos) break;
            rowStart++;
            row++;
        }
    }
    
    void processMoveRequest(SOCKET clientSocket, const string& jsonData) {
        try {
            // 解析 JSON 数据
            vector<vector<int>> board(BOARD_SIZE, vector<int>(BOARD_SIZE, 0));
            int player = parsePlayer(jsonData);
            parseBoard(jsonData, board);
            
            // 验证玩家颜色
            if (player != BLACK && player != WHITE) {
                string errorResponse = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                errorResponse += "{\"error\": \"Invalid player color\"}";
                send(clientSocket, errorResponse.c_str(), errorResponse.size(), 0);
                return;
            }
            
            // 设置 AI 颜色
            ai.init((Piece)player);
            
            // 清空棋盘
            ai.board.clear();
            
            // 更新棋盘
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (board[i][j] != 0) {
                        if (board[i][j] == BLACK || board[i][j] == WHITE) {
                            ai.board.place(i, j, (Piece)board[i][j]);
                        }
                    }
                }
            }
            
            // 计算 AI 走法
            Move move = ai.think();
            
            // 验证走法
            if (move.p1.x < 0 || move.p1.x >= BOARD_SIZE || move.p1.y < 0 || move.p1.y >= BOARD_SIZE) {
                string errorResponse = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                errorResponse += "{\"error\": \"Invalid move generated\"}";
                send(clientSocket, errorResponse.c_str(), errorResponse.size(), 0);
                return;
            }
            
            // 构建响应
            string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            response += "{";
            response += "\"move\": {";
            response += "\"p1\": {\"x\": " + to_string(move.p1.x) + ", \"y\": " + to_string(move.p1.y) + "},";
            response += "\"p2\": {\"x\": " + to_string(move.p2.x) + ", \"y\": " + to_string(move.p2.y) + "}";
            response += "}";
            response += "}";
            
            int result = send(clientSocket, response.c_str(), response.size(), 0);
            if (result == SOCKET_ERROR) {
                cerr << "Send failed: " << WSAGetLastError() << endl;
            }
        } catch (const exception& e) {
            string errorResponse = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            errorResponse += "{\"error\": \"Internal server error\"}";
            send(clientSocket, errorResponse.c_str(), errorResponse.size(), 0);
            cerr << "Error processing move request: " << e.what() << endl;
        }
    }
};

int main() {
    HttpServer server;
    server.start();
    return 0;
}
#endif

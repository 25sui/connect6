// 六子棋游戏核心逻辑与AI算法实现
// ========== 配置与状态 ==========
// 棋盘大小：19×19标准
const BOARD_SIZE = 19;
// 胜利条件：连成6子
const WIN_COUNT = 6;

// AI策略配置：均衡/稳健/进攻三种风格
// 参数控制
const AI_PROFILES = {
    balanced: {  // 均衡策略
        threatBonusHigh: 2600,    // 高威胁奖励
        threatBonusMedium: 1200,  // 中等威胁奖励
        defensiveCoverageWeight: 0.05,  // 防守覆盖权重
        defensiveThreatBonus: 500,       // 防守威胁奖励
        pairRiskWeight: 0.16,    // 双落子风险权重
        pairLossPenalty: 22000,  // 双落子损失惩罚
        sameLineBonus: 260,      // 同线奖励
        closeDistanceBonus: 320, // 近距离协同奖励
        mediumDistanceBonus: 120,// 中距离协同奖励
        farDistancePenalty: 180  // 远距离惩罚
    },
    solid: {  // 稳健策略：更注重防守
        threatBonusHigh: 2200,
        threatBonusMedium: 1000,
        defensiveCoverageWeight: 0.08,  // 更高的防守权重
        defensiveThreatBonus: 800,
        pairRiskWeight: 0.22,
        pairLossPenalty: 28000,
        sameLineBonus: 220,
        closeDistanceBonus: 260,
        mediumDistanceBonus: 100,
        farDistancePenalty: 220
    },
    aggressive: {  // 进攻策略：更注重进攻
        threatBonusHigh: 3200,
        threatBonusMedium: 1600,
        defensiveCoverageWeight: 0.04,  // 更低的防守权重
        defensiveThreatBonus: 350,
        pairRiskWeight: 0.1,
        pairLossPenalty: 18000,
        sameLineBonus: 320,
        closeDistanceBonus: 380,
        mediumDistanceBonus: 160,
        farDistancePenalty: 140
    }
};

// AI思考时间配置
const AI_THINK_DELAY = {
    easy: 2000,
    medium: 5000,
    hard: 8000,
    master: 15000
};

//强化学习组件
// 基于简单神经网络的强化学习AI实现
// 包含策略网络（选择落子）和价值网络（评估局面）
class ReinforcementLearning {
    constructor() {
        // 策略网络：输入棋盘状态，输出每个位置的落子概率
        this.policyNetwork = null;
        // 价值网络：输入棋盘状态，输出当前玩家的获胜概率
        this.valueNetwork = null;
        // 训练数据缓存
        this.trainingData = [];
        // 训练回合数
        this.trainingEpisodes = 0;
        // 最佳模型保存
        this.bestModel = null;
        // 训练超参数配置
        this.trainingConfig = {
            learningRate: 0.001,     // 学习率
            discountFactor: 0.99,    // 折扣因子（用于计算未来奖励的当前价值）
            explorationRate: 0.2,    // 探索率（ε-greedy策略）
            batchSize: 32,           // 批次大小
            epochs: 10               // 每批训练的轮数
        };
    }

    // 编码棋盘状态：将当前棋盘转换为神经网络输入向量
    // 输入：player - 当前玩家（1或2）
    // 输出：一维数组，1表示当前玩家棋子，-1表示对手，0表示空位
    encodeState(player) {
        const state = [];
        const opponentPlayer = player === 1 ? 2 : 1;
        
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] === player) {
                    state.push(1);
                } else if (board[y][x] === opponentPlayer) {
                    state.push(-1);
                } else {
                    state.push(0);
                }
            }
        }
        
        return state;
    }

    // 计算奖励函数
    // 获胜得+1，失败得-1，平局得0
    calculateReward(player, winner) {
        if (winner === player) return 1;
        if (winner === (player === 1 ? 2 : 1)) return -1;
        return 0;
    }

    // 创建策略网络结构
    // 答辩要点：三层全连接网络，输入361维（19×19），输出361维（softmax概率）
    createPolicyNetwork() {
        return {
            layers: [
                { type: 'dense', inputSize: BOARD_SIZE * BOARD_SIZE, outputSize: 256, activation: 'relu' },  // 第一层：256隐藏单元
                { type: 'dense', inputSize: 256, outputSize: 128, activation: 'relu' },                   // 第二层：128隐藏单元
                { type: 'dense', inputSize: 128, outputSize: BOARD_SIZE * BOARD_SIZE, activation: 'softmax' }  // 输出层：softmax概率
            ],
            weights: this.initializeWeights()  // 初始化权重
        };
    }

    // 创建价值网络结构
    // 答辩要点：三层全连接网络，输出单一值（tanh归一化到[-1,1]）
    createValueNetwork() {
        return {
            layers: [
                { type: 'dense', inputSize: BOARD_SIZE * BOARD_SIZE, outputSize: 256, activation: 'relu' },
                { type: 'dense', inputSize: 256, outputSize: 128, activation: 'relu' },
                { type: 'dense', inputSize: 128, outputSize: 1, activation: 'tanh' }  // 输出：局面价值
            ],
            weights: this.initializeWeights()
        };
    }

    // 初始化网络权重：使用Xavier/Glorot初始化
    initializeWeights() {
        const weights = [];
        const layers = [BOARD_SIZE * BOARD_SIZE, 256, 128, BOARD_SIZE * BOARD_SIZE];
        
        for (let i = 0; i < layers.length - 1; i++) {
            const w = [];
            for (let j = 0; j < layers[i + 1]; j++) {
                w.push([]);
                for (let k = 0; k < layers[i]; k++) {
                    // Xavier初始化：范围为[-sqrt(6/(in+out)), sqrt(6/(in+out))]
                    w[j].push((Math.random() - 0.5) * 2 * Math.sqrt(6 / (layers[i] + layers[i + 1])));
                }
            }
            weights.push(w);
        }
        return weights;
    }

    // 前向传播：计算网络输出
    forward(network, input) {
        let output = [...input];
        
        for (let i = 0; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const weights = network.weights[i];
            const newOutput = [];
            
            // 计算每层的加权和
            for (let j = 0; j < weights.length; j++) {
                let sum = 0;
                for (let k = 0; k < weights[j].length; k++) {
                    sum += output[k] * weights[j][k];
                }
                
                // 应用激活函数
                if (layer.activation === 'relu') {
                    newOutput.push(Math.max(0, sum));  // ReLU激活
                } else if (layer.activation === 'softmax') {
                    newOutput.push(Math.exp(sum));     // softmax先求指数
                } else if (layer.activation === 'tanh') {
                    newOutput.push(Math.tanh(sum));    // tanh激活
                } else {
                    newOutput.push(sum);
                }
            }
            
            // softmax归一化
            if (layer.activation === 'softmax') {
                const total = newOutput.reduce((a, b) => a + b, 0);
                output = newOutput.map(v => v / total);
            } else {
                output = newOutput;
            }
        }
        
        return output;
    }

    // 预测策略：输出各位置的落子概率
    predictPolicy(state) {
        if (!this.policyNetwork) return null;
        return this.forward(this.policyNetwork, state);
    }

    // 预测价值：输出当前局面的价值
    predictValue(state) {
        if (!this.valueNetwork) return 0;
        const output = this.forward(this.valueNetwork, state);
        return output[0];
    }

    // 选择落子：ε-greedy策略（探索与利用平衡）
    selectMove(state, exploration = true) {
        const policy = this.predictPolicy(state);
        if (!policy) return null;
        
        // 探索阶段：随机选择（概率为explorationRate）
        if (exploration && Math.random() < this.trainingConfig.explorationRate) {
            const candidates = [];
            for (let i = 0; i < policy.length; i++) {
                const x = i % BOARD_SIZE;
                const y = Math.floor(i / BOARD_SIZE);
                if (board[y][x] === 0) {
                    candidates.push({ x, y, prob: policy[i] });
                }
            }
            if (candidates.length === 0) return null;
            return candidates[Math.floor(Math.random() * candidates.length)];
        }
        
        // 利用阶段：选择概率最高的落子
        let bestMove = null;
        let bestProb = -1;
        for (let i = 0; i < policy.length; i++) {
            const x = i % BOARD_SIZE;
            const y = Math.floor(i / BOARD_SIZE);
            if (board[y][x] === 0 && policy[i] > bestProb) {
                bestProb = policy[i];
                bestMove = { x, y };
            }
        }
        return bestMove;
    }

    // 训练函数：进行自我对弈并更新网络
    train(episodes = 100, onProgress = null) {
        console.log(`开始训练，共 ${episodes} 局...`);
        
        for (let episode = 0; episode < episodes; episode++) {
            this.selfPlay();  // 自我对弈一局
            this.trainingEpisodes++;
            
            // 关键修改：每局训练后立即更新网络并清空数据
            if (this.trainingData.length > 0) {
                this.updateNetworks();
            }
            
            // 每10局显示进度
            if ((episode + 1) % 10 === 0) {
                const progress = `训练进度: ${episode + 1}/${episodes}`;
                console.log(progress);
                if (onProgress) onProgress(episode + 1, episodes);
            }
            
            // 每100局自动保存检查点
            if ((episode + 1) % 100 === 0) {
                console.log(`已到达检查点，正在保存...`);
                this.saveCheckpoint();
            }
        }
        
        console.log('训练完成！');
        this.trainingEpisodes = episodes;
    }

    // 保存检查点：将模型压缩后保存到localStorage
    saveCheckpoint() {
        try {
            const checkpoint = {
                policyNetwork: this.policyNetwork,
                valueNetwork: this.valueNetwork,
                trainingEpisodes: this.trainingEpisodes,
                timestamp: Date.now(),
                version: 1
            };
            
            const compressed = this.compressCheckpoint(checkpoint);
            
            // 尝试保存到浏览器存储
            localStorage.setItem('connect6_rl_checkpoint', compressed);
            console.log(`检查点已保存 (${this.trainingEpisodes}局, 大小: ${(compressed.length / 1024).toFixed(2)} KB)`);
            return true;
        } catch (e) {
            console.warn('检查点保存失败，尝试清理旧数据...');
            
            // 清理旧数据
            this.clearOldData();
            
            // 再次尝试保存
            try {
                const checkpoint = {
                    policyNetwork: this.policyNetwork,
                    valueNetwork: this.valueNetwork,
                    trainingEpisodes: this.trainingEpisodes,
                    timestamp: Date.now(),
                    version: 1
                };
                
                const compressed = this.compressCheckpoint(checkpoint);
                localStorage.setItem('connect6_rl_checkpoint', compressed);
                console.log(`检查点已保存 (${this.trainingEpisodes}局, 清理后)`);
                return true;
            } catch (e2) {
                console.error('检查点保存失败:', e2.message);
                return false;
            }
        }
    }

    // 清理旧的localStorage数据
    clearOldData() {
        let cleared = 0;
        for (const key of Object.keys(localStorage)) {
            if (key.startsWith('connect6_') || key.includes('model') || key.includes('rl')) {
                localStorage.removeItem(key);
                cleared++;
            }
        }
        console.log(`已清理 ${cleared} 个旧数据文件`);
    }

    // 加载检查点
    loadCheckpoint() {
        const saved = localStorage.getItem('connect6_rl_checkpoint');
        if (saved) {
            try {
                const checkpoint = this.decompressCheckpoint(saved);
                if (checkpoint.version === 1) {
                    this.policyNetwork = checkpoint.policyNetwork;
                    this.valueNetwork = checkpoint.valueNetwork;
                    this.trainingEpisodes = checkpoint.trainingEpisodes;
                    console.log(`检查点已加载 (${this.trainingEpisodes}局)`);
                    return true;
                }
            } catch (e) {
                console.error('检查点加载失败:', e.message);
            }
        }
        return false;
    }

    // 压缩检查点数据：将权重数组扁平化后base64编码
    compressCheckpoint(checkpoint) {
        const model = {
            policyWeights: this.flattenWeights(checkpoint.policyNetwork?.weights || []),
            valueWeights: this.flattenWeights(checkpoint.valueNetwork?.weights || []),
            trainingEpisodes: checkpoint.trainingEpisodes,
            timestamp: checkpoint.timestamp,
            version: checkpoint.version
        };
        
        const jsonStr = JSON.stringify(model);
        return this.encodeString(jsonStr);
    }

    // 解压检查点数据
    decompressCheckpoint(compressed) {
        const jsonStr = this.decodeString(compressed);
        const model = JSON.parse(jsonStr);
        
        return {
            policyNetwork: {
                layers: this.createPolicyNetwork().layers,
                weights: this.unflattenWeights(model.policyWeights, [[324, 256], [256, 128], [128, 324]])
            },
            valueNetwork: {
                layers: this.createValueNetwork().layers,
                weights: this.unflattenWeights(model.valueWeights, [[324, 256], [256, 128], [128, 1]])
            },
            trainingEpisodes: model.trainingEpisodes,
            timestamp: model.timestamp,
            version: model.version
        };
    }

    // 自我对弈：AI自己与自己下棋，收集训练数据
    selfPlay() {
        const tempBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
        const history = [];  // 记录对局过程
        let currentPlayer = 1;
        let gameOver = false;
        let winner = null;
        let moveCount = 0;
        let totalMoveCount = 0;

        while (!gameOver) {
            const state = this.encodeBoard(tempBoard, currentPlayer);
            const policy = this.predictPolicy(state) || this.getRandomPolicy(tempBoard);
            
            let move = this.selectMoveFromPolicy(policy, tempBoard);
            if (!move) {
                gameOver = true;
                break;
            }

            tempBoard[move.y][move.x] = currentPlayer;
            history.push({ state, policy, player: currentPlayer, move });
            
            // 检查是否获胜
            if (this.checkWinOnBoard(tempBoard, move.x, move.y, currentPlayer)) {
                winner = currentPlayer;
                gameOver = true;
                break;
            }

            totalMoveCount++;
            moveCount++;
            // 六子棋规则：第一手1子，之后每手2子
            if (totalMoveCount > 1 && moveCount >= 2) {
                moveCount = 0;
                currentPlayer = currentPlayer === 1 ? 2 : 1;
            }
            
            // 棋盘下满平局
            if (totalMoveCount > BOARD_SIZE * BOARD_SIZE) {
                gameOver = true;
                break;
            }
        }

        // 将对局数据加入训练集
        this.addTrainingData(history, winner);
    }

    // 编码棋盘状态（用于自我对弈）
    encodeBoard(boardState, player) {
        const state = [];
        const opponent = player === 1 ? 2 : 1;
        
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (boardState[y][x] === player) {
                    state.push(1);
                } else if (boardState[y][x] === opponent) {
                    state.push(-1);
                } else {
                    state.push(0);
                }
            }
        }
        return state;
    }

    // 获取随机策略（当网络未训练时使用）
    getRandomPolicy(boardState) {
        const policy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
        let count = 0;
        
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (boardState[y][x] === 0) {
                    policy[y * BOARD_SIZE + x] = 1;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            return policy.map(v => v / count);
        }
        return policy;
    }

    // 根据策略选择落子：带采样的概率选择
    selectMoveFromPolicy(policy, boardState) {
        const candidates = [];
        for (let i = 0; i < policy.length; i++) {
            const x = i % BOARD_SIZE;
            const y = Math.floor(i / BOARD_SIZE);
            if (boardState[y][x] === 0) {
                candidates.push({ x, y, prob: policy[i] });
            }
        }
        
        if (candidates.length === 0) return null;
        
        // 轮盘赌采样
        const total = candidates.reduce((sum, c) => sum + c.prob, 0);
        let r = Math.random() * total;
        
        for (const candidate of candidates) {
            r -= candidate.prob;
            if (r <= 0) {
                return candidate;
            }
        }
        
        return candidates[0];
    }

    // 在指定棋盘上检查获胜
    checkWinOnBoard(boardState, x, y, player) {
        const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
        
        for (const [dx, dy] of directions) {
            let count = 1;
            
            // 向一个方向延伸
            for (let i = 1; i <= 5; i++) {
                const nx = x + dx * i;
                const ny = y + dy * i;
                if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) break;
                if (boardState[ny][nx] !== player) break;
                count++;
            }
            
            // 向相反方向延伸
            for (let i = 1; i <= 5; i++) {
                const nx = x - dx * i;
                const ny = y - dy * i;
                if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) break;
                if (boardState[ny][nx] !== player) break;
                count++;
            }
            
            if (count >= 6) return true;
        }
        
        return false;
    }

    // 添加训练数据：计算每个状态的回报（使用折扣因子）
    addTrainingData(history, winner) {
        for (let i = history.length - 1; i >= 0; i--) {
            const item = history[i];
            const reward = this.calculateReward(item.player, winner);
            const nextValue = i === history.length - 1 ? 0 : history[i + 1].value || 0;
            // 贝尔曼方程：当前价值 = 即时奖励 + 折扣因子*下一个状态的价值
            const value = reward + this.trainingConfig.discountFactor * nextValue;
            
            item.value = value;
            this.trainingData.push(item);
        }
    }

    // 更新网络权重：使用反向传播
    updateNetworks() {
        const batch = this.sampleBatch();
        
        for (let epoch = 0; epoch < this.trainingConfig.epochs; epoch++) {
            this.trainPolicyNetwork(batch);  // 训练策略网络
            this.trainValueNetwork(batch);   // 训练价值网络
        }
        
        this.trainingData = [];  // 清空训练数据
    }

    // 采样批次数据
    sampleBatch() {
        const batch = [];
        for (let i = 0; i < this.trainingConfig.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.trainingData.length);
            batch.push(this.trainingData[idx]);
        }
        return batch;
    }

    // 训练策略网络：最大化采取的动作的概率
    trainPolicyNetwork(batch) {
        if (!this.policyNetwork) {
            this.policyNetwork = this.createPolicyNetwork();
        }
        
        const lr = this.trainingConfig.learningRate;
        
        for (const item of batch) {
            const output = this.forward(this.policyNetwork, item.state);
            const target = [...output];
            
            // 对于采取的动作，将目标概率设为1（类似策略梯度）
            const moveIdx = item.move.y * BOARD_SIZE + item.move.x;
            target[moveIdx] = 1;
            
            this.backpropagate(this.policyNetwork, item.state, target, lr);
        }
    }

    // 训练价值网络：逼近状态价值
    trainValueNetwork(batch) {
        if (!this.valueNetwork) {
            this.valueNetwork = this.createValueNetwork();
        }
        
        const lr = this.trainingConfig.learningRate;
        
        for (const item of batch) {
            const output = this.forward(this.valueNetwork, item.state);
            const target = [item.value];  // 目标为计算的价值
            
            this.backpropagate(this.valueNetwork, item.state, target, lr);
        }
    }

    // 反向传播算法：计算梯度并更新权重
    backpropagate(network, input, target, lr) {
        const outputs = [];
        let current = [...input];
        outputs.push(current);
        
        // 前向传播，记录每层输出
        for (let i = 0; i < network.layers.length; i++) {
            const layer = network.layers[i];
            const weights = network.weights[i];
            const newOutput = [];
            
            for (let j = 0; j < weights.length; j++) {
                let sum = 0;
                for (let k = 0; k < weights[j].length; k++) {
                    sum += current[k] * weights[j][k];
                }
                
                if (layer.activation === 'relu') {
                    newOutput.push(Math.max(0, sum));
                } else if (layer.activation === 'softmax') {
                    newOutput.push(Math.exp(sum));
                } else if (layer.activation === 'tanh') {
                    newOutput.push(Math.tanh(sum));
                } else {
                    newOutput.push(sum);
                }
            }
            
            if (layer.activation === 'softmax') {
                const total = newOutput.reduce((a, b) => a + b, 0);
                current = newOutput.map(v => v / total);
            } else {
                current = newOutput;
            }
            outputs.push(current);
        }
        
        // 计算输出层误差
        let errors = [];
        for (let i = 0; i < target.length; i++) {
            errors.push(target[i] - current[i]);
        }
        
        // 反向传播误差，更新权重
        for (let i = network.layers.length - 1; i >= 0; i--) {
            const layer = network.layers[i];
            const weights = network.weights[i];
            const prevOutput = outputs[i];
            const currOutput = outputs[i + 1];
            
            const newErrors = Array(prevOutput.length).fill(0);
            
            for (let j = 0; j < weights.length; j++) {
                let gradient = errors[j];
                
                // 计算激活函数导数
                if (layer.activation === 'relu') {
                    gradient *= currOutput[j] > 0 ? 1 : 0;
                } else if (layer.activation === 'tanh') {
                    gradient *= (1 - currOutput[j] * currOutput[j]);
                }
                
                // 更新权重
                for (let k = 0; k < weights[j].length; k++) {
                    weights[j][k] += lr * gradient * prevOutput[k];
                    newErrors[k] += gradient * weights[j][k];
                }
            }
            
            errors = newErrors;
        }
    }

    // 保存模型
    saveModel(filename) {
        try {
            const compressed = this.compressModel();
            localStorage.setItem(filename, compressed);
            const size = compressed.length;
            console.log(`模型已保存，大小: ${(size / 1024).toFixed(2)} KB`);
            return true;
        } catch (e) {
            console.error('保存失败:', e.message);
            return false;
        }
    }

    // 加载模型
    loadModel(filename) {
        const saved = localStorage.getItem(filename);
        if (saved) {
            try {
                const model = this.decompressModel(saved);
                this.policyNetwork = model.policyNetwork;
                this.valueNetwork = model.valueNetwork;
                this.trainingEpisodes = model.trainingEpisodes;
                console.log(`模型已加载，已训练 ${this.trainingEpisodes} 局`);
                return true;
            } catch (e) {
                console.error('加载失败:', e.message);
                return false;
            }
        }
        return false;
    }

    // 压缩模型：将权重扁平化并量化为整数（减少存储空间）
    compressModel() {
        const model = {
            policyWeights: this.flattenWeights(this.policyNetwork?.weights || []),
            valueWeights: this.flattenWeights(this.valueNetwork?.weights || []),
            trainingEpisodes: this.trainingEpisodes,
            timestamp: Date.now()
        };
        
        const jsonStr = JSON.stringify(model);
        return this.encodeString(jsonStr);
    }

    // 解压模型
    decompressModel(compressed) {
        const jsonStr = this.decodeString(compressed);
        const model = JSON.parse(jsonStr);
        
        return {
            policyNetwork: {
                layers: this.createPolicyNetwork().layers,
                weights: this.unflattenWeights(model.policyWeights, [[324, 256], [256, 128], [128, 324]])
            },
            valueNetwork: {
                layers: this.createValueNetwork().layers,
                weights: this.unflattenWeights(model.valueWeights, [[324, 256], [256, 128], [128, 1]])
            },
            trainingEpisodes: model.trainingEpisodes
        };
    }

    // 扁平化权重数组：将多维数组转为一维
    flattenWeights(weights) {
        const flat = [];
        for (const layer of weights) {
            for (const neuron of layer) {
                for (const weight of neuron) {
                    // 量化：将浮点数乘以32767后取整，节省空间
                    flat.push(Math.round(weight * 32767));
                }
            }
        }
        return flat;
    }

    // 恢复权重数组
    unflattenWeights(flat, shape) {
        const weights = [];
        let idx = 0;
        
        for (const [inputSize, outputSize] of shape) {
            const layer = [];
            for (let i = 0; i < outputSize; i++) {
                const neuron = [];
                for (let j = 0; j < inputSize; j++) {
                    // 反量化
                    neuron.push(flat[idx++] / 32767);
                }
                layer.push(neuron);
            }
            weights.push(layer);
        }
        return weights;
    }

    // 字符串编码：使用base64
    encodeString(str) {
        const uint8array = new TextEncoder().encode(str);
        let binary = '';
        for (let i = 0; i < uint8array.length; i++) {
            binary += String.fromCharCode(uint8array[i]);
        }
        return btoa(binary);
    }

    // 字符串解码
    decodeString(encoded) {
        const binary = atob(encoded);
        const uint8array = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            uint8array[i] = binary.charCodeAt(i);
        }
        return new TextDecoder().decode(uint8array);
    }
}

// 创建强化学习实例
let rlAgent = new ReinforcementLearning();
// AI模式：'classic'为经典α-β搜索，'rl'为强化学习
let aiMode = 'classic';

// 切换AI模式
function setAIMode(mode) {
    aiMode = mode;
    showMessage(`AI模式已切换为${mode === 'rl' ? '强化学习' : '经典算法'}`);
}

// 启动强化学习训练
function startRLTraining(episodes) {
    showMessage(`开始强化学习训练，共${episodes}局...`);
    
    // 尝试加载检查点
    if (rlAgent.trainingEpisodes > 0) {
        console.log(`当前已有 ${rlAgent.trainingEpisodes} 局训练记录`);
    }
    
    // 带进度回调的训练
    rlAgent.train(episodes, (current, total) => {
        if (current % 100 === 0) {
            console.log(`训练进度: ${current}/${total}`);
        }
    });
    
    // 训练完成后保存最终检查点
    rlAgent.saveCheckpoint();
    showMessage(`训练完成！共训练${rlAgent.trainingEpisodes}局`);
}

// 继续强化学习的辅助函数
function resumeTraining(additionalEpisodes) {
    // 先尝试加载检查点
    if (!rlAgent.loadCheckpoint()) {
        console.log('未找到检查点，将从头开始训练');
    }
    
    const totalEpisodes = rlAgent.trainingEpisodes + additionalEpisodes;
    showMessage(`继续训练，从${rlAgent.trainingEpisodes}局增至${totalEpisodes}局...`);
    
    rlAgent.train(additionalEpisodes, (current, total) => {
        if (current % 100 === 0) {
            const currentTotal = rlAgent.trainingEpisodes - additionalEpisodes + current;
            console.log(`训练进度: ${currentTotal}/${totalEpisodes}`);
        }
    });
    
    // 保存检查点
    rlAgent.saveCheckpoint();
    showMessage(`断点续训完成！共训练${rlAgent.trainingEpisodes}局`);
}

function saveRLModel() {
    if (!rlAgent.saveModel('connect6_rl_model')) {
        console.warn('保存失败，尝试清理旧数据...');
        clearOldModels();
        if (rlAgent.saveModel('connect6_rl_model')) {
            showMessage('强化学习模型已保存（已清理旧数据）');
        } else {
            showMessage('保存失败：存储空间不足，请清理浏览器数据');
        }
    } else {
        showMessage('强化学习模型已保存');
    }
}

function clearOldModels() {
    const keys = Object.keys(localStorage);
    let cleared = 0;
    for (const key of keys) {
        if (key.startsWith('connect6_') || key.includes('model') || key.includes('rl')) {
            localStorage.removeItem(key);
            cleared++;
        }
    }
    console.log(`已清理 ${cleared} 个旧模型文件`);
}

function checkStorage() {
    let total = 0;
    for (const key of Object.keys(localStorage)) {
        const value = localStorage.getItem(key);
        total += value ? value.length : 0;
    }
    const usedKB = (total / 1024).toFixed(2);
    const quotaKB = (5 * 1024).toFixed(2); // 通常浏览器配额约5MB
    console.log(`存储使用: ${usedKB} KB / ${quotaKB} KB`);
    return { used: parseFloat(usedKB), quota: parseFloat(quotaKB) };
}

function loadRLModel() {
    if (rlAgent.loadModel('connect6_rl_model')) {
        showMessage(`强化学习模型已加载，已训练${rlAgent.trainingEpisodes}局`);
    } else {
        showMessage('未找到已保存的模型');
    }
}

// 获取强化学习的落子
function getRLMove() {
    const aiPlayer = getAITurnPlayer();
    const state = rlAgent.encodeState(aiPlayer);
    const move = rlAgent.selectMove(state, false);
    return move;
}

// 开局库 
// 包含多种常见开局策略，用于加快游戏初期的决策速度
const OPENING_BOOK = [
    { name: '天元', moves: [{ x: 9, y: 9 }] },
    { name: '天元+星位', moves: [{ x: 9, y: 9 }, { x: 10, y: 10 }] },
    { name: '天元+小跳', moves: [{ x: 9, y: 9 }, { x: 8, y: 10 }] },
    { name: '天元+反向小跳', moves: [{ x: 9, y: 9 }, { x: 10, y: 8 }] },
    { name: '横向展开', moves: [{ x: 8, y: 9 }, { x: 10, y: 9 }] },
    { name: '纵向展开', moves: [{ x: 9, y: 8 }, { x: 9, y: 10 }] },
    { name: '主对角线', moves: [{ x: 8, y: 8 }, { x: 10, y: 10 }] },
    { name: '副对角线', moves: [{ x: 10, y: 8 }, { x: 8, y: 10 }] },
    { name: '中国流', moves: [{ x: 7, y: 9 }, { x: 9, y: 7 }] },
    { name: '小林流', moves: [{ x: 9, y: 9 }, { x: 7, y: 8 }] },
    { name: '星位开局', moves: [{ x: 10, y: 10 }, { x: 8, y: 8 }] },
    { name: '错小目', moves: [{ x: 8, y: 9 }, { x: 9, y: 8 }] },
    { name: '三连星', moves: [{ x: 7, y: 9 }, { x: 9, y: 9 }, { x: 11, y: 9 }] },
    { name: '对角星', moves: [{ x: 7, y: 7 }, { x: 11, y: 11 }] },
    { name: '侧挂', moves: [{ x: 9, y: 9 }, { x: 6, y: 9 }] },
    { name: '高挂', moves: [{ x: 9, y: 9 }, { x: 9, y: 6 }] }
];

//  全局游戏状态
let board = [];  // 棋盘状态，0为空，1为黑，2为白
let currentPlayer = 1; // 当前玩家
let gameOver = false; // 游戏是否结束
let winner = null;  // 获胜者
let moveCount = 0; // 当前回合已下子数
let totalMoveCount = 0; // 总步数
let turnNumber = 1; // 回合数
let lastMove = null; // 最后一步

let aiDifficulty = 'easy'; // AI难度
let aiProfile = 'balanced'; // AI策略配置
let firstPlayer = 'human'; // 先手玩家
let gameMode = 'ai'; // 游戏模式：人机或人人
let lastAIDecision = '等待本回合分析';
let coachSuggestion = '当前暂无建议。';
let coachPanelVisible = false;
let opponentProfile = null;

let moveHistory = []; // 历史记录
let viewedHistoryIndex = null; // 浏览历史的索引
let notationEnabled = false;

let blackTime = 900; // 黑方时间（秒）
let whiteTime = 900; // 白方时间（秒）
let timerInterval = null; // 计时器
let isTimerRunning = false;
let timerStartTime = null;
let timerStartValue = 0;

const GAME_CODE = 'C6';
const GAME_NAME = 'Connect6';

// ========== 状态辅助函数 ==========
function updateModeButtons(selectedButton, buttonIds) {
    buttonIds.forEach(id => {
        const button = document.getElementById(id);
        if (button) {
            button.classList.toggle('active', button === selectedButton);
        }
    });
}

// 获取AI执子方
function getAITurnPlayer() {
    return firstPlayer === 'ai' ? 1 : 2;
}

// 获取当前AI策略配置
function getCurrentAIProfile() {
    return AI_PROFILES[aiProfile] || AI_PROFILES.balanced;
}

// 检查是否是AI的回合
function isAITurn() {
    return gameMode === 'ai' && currentPlayer === getAITurnPlayer();
}

// 获取当前回合已下子数
function getStonesPlayedInCurrentTurn(totalMoves) {
    if (totalMoves <= 0) return 0;
    if (totalMoves === 1) return 1;
    return totalMoves % 2 === 0 ? 1 : 2;
}

// 检查是否在浏览历史
function isBrowsingHistory() {
    return viewedHistoryIndex !== null;
}

// 验证棋谱格式
function validateMoveHistory(history) {
    if (!Array.isArray(history)) {
        throw new Error('无效的棋谱格式');
    }

    const seen = new Set();

    history.forEach((move, index) => {
        if (!move || !Number.isInteger(move.x) || !Number.isInteger(move.y)) {
            throw new Error(`第${index + 1}步坐标无效`);
        }

        if (!isValid(move.x, move.y)) {
            throw new Error(`第${index + 1}步超出棋盘范围`);
        }

        if (move.player !== 1 && move.player !== 2) {
            throw new Error(`第${index + 1}步执子方无效`);
        }

        const key = `${move.x},${move.y}`;
        if (seen.has(key)) {
            throw new Error(`第${index + 1}步与之前落子重复`);
        }

        seen.add(key);
    });
}

// 从历史记录重建棋盘状态
function rebuildStateFromHistory(history, upToIndex = history.length - 1) {
    board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
    currentPlayer = 1;
    gameOver = false;
    winner = null;
    moveCount = 0;
    totalMoveCount = 0;
    turnNumber = 1;
    lastMove = null;

    if (upToIndex < 0) return;

    for (let i = 0; i <= upToIndex; i++) {
        const move = history[i];
        board[move.y][move.x] = move.player;
        lastMove = { x: move.x, y: move.y };
        totalMoveCount++;

        if (checkWin(move.x, move.y, move.player)) {
            gameOver = true;
            winner = move.player;
        }

        if (gameOver) {
            currentPlayer = move.player;
            moveCount = getStonesPlayedInCurrentTurn(totalMoveCount);
            turnNumber = Math.floor(Math.max(totalMoveCount - 1, 0) / 2) + 1;
            break;
        }

        const isFirstMove = totalMoveCount === 1;
        const shouldSwitch = isFirstMove || moveCount + 1 >= 2;

        if (shouldSwitch) {
            currentPlayer = currentPlayer === 1 ? 2 : 1;
            moveCount = 0;
            if (currentPlayer === 1) {
                turnNumber++;
            }
        } else {
            moveCount++;
        }
    }
}

// ========== UI 辅助函数 ==========
function updateHistoryModeBadge() {
    const badge = document.getElementById('historyModeBadge');
    if (!badge) return;

    if (isBrowsingHistory()) {
        badge.textContent = `浏览中: 第 ${viewedHistoryIndex + 1} 步`;
        badge.classList.add('browsing');
    } else {
        badge.textContent = '当前局面';
        badge.classList.remove('browsing');
    }
}

function updateAIInsight(message = lastAIDecision) {
    const insightEl = document.getElementById('aiInsight');
    if (!insightEl) return;
    insightEl.textContent = `AI决策说明：${message}`;
    lastAIDecision = message;
}

function updateCoachSuggestion(message = coachSuggestion) {
    const coachContent = document.getElementById('coachContent');
    if (!coachContent) return;
    coachContent.textContent = message;
    coachSuggestion = message;
}

function toggleCoachPanel(forceVisible = null) {
    const panel = document.getElementById('coachPanel');
    if (!panel) return;

    coachPanelVisible = forceVisible === null ? !coachPanelVisible : forceVisible;
    panel.hidden = !coachPanelVisible;
}

// 生成教练建议：分析对手风格并给出建议
function generateCoachSuggestion() {
    if (!gameOver || gameMode !== 'ai') {
        updateCoachSuggestion('当前暂无建议。');
        return;
    }

    const aiPlayer = getAITurnPlayer();
    const aiWon = winner === aiPlayer;
    const profile = analyzeOpponentProfile();
    opponentProfile = profile;
    const styleHint = profile.summary;

    if (winner === null) {
        updateCoachSuggestion(`本局形成和局。对手风格：${styleHint}。建议下一局保持“均衡”作为默认策略。`);
        return;
    }

    if (aiWon) {
        if (aiProfile === 'balanced') {
            updateCoachSuggestion(`本局取胜。对手风格：${styleHint}。若对手继续偏保守，可尝试切到“进攻”扩大压制。`);
        } else {
            updateCoachSuggestion(`本局取胜。对手风格：${styleHint}。建议下一局先保持当前策略，除非对手明显改变风格。`);
        }
        return;
    }

    if (aiProfile === 'aggressive') {
        updateCoachSuggestion(`本局失利。对手风格：${styleHint}。当前为“进攻”策略，建议下一局切换为“稳健”，优先降低送机会风险。`);
    } else if (aiProfile === 'balanced') {
        updateCoachSuggestion(`本局失利。对手风格：${styleHint}。建议下一局先尝试“稳健”策略，提升防守覆盖和组合风险过滤。`);
    } else {
        updateCoachSuggestion(`本局失利且已使用“稳健”策略。对手风格：${styleHint}。建议保留稳健，重点观察对手是否持续偏边路或强攻。`);
    }
}

// 分析对手下棋风格
function analyzeOpponentProfile() {
    const aiPlayer = getAITurnPlayer();
    const opponentPlayer = aiPlayer === 1 ? 2 : 1;
    const opponentMoves = moveHistory.filter(move => move.player === opponentPlayer);

    if (opponentMoves.length === 0) {
        return { summary: '样本不足' };
    }

    const center = Math.floor(BOARD_SIZE / 2);
    let centerMoves = 0;
    let edgeMoves = 0;
    let closeFollowMoves = 0;

    opponentMoves.forEach((move, index) => {
        const distToCenter = Math.abs(move.x - center) + Math.abs(move.y - center);
        if (distToCenter <= 4) centerMoves++;
        if (move.x <= 3 || move.y <= 3 || move.x >= BOARD_SIZE - 4 || move.y >= BOARD_SIZE - 4) edgeMoves++;

        if (index > 0) {
            const prev = opponentMoves[index - 1];
            if (Math.max(Math.abs(move.x - prev.x), Math.abs(move.y - prev.y)) <= 2) {
                closeFollowMoves++;
            }
        }
    });

    const centerRatio = centerMoves / opponentMoves.length;
    const edgeRatio = edgeMoves / opponentMoves.length;
    const closeRatio = opponentMoves.length > 1 ? closeFollowMoves / (opponentMoves.length - 1) : 0;

    if (edgeRatio >= 0.35) {
        return { summary: '偏边路展开' };
    }
    if (centerRatio >= 0.45) {
        return { summary: '偏中心争夺' };
    }
    if (closeRatio >= 0.55) {
        return { summary: '偏局部缠斗' };
    }

    return { summary: '风格较均衡' };
}

function updateActionButtons() {
    const aiButton = document.getElementById('aiBtn');
    const undoButton = document.getElementById('undoBtn');
    const continueButton = document.getElementById('continueBtn');
    const returnLatestButton = document.getElementById('returnLatestBtn');

    if (aiButton) {
        aiButton.disabled = gameMode !== 'ai' || gameOver || !isAITurn() || isBrowsingHistory();
    }

    if (undoButton) {
        undoButton.disabled = moveHistory.length === 0 || isBrowsingHistory();
    }

    if (continueButton) {
        continueButton.disabled = !isBrowsingHistory();
    }

    if (returnLatestButton) {
        returnLatestButton.disabled = !isBrowsingHistory();
    }
}

function showMessage(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ========== 计时器系统 ==========
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function getTeamNames() {
    const blackTeam = document.getElementById('blackTeamInput')?.value.trim() || '黑方参赛队';
    const whiteTeam = document.getElementById('whiteTeamInput')?.value.trim() || '白方参赛队';
    return { blackTeam, whiteTeam };
}

function formatTimestampForFilename(date = new Date()) {
    const year = date.getFullYear().toString();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hour = String(date.getHours()).padStart(2, '0');
    const minute = String(date.getMinutes()).padStart(2, '0');
    return `${year}${month}${day}${hour}${minute}`;
}

function getWinnerLabel() {
    if (!gameOver || winner === null) {
        return '未结束';
    }
    return winner === 1 ? '先手胜' : '后手胜';
}

function buildExportFilename() {
    const { blackTeam, whiteTeam } = getTeamNames();
    const resultLabel = getWinnerLabel();
    const timestamp = formatTimestampForFilename();
    return `${GAME_CODE}-${blackTeam} vs ${whiteTeam}-${resultLabel}-${timestamp}.txt`;
}

function buildExportText() {
    const { blackTeam, whiteTeam } = getTeamNames();
    const lines = [
        `棋种代码: ${GAME_CODE}`,
        `棋种名称: ${GAME_NAME}`,
        `先手队名: ${blackTeam}`,
        `后手队名: ${whiteTeam}`,
        `先手颜色: 黑方`,
        `结果: ${getWinnerLabel()}`,
        `总步数: ${totalMoveCount}`,
        `回合数: ${turnNumber}`,
        `时间戳: ${formatTimestampForFilename()}`,
        '落子记录:'
    ];

    moveHistory.forEach((move, index) => {
        lines.push(`${index + 1}. ${move.player === 1 ? '黑' : '白'} ${displayCoord(move.x)},${displayCoord(move.y)}`);
    });

    return lines.join('\n');
}

function parseImportedText(content) {
    const lines = content.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    const moveStart = lines.findIndex(line => line === '落子记录:');

    if (moveStart === -1) {
        throw new Error('棋谱缺少落子记录段');
    }

    const metadata = {};
    for (let i = 0; i < moveStart; i++) {
        const parts = lines[i].split(':');
        if (parts.length >= 2) {
            metadata[parts[0].trim()] = parts.slice(1).join(':').trim();
        }
    }

    if (metadata['棋种代码'] !== GAME_CODE) {
        throw new Error(`仅支持 ${GAME_CODE} 棋谱导入`);
    }

    const parsedMoves = lines.slice(moveStart + 1).map((line, index) => {
        const match = line.match(/^\d+\.\s*(黑|白)\s*(\d+),(\d+)$/);
        if (!match) {
            throw new Error(`第${index + 1}条落子记录格式无效`);
        }

        const parsedX = Number(match[2]);
        const parsedY = Number(match[3]);
        
        return {
            player: match[1] === '黑' ? 1 : 2,
            x: parsedX >= 1 && parsedX <= 19 ? parsedX - 1 : parsedX,
            y: parsedY >= 1 && parsedY <= 19 ? parsedY - 1 : parsedY
        };
    });

    validateMoveHistory(parsedMoves);

    return {
        moveHistory: parsedMoves,
        blackTeam: metadata['先手队名'] || '黑方参赛队',
        whiteTeam: metadata['后手队名'] || '白方参赛队'
    };
}

function updateTimerDisplay() {
    const blackTimerEl = document.getElementById('blackTimer');
    const whiteTimerEl = document.getElementById('whiteTimer');

    if (blackTimerEl) {
        blackTimerEl.textContent = formatTime(blackTime);
        blackTimerEl.className = `timer ${blackTime < 10 ? 'warning' : ''}`;
    }

    if (whiteTimerEl) {
        whiteTimerEl.textContent = formatTime(whiteTime);
        whiteTimerEl.className = `timer ${whiteTime < 10 ? 'warning' : ''}`;
    }
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    isTimerRunning = false;
}

function resetTimer() {
    stopTimer();
    blackTime = 900;
    whiteTime = 900;
    updateTimerDisplay();
}

function handleTimeout(player) {
    stopTimer();
    gameOver = true;
    winner = player === 1 ? 2 : 1;
    updateStatus();
    updateStats();
    updateActionButtons();
    generateCoachSuggestion();
    showMessage(`${player === 1 ? '黑方' : '白方'}超时，对方获胜！`);
}

function startTimer() {
    if (isTimerRunning) return;

    isTimerRunning = true;
    timerStartTime = Date.now();
    timerStartValue = currentPlayer === 1 ? blackTime : whiteTime;

    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - timerStartTime) / 1000);
        const remaining = Math.max(0, timerStartValue - elapsed);

        if (currentPlayer === 1) {
            blackTime = remaining;
        } else {
            whiteTime = remaining;
        }

        if (remaining <= 0) {
            stopTimer();
            handleTimeout(currentPlayer);
        }

        updateTimerDisplay();
    }, 100);
}

// ========== 渲染与显示 ==========
function displayCoord(internal) {
    return internal + 1;
}

function internalCoord(display) {
    return display - 1;
}

function renderCoordinateLabels() {
    const xAxisEl = document.getElementById('xAxis');
    const yAxisEl = document.getElementById('yAxis');
    
    xAxisEl.innerHTML = '';
    yAxisEl.innerHTML = '';
    
    for (let i = 1; i <= BOARD_SIZE; i++) {
        const xSpan = document.createElement('span');
        xSpan.textContent = i;
        xAxisEl.appendChild(xSpan);
        
        const ySpan = document.createElement('span');
        ySpan.textContent = i;
        yAxisEl.appendChild(ySpan);
    }
}

function renderBoard() {
    const boardEl = document.getElementById('board');
    boardEl.innerHTML = '';
    
    renderCoordinateLabels();
    
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.x = x;
            cell.dataset.y = y;
            cell.onclick = !gameOver ? () => handleCellClick(x, y) : null;

            if (board[y][x] !== 0) {
                const piece = document.createElement('div');
                piece.className = `piece ${board[y][x] === 1 ? 'black' : 'white'}`;

                if (lastMove && lastMove.x === x && lastMove.y === y) {
                    piece.classList.add('last-move');
                }

                if (notationEnabled) {
                    const notation = document.createElement('div');
                    notation.className = 'piece-notation';
                    const moveIndex = moveHistory.findIndex(move => move.x === x && move.y === y);
                    if (moveIndex !== -1) {
                        notation.textContent = moveIndex + 1;
                        notation.style.color = board[y][x] === 1 ? 'white' : 'black';
                        piece.appendChild(notation);
                    }
                }

                cell.appendChild(piece);
            }

            boardEl.appendChild(cell);
        }
    }
}

function updateHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';

    moveHistory.forEach((move, index) => {
        const li = document.createElement('li');
        li.className = 'history-item';

        if ((isBrowsingHistory() && viewedHistoryIndex === index) || (!isBrowsingHistory() && index === moveHistory.length - 1)) {
            li.classList.add('active');
        }

        li.textContent = `第${index + 1}步: ${move.player === 1 ? '黑' : '白'} ${displayCoord(move.x)},${displayCoord(move.y)}`;
        li.onclick = () => jumpToMove(index);
        historyList.appendChild(li);
    });

    updateActionButtons();
    updateHistoryModeBadge();
}

function updateStatus() {
    const statusEl = document.getElementById('status');
    const moveInfoEl = document.getElementById('moveInfo');

    if (gameOver) {
        if (winner) {
            statusEl.textContent = winner === 1 ? '黑棋获胜！' : '白棋获胜！';
            statusEl.className = 'status-value winner';
        } else {
            statusEl.textContent = '平局';
            statusEl.className = 'status-value';
        }

        moveInfoEl.textContent = '游戏结束';
        updateAIInsight(lastAIDecision);
        return;
    }

    const isFirstMove = totalMoveCount === 0;
    const movesNeeded = isFirstMove ? 1 : (2 - moveCount);

    statusEl.textContent = currentPlayer === 1 ? '黑棋' : '白棋';
    statusEl.className = `status-value ${currentPlayer === 1 ? 'black-turn' : 'white-turn'}`;

    if (isBrowsingHistory()) {
        moveInfoEl.textContent = `历史浏览：第 ${viewedHistoryIndex + 1} 步`;
        updateAIInsight('当前处于历史浏览模式');
        return;
    }

    if (gameMode === 'human') {
        if (isFirstMove) {
            moveInfoEl.textContent = '黑方回合：下 1 子';
        } else {
            moveInfoEl.textContent = `${currentPlayer === 1 ? '黑方' : '白方'}回合：还需下 ${movesNeeded} 子`;
        }
    } else {
        const turnText = isAITurn() ? 'AI' : '人类';
        if (isFirstMove) {
            moveInfoEl.textContent = `${turnText}回合：黑方第一手下 1 子`;
        } else {
            moveInfoEl.textContent = `${turnText}回合：还需下 ${movesNeeded} 子`;
        }
    }

    updateAIInsight(lastAIDecision);
}

function updateStats() {
    let blackCount = 0;
    let whiteCount = 0;

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === 1) blackCount++;
            else if (board[y][x] === 2) whiteCount++;
        }
    }

    document.getElementById('blackCount').textContent = blackCount;
    document.getElementById('whiteCount').textContent = whiteCount;
    document.getElementById('totalMoves').textContent = totalMoveCount;
    document.getElementById('turnCount').textContent = turnNumber;
}

// ========== 规则与棋盘逻辑 ==========
// 检查坐标是否在棋盘范围内
function isValid(x, y) {
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
}

// 检查在(x, y)位置落子后是否获胜
// 检查四个方向（横、竖、左斜、右斜）是否连成6子
function checkWin(x, y, player) {
    const directions = [
        [1, 0],    // 水平方向
        [0, 1],    // 垂直方向
        [1, 1],    // 主对角线方向
        [1, -1]    // 副对角线方向
    ];

    for (const [dx, dy] of directions) {
        let count = 1;
        let positions = [[x, y]];

        // 向一个方向搜索
        let nx = x + dx;
        let ny = y + dy;
        while (isValid(nx, ny) && board[ny][nx] === player) {
            count++;
            positions.push([nx, ny]);
            nx += dx;
            ny += dy;
        }

        // 向相反方向搜索
        nx = x - dx;
        ny = y - dy;
        while (isValid(nx, ny) && board[ny][nx] === player) {
            count++;
            positions.push([nx, ny]);
            nx -= dx;
            ny -= dy;
        }

        // 如果连成6子或更多，返回获胜位置
        if (count >= WIN_COUNT) {
            return positions;
        }
    }

    return null;
}

function highlightWinningLine(x, y, player) {
    const winningPositions = checkWin(x, y, player);
    if (!winningPositions) return;

    const boardEl = document.getElementById('board');
    boardEl.innerHTML = '';

    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.x = col;
            cell.dataset.y = row;

            if (board[row][col] !== 0) {
                const piece = document.createElement('div');
                piece.className = `piece ${board[row][col] === 1 ? 'black' : 'white'}`;

                const isWinning = winningPositions.some(([wx, wy]) => wx === col && wy === row);
                if (isWinning) {
                    piece.style.boxShadow = '0 0 25px #CD5C5C, 0 0 50px #CD5C5C';
                    piece.style.animation = 'pulse 0.6s infinite';
                }

                cell.appendChild(piece);
            }

            boardEl.appendChild(cell);
        }
    }
}

function makeMove(x, y) {
    if (isBrowsingHistory()) return;

    stopTimer();

    board[y][x] = currentPlayer;
    lastMove = { x, y };
    moveCount++;
    totalMoveCount++;
    moveHistory.push({ x, y, player: currentPlayer });

    renderBoard();
    updateHistory();

    if (checkWin(x, y, currentPlayer)) {
        gameOver = true;
        winner = currentPlayer;
        highlightWinningLine(x, y, currentPlayer);
        updateStatus();
        updateStats();
        generateCoachSuggestion();
        showMessage(`${currentPlayer === 1 ? '黑棋' : '白棋'}获胜！`);
        return;
    }

    if (totalMoveCount >= BOARD_SIZE * BOARD_SIZE) {
        gameOver = true;
        updateStatus();
        updateStats();
        generateCoachSuggestion();
        showMessage('平局');
        return;
    }

    const isFirstMove = totalMoveCount === 1;
    const shouldSwitch = isFirstMove || moveCount >= 2;

    if (shouldSwitch) {
        currentPlayer = currentPlayer === 1 ? 2 : 1;
        moveCount = 0;
        if (currentPlayer === 1) {
            turnNumber++;
        }

        if (!gameOver) {
            startTimer();
        }
    } else {
        startTimer();
    }

    updateStatus();
    updateStats();
    updateActionButtons();
}

function handleCellClick(x, y) {
    if (gameOver || board[y][x] !== 0) return;

    if (isBrowsingHistory()) {
        showMessage('当前处于历史浏览模式，请先点击“从此继续”');
        return;
    }

    if (gameMode === 'human') {
        startTimer();
        makeMove(x, y);
        return;
    }

    const isHumanTurn = currentPlayer !== getAITurnPlayer();
    if (!isHumanTurn) {
        showMessage('现在是AI回合，请点击“AI 走棋”按钮');
        return;
    }

    startTimer();
    makeMove(x, y);
}

// AI核心算法
// 检查位置(x, y)附近是否有棋子（剪枝优化用）
// 用于减少搜索空间，只考虑有棋子附近的位置
function hasNearbyPiece(x, y, distance = 2) {
    for (let dy = -distance; dy <= distance; dy++) {
        for (let dx = -distance; dx <= distance; dx++) {
            if (dx === 0 && dy === 0) continue;  // 跳过当前位置
            const nx = x + dx;
            const ny = y + dy;
            if (isValid(nx, ny) && board[ny][nx] !== 0) {
                return true;
            }
        }
    }
    return false;
}

// 扩展范围检查附近是否有棋子
function hasNearbyPieceExtended(x, y, distance = 4) {
    for (let dy = -distance; dy <= distance; dy++) {
        for (let dx = -distance; dx <= distance; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (isValid(nx, ny) && board[ny][nx] !== 0) {
                return true;
            }
        }
    }
    return false;
}

// 检查是否为开局候选位置（棋盘中心附近）
function isOpeningCandidate(x, y) {
    const centerX = Math.floor(BOARD_SIZE / 2);
    const centerY = Math.floor(BOARD_SIZE / 2);
    return Math.abs(x - centerX) <= 3 && Math.abs(y - centerY) <= 3;
}

// 检查是否为战术要点（有3连或更多的位置）
function isTacticalPoint(x, y) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    const aiPlayer = getAITurnPlayer();
    const opponentPlayer = aiPlayer === 1 ? 2 : 1;

    for (const [dx, dy] of directions) {
        let aiCount = 0;
        let oppCount = 0;
        
        // 检查AI方的连线
        for (let dir = -1; dir <= 1; dir += 2) {
            let nx = x + dx * dir;
            let ny = y + dy * dir;
            while (isValid(nx, ny) && board[ny][nx] === aiPlayer) {
                aiCount++;
                nx += dx * dir;
                ny += dy * dir;
            }
        }
        
        // 检查对手的连线
        for (let dir = -1; dir <= 1; dir += 2) {
            let nx = x + dx * dir;
            let ny = y + dy * dir;
            while (isValid(nx, ny) && board[ny][nx] === opponentPlayer) {
                oppCount++;
                nx += dx * dir;
                ny += dy * dir;
            }
        }

        if (aiCount >= 3 || oppCount >= 3) {
            return true;
        }
    }
    return false;
}

// 获取候选落子位置
// 只考虑有意义的位置，大幅减少搜索空间
function getCandidateMoves(limit = getCandidateLimit()) {
    const candidates = [];
    const recentMoves = moveHistory.slice(-8);  // 考虑最近8步
    let searchDistance;
    if (aiDifficulty === 'master') searchDistance = 7;
    else if (aiDifficulty === 'hard') searchDistance = 6;
    else if (aiDifficulty === 'medium') searchDistance = 4;
    else searchDistance = 3;

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;  // 跳过已有棋子位置

            let isCandidate = false;

            // 开局阶段：只考虑中心附近
            if (totalMoveCount < 2) {
                isCandidate = isOpeningCandidate(x, y);
            } 
            // 早期：考虑3格范围内
            else if (totalMoveCount < 6) {
                isCandidate = hasNearbyPiece(x, y, 3);
            } 
            // 中后期：根据难度确定范围
            else {
                isCandidate = hasNearbyPieceExtended(x, y, searchDistance);
            }

            // 补充：最近落子附近的位置
            if (!isCandidate && recentMoves.length > 0) {
                isCandidate = recentMoves.some(move => Math.abs(move.x - x) <= searchDistance && Math.abs(move.y - y) <= searchDistance);
            }

            // 补充：战术要点位置
            if (!isCandidate && totalMoveCount >= 6) {
                isCandidate = isTacticalPoint(x, y);
            }

            if (isCandidate) {
                candidates.push({ x, y });
            }
        }
    }

    if (candidates.length === 0) {
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] === 0) {
                    candidates.push({ x, y });
                }
            }
        }
    }

    if (candidates.length <= limit) {
        return candidates;
    }

    return candidates
        .map(move => ({ ...move, priority: evaluatePosition(move.x, move.y) }))
        .sort((a, b) => b.priority - a.priority)
        .slice(0, limit)
        .map(({ x, y }) => ({ x, y }));
}

// 评估单个方向的棋型
// 统计连子数、两端开放情况、跳连情况
function evaluateLinePattern(x, y, player, dx, dy) {
    let total = 1;          // 连子总数（包括当前位置）
    let openEnds = 0;       // 两端开放数（0-2）
    let jumpSupport = 0;    // 跳连支持数

    for (let dir = -1; dir <= 1; dir += 2) {  // 两个方向
        let nx = x + dx * dir;
        let ny = y + dy * dir;
        let count = 0;

        // 统计连子数
        while (isValid(nx, ny) && board[ny][nx] === player) {
            count++;
            nx += dx * dir;
            ny += dy * dir;
        }

        total += count;

        // 检查端点是否开放
        if (isValid(nx, ny) && board[ny][nx] === 0) {
            openEnds++;

            // 检查是否有跳连（如：X_X，中间空一格）
            const jumpX = nx + dx * dir;
            const jumpY = ny + dy * dir;
            if (isValid(jumpX, jumpY) && board[jumpY][jumpX] === player) {
                jumpSupport++;
            }
        }
    }

    return { total, openEnds, jumpSupport };
}

// 为棋型打分
// 根据连子数、开放端、跳连给出分数
function scorePattern(pattern, mode, isDefense = false) {
    // 防守时增加威胁权重
    const threatScale = isDefense
        ? (mode === 'easy' ? 1.8 : mode === 'medium' ? 1.6 : 1.4)
        : 1;

    let score = 0;

    // 各种棋型的分数（从高到低）
    if (pattern.total >= 6) score = 10000000;           // 六连：必胜
    else if (pattern.total === 5 && pattern.openEnds >= 2) score = 2000000;  // 活五
    else if (pattern.total === 5 && pattern.openEnds === 1) score = 800000;  // 冲五
    else if (pattern.total === 4 && pattern.openEnds === 2) score = 500000;  // 活四
    else if (pattern.total === 4 && pattern.openEnds === 1) score = 150000;  // 冲四
    else if (pattern.total === 3 && pattern.openEnds === 2) score = 50000;   // 活三
    else if (pattern.total === 3 && pattern.openEnds === 1) score = 12000;   // 冲三
    else if (pattern.total === 2 && pattern.openEnds === 2) score = 3000;    // 活二
    else if (pattern.total === 2 && pattern.openEnds === 1) score = 800;     // 冲二
    else if (pattern.total === 1 && pattern.openEnds === 2) score = 150;     // 活一

    // 跳连加分
    if (pattern.jumpSupport > 0) {
        score += pattern.jumpSupport * (pattern.total >= 3 ? 1500 : 300);
    }

    // 活三及以上额外加分
    if (pattern.total >= 3 && pattern.openEnds >= 2) {
        score += 3000;
    }

    return Math.round(score * threatScale);
}

// 检测双重威胁（两个方向都有威胁）
function detectDoubleThreat(x, y, player) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let threatCount = 0;

    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, player, dx, dy);
        // 四连或活三都算威胁
        if ((pattern.total >= 4 && pattern.openEnds >= 1) || 
            (pattern.total >= 3 && pattern.openEnds === 2)) {
            threatCount++;
        }
    }

    return threatCount >= 2;
}

// 检测四三威胁（一个方向有四连，另一个方向有活三）
function detectFourThreeThreat(x, y, player) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let hasFour = false;
    let hasThree = false;

    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, player, dx, dy);
        if (pattern.total >= 4 && pattern.openEnds >= 1) {
            hasFour = true;
        } else if (pattern.total >= 3 && pattern.openEnds === 2) {
            hasThree = true;
        }
    }

    return hasFour && hasThree;
}

function detectThreatChain(x, y, player) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    const threats = [];

    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, player, dx, dy);
        if (pattern.total >= 3 && pattern.openEnds > 0) {
            threats.push({ direction: [dx, dy], pattern });
        }
    }

    if (threats.length < 2) return 0;

    let chainScore = 0;

    for (let i = 0; i < threats.length; i++) {
        for (let j = i + 1; j < threats.length; j++) {
            const dir1 = threats[i].direction;
            const dir2 = threats[j].direction;

            const dotProduct = dir1[0] * dir2[0] + dir1[1] * dir2[1];
            const angle = Math.abs(dotProduct);

            let comboScore = 0;
            const totalThreat = threats[i].pattern.total + threats[j].pattern.total;
            const totalOpen = threats[i].pattern.openEnds + threats[j].pattern.openEnds;

            if (totalThreat >= 8) {
                comboScore = 800000;
            } else if (totalThreat >= 7) {
                comboScore = 500000;
            } else if (totalThreat >= 6) {
                comboScore = 300000;
            } else if (totalThreat >= 5) {
                comboScore = 150000;
            } else {
                comboScore = 50000;
            }

            if (totalOpen >= 3) {
                comboScore *= 1.5;
            } else if (totalOpen === 2) {
                comboScore *= 1.2;
            }

            if (angle === 0) {
                comboScore *= 1.3;
            } else if (angle === 1) {
                comboScore *= 1.1;
            }

            chainScore += comboScore;
        }
    }

    if (threats.length >= 3) {
        chainScore *= 2.5;
    } else if (threats.length >= 2) {
        chainScore *= 1.8;
    }

    return Math.round(chainScore);
}

function detectThreatChainSimple(x, y, player) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let threatCount = 0;
    let totalThreatValue = 0;

    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, player, dx, dy);
        if (pattern.total >= 3 && pattern.openEnds > 0) {
            threatCount++;
            if (pattern.total >= 4) {
                totalThreatValue += 200000;
            } else {
                totalThreatValue += 30000;
            }
        }
    }

    if (threatCount >= 3) {
        return totalThreatValue * 3;
    } else if (threatCount === 2) {
        return totalThreatValue * 2;
    }

    return 0;
}

function scoreMoveSynergy(x, y, context) {
    if (!context || !context.firstMove) return 0;

    const { firstMove } = context;
    const dx = Math.abs(firstMove.x - x);
    const dy = Math.abs(firstMove.y - y);
    const sameLine = firstMove.x === x || firstMove.y === y || dx === dy;
    const distance = Math.max(dx, dy);

    let score = 0;
    if (sameLine) score += 180;
    if (distance <= 2) score += 220;
    else if (distance <= 4) score += 90;
    else if (distance >= 7) score -= 120;

    return score;
}

function evaluateOpponentBestResponse(limit = 3) {
    const aiPlayer = getAITurnPlayer();
    const opponentPlayer = aiPlayer === 1 ? 2 : 1;
    const candidateMoves = getCandidateMoves(8);
    let maxThreat = 0;

    for (const { x, y } of candidateMoves.slice(0, limit)) {
        board[y][x] = opponentPlayer;

        if (checkWin(x, y, opponentPlayer)) {
            board[y][x] = 0;
            return 100000;
        }

        maxThreat = Math.max(maxThreat, evaluatePosition(x, y, { playerOverride: opponentPlayer, defenseOverride: aiPlayer }));
        board[y][x] = 0;
    }

    return maxThreat;
}

function describeAIMove(move, score, context = null) {
    if (!move) return '等待本回合分析';
    if (score >= 100000) return '优先选择直接取胜点';
    if (score >= 90000) return '优先封堵对手的直接胜点';

    if (context?.firstMove) {
        const dx = Math.abs(context.firstMove.x - move.x);
        const dy = Math.abs(context.firstMove.y - move.y);
        if (dx <= 2 && dy <= 2) return '第二子贴近首子，强化局部联动';
        if (context.firstMove.x === move.x || context.firstMove.y === move.y || dx === dy) {
            return '第二子沿同线扩展，保持连续压迫';
        }
    }

    if (score > 12000) return '兼顾进攻与防守，优先处理高威胁区域';
    if (score > 4000) return '围绕现有棋形扩张，提升后续威胁';
    return '选择局部密度更高的位置，保持棋形稳定';
}

function countStrongThreatsForPlayer(player, moves) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let strongThreats = 0;

    for (const move of moves) {
        for (const [dx, dy] of directions) {
            const pattern = evaluateLinePattern(move.x, move.y, player, dx, dy);
            if ((pattern.total >= 4 && pattern.openEnds > 0) || (pattern.total >= 3 && pattern.openEnds === 2)) {
                strongThreats++;
            }
        }
    }

    return strongThreats;
}

function scoreDefensiveCoverage(moves, player, opponentPlayer) {
    const profile = getCurrentAIProfile();
    let score = 0;

    for (const move of moves) {
        board[move.y][move.x] = opponentPlayer;
        const opponentThreat = evaluatePosition(move.x, move.y, { playerOverride: opponentPlayer, defenseOverride: player });
        board[move.y][move.x] = player;
        const blockValue = evaluatePosition(move.x, move.y, { playerOverride: player, defenseOverride: opponentPlayer });
        board[move.y][move.x] = 0;

        if (opponentThreat > 8000) {
            score += profile.defensiveThreatBonus;
        }
        score += Math.floor(blockValue * profile.defensiveCoverageWeight);
    }

    return score;
}

function evaluatePairResponseRisk(firstMove, secondMove, player) {
    const opponentPlayer = player === 1 ? 2 : 1;
    const responseMoves = getCandidateMoves(getSearchMoveLimit() + 4)
        .map(move => ({
            ...move,
            score: evaluatePosition(move.x, move.y, {
                playerOverride: opponentPlayer,
                defenseOverride: player
            })
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, getSearchMoveLimit() + 2);
    let maxRisk = 0;

    for (const responseMove of responseMoves) {
        board[responseMove.y][responseMove.x] = opponentPlayer;

        if (checkWin(responseMove.x, responseMove.y, opponentPlayer)) {
            board[responseMove.y][responseMove.x] = 0;
            return 1000000;
        }

        if (detectFourThreeThreat(responseMove.x, responseMove.y, opponentPlayer)) {
            board[responseMove.y][responseMove.x] = 0;
            return 500000;
        }

        if (detectDoubleThreat(responseMove.x, responseMove.y, opponentPlayer)) {
            board[responseMove.y][responseMove.x] = 0;
            return 300000;
        }

        const riskScore = evaluatePosition(responseMove.x, responseMove.y, {
            playerOverride: opponentPlayer,
            defenseOverride: player
        });

        if (riskScore > maxRisk) {
            maxRisk = riskScore;
        }

        board[responseMove.y][responseMove.x] = 0;
    }

    return maxRisk;
}

function scoreMovePair(firstMove, secondMove, player) {
    let score = firstMove.score + secondMove.score;
    const opponentPlayer = player === 1 ? 2 : 1;
    const profile = getCurrentAIProfile();

    const dx = Math.abs(firstMove.x - secondMove.x);
    const dy = Math.abs(firstMove.y - secondMove.y);
    const sameLine = firstMove.x === secondMove.x || firstMove.y === secondMove.y || dx === dy;
    const distance = Math.max(dx, dy);

    const difficultyMultiplier = aiDifficulty === 'master' ? 2 : (aiDifficulty === 'hard' ? 1.5 : 1);

    if (sameLine) {
        score += profile.sameLineBonus * difficultyMultiplier;
    }

    if (distance <= 1) {
        score += profile.closeDistanceBonus * 1.3;
    } else if (distance <= 2) {
        score += profile.closeDistanceBonus;
    } else if (distance <= 3) {
        score += profile.mediumDistanceBonus * 1.3;
    } else if (distance <= 4) {
        score += profile.mediumDistanceBonus;
    } else if (distance >= 7) {
        score -= profile.farDistancePenalty * difficultyMultiplier;
    }

    board[firstMove.y][firstMove.x] = player;
    board[secondMove.y][secondMove.x] = player;

    if (checkWin(firstMove.x, firstMove.y, player) || checkWin(secondMove.x, secondMove.y, player)) {
        score += 100000;
    }

    if (detectFourThreeThreat(firstMove.x, firstMove.y, player)) {
        score += 45000;
    } else if (detectDoubleThreat(firstMove.x, firstMove.y, player)) {
        score += 35000;
    }

    if (detectFourThreeThreat(secondMove.x, secondMove.y, player)) {
        score += 40000;
    } else if (detectDoubleThreat(secondMove.x, secondMove.y, player)) {
        score += 30000;
    }

    score += Math.floor(evaluatePosition(secondMove.x, secondMove.y, { playerOverride: player }) * 0.1);

    const threatCount = countStrongThreatsForPlayer(player, [firstMove, secondMove]);
    if (threatCount >= 3) {
        score += profile.threatBonusHigh * (aiDifficulty === 'master' ? 1.5 : (aiDifficulty === 'hard' ? 1.3 : 1));
    } else if (threatCount >= 2) {
        score += profile.threatBonusMedium * (aiDifficulty === 'master' ? 1.4 : (aiDifficulty === 'hard' ? 1.2 : 1));
    }

    const defenseScore = scoreDefensiveCoverage([firstMove, secondMove], player, opponentPlayer);
    score += defenseScore;

    const responseRisk = evaluatePairResponseRisk(firstMove, secondMove, player);
    const riskMultiplier = aiDifficulty === 'master' ? 2 : (aiDifficulty === 'hard' ? 1.5 : 1);
    if (responseRisk >= 1000000) {
        score -= profile.pairLossPenalty * 3;
    } else if (responseRisk >= 500000) {
        score -= profile.pairLossPenalty * 1.5;
    } else if (responseRisk >= 300000) {
        score -= profile.pairLossPenalty;
    } else {
        score -= Math.floor(responseRisk * profile.pairRiskWeight * riskMultiplier);
    }

    board[firstMove.y][firstMove.x] = 0;
    board[secondMove.y][secondMove.x] = 0;

    return score;
}

function getOpeningBookMove() {
    if (totalMoveCount >= 6) return null;
    
    const aiPlayer = getAITurnPlayer();
    const aiMoves = moveHistory.filter(m => m.player === aiPlayer);
    const aiMoveCount = aiMoves.length;
    
    if (aiMoveCount >= 3) return null;

    const candidates = OPENING_BOOK.filter(opening => {
        if (opening.moves.length <= aiMoveCount) return false;
        
        for (let i = 0; i < aiMoveCount; i++) {
            const expected = opening.moves[i];
            const actual = aiMoves[i];
            if (!actual || actual.x !== expected.x || actual.y !== expected.y) {
                return false;
            }
        }
        return true;
    });

    if (candidates.length === 0) {
        const allOpenings = OPENING_BOOK.filter(o => o.moves.length > aiMoveCount);
        if (allOpenings.length > 0 && aiMoveCount === 0) {
            const randomIndex = Math.floor(Math.random() * allOpenings.length);
            const nextMove = allOpenings[randomIndex].moves[0];
            if (board[nextMove.y][nextMove.x] === 0) {
                return nextMove;
            }
        }
        return null;
    }

    const randomIndex = Math.floor(Math.random() * candidates.length);
    const chosenOpening = candidates[randomIndex];
    const nextMove = chosenOpening.moves[aiMoveCount];
    
    if (board[nextMove.y][nextMove.x] !== 0) return null;

    if (aiDifficulty === 'easy' && Math.random() < 0.3) {
        return null;
    }

    return nextMove;
}

function findBestTurnMoves() {
    const aiPlayer = getAITurnPlayer();
    const profile = getCurrentAIProfile();
    const stonesLeft = totalMoveCount === 0 ? 1 : (2 - moveCount);
    const useSearch = aiDifficulty === 'master' || aiDifficulty === 'hard' || (aiDifficulty === 'medium' && totalMoveCount >= 8);
    
    console.log(`[AI决策] 回合开始 - 难度:${aiDifficulty}, 使用搜索:${useSearch}, 剩余棋子:${stonesLeft}`);
    
    if (totalMoveCount < 6 && aiDifficulty !== 'easy') {
        const openingMove = getOpeningBookMove();
        if (openingMove) {
            lastAIDecision = '使用开局库策略';
            console.log(`[AI决策] 使用开局库: (${openingMove.x}, ${openingMove.y})`);
            return [{ x: openingMove.x, y: openingMove.y }];
        }
    }

    // 使用深度搜索找到第一子
    const firstMoves = useSearch ? findBestMovesWithSearch() : findBestMoves();
    if (firstMoves.length === 0) {
        console.log(`[AI决策] 错误：没有找到候选落点`);
        return [];
    }

    if (stonesLeft <= 1) {
        const bestMove = firstMoves[0];
        console.log(`[AI决策] 单子模式，选择: (${bestMove.x}, ${bestMove.y}), 分数:${bestMove.score}`);
        lastAIDecision = `选择位置(${bestMove.x}, ${bestMove.y})，评估分数:${Math.round(bestMove.score)}`;
        return [bestMove];
    }

    // 双落子模式：使用深度搜索评估
    let topFirstMoves;
    if (aiDifficulty === 'master') topFirstMoves = firstMoves.slice(0, 10);
    else if (aiDifficulty === 'hard') topFirstMoves = firstMoves.slice(0, 8);
    else if (aiDifficulty === 'medium') topFirstMoves = firstMoves.slice(0, 7);
    else topFirstMoves = firstMoves.slice(0, 6);
    
    console.log(`[AI决策] 双落子模式，评估${topFirstMoves.length}个候选第一子...`);
    
    let bestPlan = null;
    let evaluatedPairs = 0;

    for (const firstMove of topFirstMoves) {
        board[firstMove.y][firstMove.x] = aiPlayer;

        // 使用深度搜索找第二子
        const secondMoves = useSearch 
            ? findBestMovesWithSearch({ firstMove }).filter(move => move.x !== firstMove.x || move.y !== firstMove.y)
            : findBestMoves({ firstMove }).filter(move => move.x !== firstMove.x || move.y !== firstMove.y);
        
        let topSecondMoves;
        if (aiDifficulty === 'master') topSecondMoves = secondMoves.slice(0, 10);
        else if (aiDifficulty === 'hard') topSecondMoves = secondMoves.slice(0, 8);
        else if (aiDifficulty === 'medium') topSecondMoves = secondMoves.slice(0, 7);
        else topSecondMoves = secondMoves.slice(0, 6);

        for (const secondMove of topSecondMoves) {
            let pairScore = scoreMovePair(firstMove, secondMove, aiPlayer);
            evaluatedPairs++;

            if (aiDifficulty === 'master' || aiDifficulty === 'hard') {
                // 评估对手回应风险
                board[firstMove.y][firstMove.x] = aiPlayer;
                board[secondMove.y][secondMove.x] = aiPlayer;
                const responseRisk = evaluatePairResponseRisk(firstMove, secondMove, aiPlayer);
                board[firstMove.y][firstMove.x] = aiPlayer;
                board[secondMove.y][secondMove.x] = 0;

                const penaltyMultiplier = aiDifficulty === 'master' ? 1.5 : 1;
                if (responseRisk >= 100000) {
                    pairScore -= profile.pairLossPenalty * penaltyMultiplier;
                } else {
                    pairScore -= Math.floor(responseRisk * profile.pairRiskWeight * penaltyMultiplier);
                }
            } else if (aiDifficulty === 'medium') {
                const responseRisk = evaluatePairResponseRisk(firstMove, secondMove, aiPlayer);
                if (responseRisk >= 100000) {
                    pairScore -= profile.pairLossPenalty * 0.7;
                }
            }

            if (!bestPlan || pairScore > bestPlan.score) {
                bestPlan = {
                    moves: [firstMove, secondMove],
                    score: pairScore,
                    reason: describeAIMove(secondMove, pairScore, { firstMove })
                };
            }
        }

        board[firstMove.y][firstMove.x] = 0;
    }

    console.log(`[AI决策] 完成，评估了${evaluatedPairs}个落子组合`);

    if (!bestPlan) {
        const fallback = firstMoves[0];
        console.log(`[AI决策] 回退到第一候选: (${fallback.x}, ${fallback.y})`);
        lastAIDecision = `选择位置(${fallback.x}, ${fallback.y})`;
        return [fallback];
    }

    console.log(`[AI决策] 最佳落子: (${bestPlan.moves[0].x}, ${bestPlan.moves[0].y}) + (${bestPlan.moves[1].x}, ${bestPlan.moves[1].y}), 分数:${bestPlan.score}`);
    lastAIDecision = bestPlan.reason;
    return bestPlan.moves;
}

function evaluatePosition(x, y, context = null) {
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    const aiPlayer = context?.playerOverride ?? getAITurnPlayer();
    const opponentPlayer = context?.defenseOverride ?? (aiPlayer === 1 ? 2 : 1);
    let attackScore = 0;
    let defenseScore = 0;
    let multiThreats = 0;

    for (const [dx, dy] of directions) {
        const attackPattern = evaluateLinePattern(x, y, aiPlayer, dx, dy);
        const defensePattern = evaluateLinePattern(x, y, opponentPlayer, dx, dy);

        attackScore += scorePattern(attackPattern, aiDifficulty, false);
        defenseScore += scorePattern(defensePattern, aiDifficulty, true);

        if (attackPattern.total >= 3 && attackPattern.openEnds > 0) multiThreats++;
        if (defensePattern.total >= 3 && defensePattern.openEnds > 0) multiThreats++;
    }

    let score = attackScore + defenseScore;

    board[y][x] = aiPlayer;
    if (detectFourThreeThreat(x, y, aiPlayer)) {
        score += 500000;
    } else if (detectDoubleThreat(x, y, aiPlayer)) {
        score += 300000;
    }

    if (detectFourThreeThreat(x, y, opponentPlayer)) {
        score += 450000;
    } else if (detectDoubleThreat(x, y, opponentPlayer)) {
        score += 250000;
    }
    board[y][x] = 0;

    if (multiThreats >= 2) {
        if (aiDifficulty === 'master') score += 1500;
        else if (aiDifficulty === 'hard') score += 1000;
        else score += 600;
    }

    board[y][x] = aiPlayer;
    const threatChainScore = detectThreatChainSimple(x, y, aiPlayer);
    const opponentThreatChainScore = detectThreatChainSimple(x, y, opponentPlayer);
    board[y][x] = 0;

    score += threatChainScore;
    score += opponentThreatChainScore * 1.1;

    let nearPieces = 0;
    for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (isValid(nx, ny) && board[ny][nx] !== 0) {
                nearPieces++;
            }
        }
    }
    score += nearPieces * 8;

    const centerX = Math.floor(BOARD_SIZE / 2);
    const centerY = Math.floor(BOARD_SIZE / 2);
    const distToCenter = Math.abs(x - centerX) + Math.abs(y - centerY);
    const centerWeight = totalMoveCount < 12 ? 20 : totalMoveCount < 30 ? 10 : 5;
    score += Math.max(0, centerWeight - distToCenter);

    return score;
}

function evaluatePositionLegacy(x, y, player = getAITurnPlayer()) {
    const opponentPlayer = player === 1 ? 2 : 1;
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let score = 0;

    for (const [dx, dy] of directions) {
        let aiCount = 0;
        let opponentCount = 0;

        for (let dir = -1; dir <= 1; dir += 2) {
            let nx = x + dx * dir;
            let ny = y + dy * dir;

            while (isValid(nx, ny) && board[ny][nx] === player) {
                aiCount++;
                nx += dx * dir;
                ny += dy * dir;
            }
        }

        for (let dir = -1; dir <= 1; dir += 2) {
            let nx = x + dx * dir;
            let ny = y + dy * dir;

            while (isValid(nx, ny) && board[ny][nx] === opponentPlayer) {
                opponentCount++;
                nx += dx * dir;
                ny += dy * dir;
            }
        }

        if (aiCount >= 4) score += 10000;
        else if (aiCount === 3) score += 1000;
        else if (aiCount === 2) score += 100;
        else if (aiCount === 1) score += 10;

        if (opponentCount >= 4) score += 6000;
        else if (opponentCount === 3) score += 600;
        else if (opponentCount === 2) score += 60;
    }

    let nearPieces = 0;
    for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (isValid(nx, ny) && board[ny][nx] !== 0) {
                nearPieces++;
            }
        }
    }
    score += nearPieces * 5;

    const centerX = Math.floor(BOARD_SIZE / 2);
    const centerY = Math.floor(BOARD_SIZE / 2);
    const distToCenter = Math.abs(x - centerX) + Math.abs(y - centerY);
    score += Math.max(0, 20 - distToCenter);

    return score;
}

function findBestMovesLegacy(player = getAITurnPlayer()) {
    const moves = [];
    const opponentPlayer = player === 1 ? 2 : 1;

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            board[y][x] = player;
            if (checkWin(x, y, player)) {
                board[y][x] = 0;
                return [{ x, y, score: 100000 }];
            }
            board[y][x] = 0;
        }
    }

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            board[y][x] = opponentPlayer;
            if (checkWin(x, y, opponentPlayer)) {
                board[y][x] = 0;
                return [{ x, y, score: 90000 }];
            }
            board[y][x] = 0;
        }
    }

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            const score = evaluatePositionLegacy(x, y, player);
            if (score > 0) {
                moves.push({ x, y, score });
            }
        }
    }

    moves.sort((a, b) => b.score - a.score);
    return moves;
}

function loadBenchmarkBoard(moves) {
    board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
    moveHistory = [];
    lastMove = null;
    totalMoveCount = 0;

    for (const move of moves) {
        board[move.y][move.x] = move.player;
        moveHistory.push(move);
        lastMove = { x: move.x, y: move.y };
        totalMoveCount++;
    }

    currentPlayer = totalMoveCount === 0 ? 1 : (totalMoveCount === 1 ? 2 : (totalMoveCount % 2 === 1 ? 1 : 2));
    moveCount = totalMoveCount <= 1 ? 0 : totalMoveCount % 2 === 0 ? 1 : 0;
    gameOver = false;
    winner = null;
}

function runAIBenchmark() {
    const presets = [
        {
            name: '必防局面',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 8, y: 9, player: 2 },
                { x: 10, y: 9, player: 1 },
                { x: 8, y: 10, player: 2 },
                { x: 11, y: 9, player: 1 },
                { x: 8, y: 11, player: 2 },
                { x: 12, y: 9, player: 1 }
            ]
        },
        {
            name: '中盘扩张',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 8, y: 8, player: 2 },
                { x: 10, y: 9, player: 1 },
                { x: 9, y: 8, player: 2 },
                { x: 11, y: 10, player: 1 },
                { x: 10, y: 8, player: 2 },
                { x: 8, y: 10, player: 1 },
                { x: 7, y: 9, player: 2 }
            ]
        },
        {
            name: '双子联动',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 7, y: 8, player: 2 },
                { x: 10, y: 10, player: 1 },
                { x: 8, y: 8, player: 2 },
                { x: 11, y: 11, player: 1 },
                { x: 8, y: 9, player: 2 }
            ]
        },
        {
            name: '中央冲四',
            moves: [
                { x: 8, y: 9, player: 1 },
                { x: 8, y: 8, player: 2 },
                { x: 9, y: 9, player: 1 },
                { x: 9, y: 8, player: 2 },
                { x: 10, y: 9, player: 1 },
                { x: 10, y: 8, player: 2 },
                { x: 11, y: 9, player: 1 }
            ]
        },
        {
            name: '边路防守',
            moves: [
                { x: 3, y: 5, player: 1 },
                { x: 4, y: 5, player: 2 },
                { x: 3, y: 6, player: 1 },
                { x: 5, y: 5, player: 2 },
                { x: 4, y: 6, player: 1 },
                { x: 6, y: 5, player: 2 },
                { x: 5, y: 6, player: 1 }
            ]
        },
        {
            name: '斜线推进',
            moves: [
                { x: 6, y: 6, player: 1 },
                { x: 7, y: 6, player: 2 },
                { x: 7, y: 7, player: 1 },
                { x: 8, y: 6, player: 2 },
                { x: 8, y: 8, player: 1 },
                { x: 9, y: 6, player: 2 },
                { x: 9, y: 9, player: 1 }
            ]
        },
        {
            name: '双向扩张',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 8, y: 9, player: 2 },
                { x: 10, y: 9, player: 1 },
                { x: 8, y: 10, player: 2 },
                { x: 9, y: 10, player: 1 },
                { x: 10, y: 8, player: 2 },
                { x: 11, y: 9, player: 1 },
                { x: 7, y: 9, player: 2 }
            ]
        },
        {
            name: '角部活棋',
            moves: [
                { x: 2, y: 2, player: 1 },
                { x: 3, y: 2, player: 2 },
                { x: 2, y: 3, player: 1 },
                { x: 4, y: 2, player: 2 },
                { x: 3, y: 3, player: 1 },
                { x: 5, y: 2, player: 2 },
                { x: 4, y: 4, player: 1 }
            ]
        },
        {
            name: '中盘对攻',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 11, y: 9, player: 2 },
                { x: 10, y: 10, player: 1 },
                { x: 11, y: 10, player: 2 },
                { x: 8, y: 10, player: 1 },
                { x: 12, y: 10, player: 2 },
                { x: 9, y: 11, player: 1 },
                { x: 10, y: 9, player: 2 }
            ]
        },
        {
            name: '纵向威胁',
            moves: [
                { x: 10, y: 5, player: 1 },
                { x: 9, y: 5, player: 2 },
                { x: 10, y: 6, player: 1 },
                { x: 9, y: 6, player: 2 },
                { x: 10, y: 7, player: 1 },
                { x: 9, y: 7, player: 2 },
                { x: 10, y: 8, player: 1 }
            ]
        },
        {
            name: '边路对攻',
            moves: [
                { x: 14, y: 8, player: 1 },
                { x: 13, y: 8, player: 2 },
                { x: 15, y: 9, player: 1 },
                { x: 13, y: 9, player: 2 },
                { x: 14, y: 10, player: 1 },
                { x: 12, y: 9, player: 2 },
                { x: 16, y: 10, player: 1 }
            ]
        },
        {
            name: '中心开局压制',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 8, y: 9, player: 2 },
                { x: 10, y: 10, player: 1 },
                { x: 9, y: 8, player: 2 },
                { x: 11, y: 11, player: 1 }
            ]
        },
        {
            name: '角部防反',
            moves: [
                { x: 2, y: 4, player: 1 },
                { x: 3, y: 4, player: 2 },
                { x: 2, y: 5, player: 1 },
                { x: 4, y: 4, player: 2 },
                { x: 3, y: 5, player: 1 },
                { x: 5, y: 4, player: 2 }
            ]
        },
        {
            name: '边路延伸',
            moves: [
                { x: 15, y: 4, player: 1 },
                { x: 14, y: 4, player: 2 },
                { x: 15, y: 5, player: 1 },
                { x: 14, y: 5, player: 2 },
                { x: 15, y: 6, player: 1 },
                { x: 13, y: 5, player: 2 }
            ]
        },
        {
            name: '中腹封堵',
            moves: [
                { x: 7, y: 9, player: 1 },
                { x: 9, y: 7, player: 2 },
                { x: 8, y: 9, player: 1 },
                { x: 9, y: 8, player: 2 },
                { x: 9, y: 9, player: 1 },
                { x: 9, y: 10, player: 2 },
                { x: 10, y: 9, player: 1 }
            ]
        },
        {
            name: '斜线断点',
            moves: [
                { x: 5, y: 5, player: 1 },
                { x: 7, y: 7, player: 2 },
                { x: 6, y: 6, player: 1 },
                { x: 8, y: 8, player: 2 },
                { x: 8, y: 6, player: 1 },
                { x: 9, y: 9, player: 2 }
            ]
        },
        {
            name: '横向冲势',
            moves: [
                { x: 6, y: 12, player: 1 },
                { x: 6, y: 11, player: 2 },
                { x: 7, y: 12, player: 1 },
                { x: 7, y: 11, player: 2 },
                { x: 8, y: 12, player: 1 },
                { x: 8, y: 11, player: 2 },
                { x: 9, y: 12, player: 1 }
            ]
        },
        {
            name: '双边展开',
            moves: [
                { x: 4, y: 9, player: 1 },
                { x: 14, y: 9, player: 2 },
                { x: 5, y: 9, player: 1 },
                { x: 13, y: 9, player: 2 },
                { x: 6, y: 10, player: 1 },
                { x: 12, y: 8, player: 2 }
            ]
        },
        {
            name: '边角过渡',
            moves: [
                { x: 3, y: 14, player: 1 },
                { x: 4, y: 13, player: 2 },
                { x: 4, y: 14, player: 1 },
                { x: 5, y: 13, player: 2 },
                { x: 5, y: 15, player: 1 },
                { x: 6, y: 13, player: 2 }
            ]
        },
        {
            name: '中盘扇形扩张',
            moves: [
                { x: 9, y: 9, player: 1 },
                { x: 8, y: 8, player: 2 },
                { x: 10, y: 9, player: 1 },
                { x: 8, y: 10, player: 2 },
                { x: 10, y: 11, player: 1 },
                { x: 7, y: 9, player: 2 },
                { x: 11, y: 10, player: 1 }
            ]
        },
        {
            name: '防守转进攻',
            moves: [
                { x: 11, y: 6, player: 1 },
                { x: 9, y: 6, player: 2 },
                { x: 11, y: 7, player: 1 },
                { x: 9, y: 7, player: 2 },
                { x: 10, y: 8, player: 1 },
                { x: 9, y: 8, player: 2 },
                { x: 12, y: 8, player: 1 }
            ]
        }
    ];

    const results = presets.map(preset => {
        loadBenchmarkBoard(preset.moves);
        const legacyStart = Date.now();
        const legacy = findBestMovesLegacy()[0] || null;
        const legacyMs = Date.now() - legacyStart;

        loadBenchmarkBoard(preset.moves);
        const modernStart = Date.now();
        const modern = findBestTurnMoves()[0] || null;
        const modernMs = Date.now() - modernStart;

        return {
            name: preset.name,
            legacy,
            modern,
            legacyMs,
            modernMs,
            differentMove: !!legacy && !!modern && (legacy.x !== modern.x || legacy.y !== modern.y)
        };
    });

    const summary = {
        totalCases: results.length,
        differentMoveCases: results.filter(r => r.differentMove).length,
        avgLegacyMs: Number((results.reduce((sum, r) => sum + r.legacyMs, 0) / results.length).toFixed(2)),
        avgModernMs: Number((results.reduce((sum, r) => sum + r.modernMs, 0) / results.length).toFixed(2))
    };

    console.table(results.map(result => ({
        局面: result.name,
        旧版: result.legacy ? `${result.legacy.x},${result.legacy.y}` : '无',
        新版: result.modern ? `${result.modern.x},${result.modern.y}` : '无',
        旧版耗时ms: result.legacyMs,
        新版耗时ms: result.modernMs,
        是否不同: result.differentMove ? '是' : '否'
    })));

    console.log('Benchmark Summary:', summary);

    return { summary, results };
}

function findBestMoves(context = null) {
    const moves = [];
    const aiPlayer = getAITurnPlayer();
    const opponentPlayer = aiPlayer === 1 ? 2 : 1;
    const candidateMoves = getCandidateMoves(getCandidateLimit() - 4);

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            board[y][x] = aiPlayer;
            if (checkWin(x, y, aiPlayer)) {
                moves.push({ x, y, score: 100000 });
            }
            board[y][x] = 0;
        }
    }
    if (moves.length > 0) {
        lastAIDecision = '优先选择直接取胜点';
        return moves;
    }

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            board[y][x] = opponentPlayer;
            if (checkWin(x, y, opponentPlayer)) {
                moves.push({ x, y, score: 90000 });
            }
            board[y][x] = 0;
        }
    }
    if (moves.length > 0) {
        lastAIDecision = '优先封堵对手的直接胜点';
        return moves;
    }

    const scoredMoves = candidateMoves.map(({ x, y }) => {
        let score = evaluatePosition(x, y, context);
        if (context?.firstMove) {
            score += scoreMoveSynergy(x, y, context);
        }
        return { x, y, score };
    });

    if (aiDifficulty === 'master' || aiDifficulty === 'hard') {
        scoredMoves.sort((a, b) => b.score - a.score);
        const checkCount = aiDifficulty === 'master' ? 8 : 6;
        const penalty = aiDifficulty === 'master' ? 20000 : 15000;
        const weight = aiDifficulty === 'master' ? 0.15 : 0.12;
        
        for (let index = 0; index < Math.min(scoredMoves.length, checkCount); index++) {
            const move = scoredMoves[index];
            board[move.y][move.x] = aiPlayer;
            const responseThreat = evaluateOpponentBestResponse(3);
            board[move.y][move.x] = 0;

            if (responseThreat >= 100000) {
                move.score -= penalty;
            } else {
                move.score -= Math.floor(responseThreat * weight);
            }
        }
    }

    for (const move of scoredMoves) {
        if (move.score > 0) {
            moves.push(move);
        }
    }

    moves.sort((a, b) => b.score - a.score);

    if (moves.length > 0) {
        lastAIDecision = describeAIMove(moves[0], moves[0].score, context);
    }

    let bestMoves = moves;
    if (aiDifficulty === 'easy' && moves.length > 3) {
        bestMoves = [moves[Math.floor(Math.random() * 3)]];
    } else if (aiDifficulty === 'medium' && moves.length > 2) {
        bestMoves = [moves[Math.floor(Math.random() * 2)]];
    }

    if (bestMoves.length === 0) {
        const emptyCells = [];
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] === 0) {
                    emptyCells.push({ x, y, score: 0 });
                }
            }
        }
        if (emptyCells.length > 0) {
            lastAIDecision = '候选点不足，退化为随机合法落子';
            return [emptyCells[Math.floor(Math.random() * emptyCells.length)]];
        }
    }

    return bestMoves;
}

function makeMoveForSearch(x, y, player) {
    board[y][x] = player;
    return { x, y, player };
}

function undoMoveForSearch(x, y) {
    board[y][x] = 0;
}

function getSearchDepth() {
    if (aiDifficulty === 'master') return 10;
    if (aiDifficulty === 'hard') return 8;
    if (aiDifficulty === 'medium') return 5;
    return 3;
}

function getCandidateLimit() {
    if (aiDifficulty === 'master') return 40;
    if (aiDifficulty === 'hard') return 30;
    if (aiDifficulty === 'medium') return 20;
    return 15;
}

function getSearchMoveLimit() {
    if (aiDifficulty === 'master') return 16;
    if (aiDifficulty === 'hard') return 12;
    if (aiDifficulty === 'medium') return 8;
    return 6;
}

// Alpha-Beta 剪枝搜索核心算法
// 经典的博弈树搜索算法，通过Alpha和Beta两个边界值进行剪枝
// Alpha表示最大化玩家当前能保证的最低分，Beta表示最小化玩家能保证的最高分
// 当Beta <= Alpha时，这个分支不可能产生更好的结果，可以直接剪枝
function alphaBetaSearch(depth, alpha, beta, maximizingPlayer, player) {
    // 叶子节点：达到搜索深度，直接评估局面
    if (depth === 0) {
        return evaluatePositionForSearch(player);
    }

    // 生成候选落点（剪枝优化：只搜索有意义的位置）
    const candidateMoves = getCandidateMoves(getSearchMoveLimit());
    const opponentPlayer = player === 1 ? 2 : 1;

    // 最大化玩家回合（AI视角）
    if (maximizingPlayer) {
        let maxScore = -Infinity;
        // 排序优化：先搜索评估分高的落点，增加剪枝效率
        const sortedMoves = candidateMoves.map(move => ({
            ...move,
            score: evaluatePosition(move.x, move.y, { playerOverride: player })
        })).sort((a, b) => b.score - a.score);

        for (const move of sortedMoves.slice(0, getSearchMoveLimit())) {
            makeMoveForSearch(move.x, move.y, player);
            
            // 立即获胜检查
            if (checkWin(move.x, move.y, player)) {
                undoMoveForSearch(move.x, move.y);
                return 1000000;
            }

            // 递归搜索下一层
            const score = alphaBetaSearch(depth - 1, alpha, beta, false, opponentPlayer);
            undoMoveForSearch(move.x, move.y);

            // 更新最大值和Alpha边界
            maxScore = Math.max(maxScore, score);
            alpha = Math.max(alpha, score);
            
            // Alpha-Beta剪枝：如果当前分支不会有更好结果，直接剪枝
            if (beta <= alpha) {
                break;
            }
        }
        return maxScore;
    } else {
        // 最小化玩家回合（对手视角）
        let minScore = Infinity;
        const sortedMoves = candidateMoves.map(move => ({
            ...move,
            score: evaluatePosition(move.x, move.y, { playerOverride: opponentPlayer })
        })).sort((a, b) => b.score - a.score);

        for (const move of sortedMoves.slice(0, getSearchMoveLimit())) {
            makeMoveForSearch(move.x, move.y, opponentPlayer);
            
            if (checkWin(move.x, move.y, opponentPlayer)) {
                undoMoveForSearch(move.x, move.y);
                return -1000000;
            }

            const score = alphaBetaSearch(depth - 1, alpha, beta, true, player);
            undoMoveForSearch(move.x, move.y);

            // 更新最小值和Beta边界
            minScore = Math.min(minScore, score);
            beta = Math.min(beta, score);
            
            // Alpha-Beta剪枝
            if (beta <= alpha) {
                break;
            }
        }
        return minScore;
    }
}

function evaluatePositionForSearch(player) {
    let score = 0;
    const opponentPlayer = player === 1 ? 2 : 1;
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];

    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === player) {
                for (const [dx, dy] of directions) {
                    const pattern = evaluateLinePattern(x, y, player, dx, dy);
                    score += scorePattern(pattern, aiDifficulty, false);
                }
            } else if (board[y][x] === opponentPlayer) {
                for (const [dx, dy] of directions) {
                    const pattern = evaluateLinePattern(x, y, opponentPlayer, dx, dy);
                    score -= scorePattern(pattern, aiDifficulty, true);
                }
            }
        }
    }

    const centerX = Math.floor(BOARD_SIZE / 2);
    const centerY = Math.floor(BOARD_SIZE / 2);
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === player) {
                const distToCenter = Math.abs(x - centerX) + Math.abs(y - centerY);
                score += Math.max(0, 20 - distToCenter);
            }
        }
    }

    return score;
}

let aiWorker = null;

function initAIWorker() {
    if (!aiWorker) {
        try {
            aiWorker = new Worker('scripts/ai-worker.js');
            aiWorker.onmessage = function(e) {
                if (e.data.type === 'result' && aiWorkerCallback) {
                    aiWorkerCallback(e.data.data);
                    aiWorkerCallback = null;
                }
            };
            aiWorker.onerror = function(e) {
                console.error('Worker error:', e.message);
                if (aiWorkerCallback) {
                    aiWorkerCallback(null);
                    aiWorkerCallback = null;
                }
            };
        } catch (e) {
            console.warn('Web Worker not supported, falling back to synchronous search');
        }
    }
}

let aiWorkerCallback = null;

function findBestMovesWithSearchAsync(player, candidateMoves, searchDepth, callback) {
    initAIWorker();
    
    if (!aiWorker) {
        const result = performSearchSync(player, candidateMoves, searchDepth);
        callback(result);
        return;
    }
    
    const boardCopy = board.map(row => [...row]);
    
    aiWorkerCallback = callback;
    aiWorker.postMessage({
        type: 'search',
        data: {
            player,
            searchDepth,
            candidateMoves,
            board: boardCopy
        }
    });
}

function performSearchSync(player, candidateMoves, searchDepth) {
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (const move of candidateMoves) {
        board[move.y][move.x] = player;
        
        if (checkWin(move.x, move.y, player)) {
            undoMoveForSearch(move.x, move.y);
            return { bestMove: { x: move.x, y: move.y }, score: 1000000 };
        }
        
        board[move.y][move.x] = 0;
        
        let score = evaluatePosition(move.x, move.y, { playerOverride: player });
        
        if (searchDepth >= 2) {
            board[move.y][move.x] = player;
            const searchResult = iterativeDeepeningSearch(player, searchDepth - 1);
            undoMoveForSearch(move.x, move.y);
            
            if (searchResult.bestScore) {
                score += searchResult.bestScore * 0.1;
            }
        }
        
        if (score > bestScore) {
            bestScore = score;
            bestMove = { x: move.x, y: move.y };
        }
    }
    
    return { bestMove, score: bestScore };
}

function performDeepSearchAsync(candidateMoves, player, searchDepth, callback) {
    initAIWorker();
    
    if (!aiWorker) {
        const result = performSearchSync(player, candidateMoves, searchDepth);
        callback(result);
        return;
    }
    
    const boardCopy = board.map(row => [...row]);
    
    aiWorkerCallback = callback;
    aiWorker.postMessage({
        type: 'search',
        data: {
            player,
            searchDepth,
            candidateMoves,
            board: boardCopy
        }
    });
}

/**
 * 使用深度搜索找到最佳落点
 * @param {Object} context - 搜索上下文
 * @param {boolean} context.isFirstMove - 是否为第一个落子（双落子中的第一子）
 * @param {Object} context.firstMove - 如果是第二子，传入第一子的位置
 * @returns {Array} 排序后的候选落点数组
 */
function findBestMovesWithSearch(context = {}) {
    const player = getAITurnPlayer();
    const searchDepth = getSearchDepth();
    const maxTime = AI_THINK_DELAY[aiDifficulty] || 2000;
    const startTime = Date.now();
    
    // 简单难度不使用深度搜索
    if (aiDifficulty === 'easy' || searchDepth <= 1) {
        return findBestMoves(context);
    }
    
    console.log(`[深度搜索] 启动搜索 - 深度:${searchDepth}, 时间限制:${maxTime}ms`);
    
    // 获取候选落点
    const candidateMoves = getCandidateMoves(getCandidateLimit());
    
    // 对每个候选点进行深度评估
    const scoredMoves = [];
    let searchedCount = 0;
    
    for (const move of candidateMoves) {
        // 超时检查
        if (Date.now() - startTime > maxTime) {
            console.log(`[深度搜索] 超时，停止搜索，已评估 ${searchedCount}/${candidateMoves.length} 个落点`);
            break;
        }
        
        // 放置棋子并检查胜负
        board[move.y][move.x] = player;
        
        if (checkWin(move.x, move.y, player)) {
            board[move.y][move.x] = 0;
            // 找到必胜点，立即返回
            console.log(`[深度搜索] 找到必胜点: (${move.x}, ${move.y})`);
            return [{ ...move, score: 1000000 }];
        }
        
        board[move.y][move.x] = 0;
        
        // 计算基础评估分数
        let baseScore = evaluatePosition(move.x, move.y, context);
        
        // 如果是双落子的第二子，计算协同效应
        if (context.firstMove) {
            baseScore += scoreMoveSynergy(move.x, move.y, { firstMove: context.firstMove });
        }
        
        // 使用深度搜索评估该落点
        const searchResult = evaluateMoveWithSearch(move, player, searchDepth, startTime, maxTime);
        
        // 综合分数 = 基础评估 + 搜索评估
        const totalScore = baseScore * 0.3 + searchResult.score * 0.7;
        
        scoredMoves.push({
            ...move,
            score: totalScore,
            baseScore: baseScore,
            searchScore: searchResult.score,
            searchDepth: searchResult.depthReached
        });
        
        searchedCount++;
        
        // 进度反馈
        if (searchedCount % 5 === 0) {
            console.log(`[深度搜索] 进度: ${searchedCount}/${candidateMoves.length} (${Math.round(searchedCount/candidateMoves.length*100)}%)`);
        }
    }
    
    // 按分数排序
    scoredMoves.sort((a, b) => b.score - a.score);
    
    const stats = searchOptimizations.getStats();
    console.log(`[深度搜索] 完成 - 评估${scoredMoves.length}个落点, 搜索节点:${stats.nodesSearched}, 缓存命中:${stats.cacheHits} (${stats.hitRate}%)`);
    
    return scoredMoves;
}

let aiThinking = false;
let aiAbortSignal = null;

/**
 * 使用深度搜索评估单个落点
 * @param {Object} move - 落点坐标 {x, y}
 * @param {number} player - 玩家编号 (1或2)
 * @param {number} maxDepth - 最大搜索深度
 * @param {number} startTime - 搜索开始时间
 * @param {number} maxTime - 最大时间限制
 * @returns {Object} 评估结果 {score, depthReached}
 */
function evaluateMoveWithSearch(move, player, maxDepth, startTime, maxTime) {
    // 保存当前局面状态
    const originalValue = board[move.y][move.x];

    // 放置棋子
    board[move.y][move.x] = player;
    const opponentPlayer = player === 1 ? 2 : 1;

    let bestScore = -Infinity;
    let depthReached = 0;

    // 清除搜索优化器，为新的搜索做准备
    searchOptimizations.clear();

    // 迭代加深搜索
    for (let depth = 1; depth <= maxDepth; depth++) {
        // 超时检查
        if (Date.now() - startTime > maxTime * 0.8) {
            console.log(`[深度搜索] 深度${depth}时超时`);
            break;
        }

        // 执行 Alpha-Beta 搜索
        const result = alphaBetaForConnect6(
            depth,
            -Infinity,
            Infinity,
            true,  // 最大化玩家
            opponentPlayer,  // 对手回合
            player,
            startTime,
            maxTime
        );

        if (result !== null) {
            bestScore = result;
            depthReached = depth;
        } else {
            // 搜索被中断
            break;
        }
    }

    // 恢复棋盘状态
    board[move.y][move.x] = originalValue;

    return {
        score: bestScore === -Infinity ? 0 : bestScore,
        depthReached: depthReached
    };
}

// ========== 六子棋专用的 Alpha-Beta 搜索 ==========
// 答辩要点：这是核心算法！相比标准Alpha-Beta有以下优化：
// 1. 置换表(Transposition Table)：缓存相同局面，避免重复计算
// 2. 历史启发式(History Heuristic)：根据历史搜索记录对落点排序
// 3. 双落子规则处理：六子棋每回合下2子（第一手除外）
// 4. 超时处理：避免搜索时间过长
function alphaBetaForConnect6(depth, alpha, beta, isMaximizing, currentPlayer, aiPlayer, startTime, maxTime) {
    // 统计搜索节点数（用于性能分析）
    searchOptimizations.nodesSearched++;

    // ========== 置换表缓存检查 ==========
    // 答辩要点：如果这个局面之前搜索过且深度足够，直接返回缓存结果
    const hash = searchOptimizations.getBoardHash();
    const cached = searchOptimizations.lookupTable(hash, depth);
    if (cached) {
        if (cached.type === 'exact') {
            return cached.score;  // 精确值，直接返回
        } else if (cached.type === 'lower' && cached.score >= beta) {
            return cached.score;  // 下界 >= beta，可以剪枝
        } else if (cached.type === 'upper' && cached.score <= alpha) {
            return cached.score;  // 上界 <= alpha，可以剪枝
        }
    }

    // ========== 叶子节点处理 ==========
    // 达到搜索深度，使用评估函数打分
    if (depth === 0) {
        const score = evaluateBoardPosition(aiPlayer);
        searchOptimizations.storeTable(hash, depth, score, 'exact', null);
        return score;
    }

    // 超时保护
    if (Date.now() - startTime > maxTime) {
        return null;
    }

    const opponentPlayer = currentPlayer === 1 ? 2 : 1;
    const isAI = currentPlayer === aiPlayer;

    // ========== 候选落点生成与排序 ==========
    const candidates = getDeepSearchCandidates(aiPlayer, currentPlayer);
    if (candidates.length === 0) {
        const score = evaluateBoardPosition(aiPlayer);
        searchOptimizations.storeTable(hash, depth, score, 'exact', null);
        return score;
    }

    // 历史启发式排序：之前搜索中表现好的落点优先
    candidates.sort((a, b) => {
        const scoreA = searchOptimizations.getHistoryScore(a.x, a.y, currentPlayer);
        const scoreB = searchOptimizations.getHistoryScore(b.x, b.y, currentPlayer);
        return scoreB - scoreA;
    });

    // 限制搜索宽度（进一步优化性能）
    const searchLimit = getSearchMoveLimit();
    const movesToSearch = candidates.slice(0, searchLimit);

    // ========== 最大化玩家（AI视角） ==========
    if (isMaximizing) {
        let maxScore = -Infinity;
        let bestMove = null;
        let entryType = 'upper';

        for (const move of movesToSearch) {
            board[move.y][move.x] = currentPlayer;

            // 即时获胜检查
            if (checkWin(move.x, move.y, currentPlayer)) {
                board[move.y][move.x] = 0;
                const winScore = isAI ? 1000000 : -1000000;
                searchOptimizations.updateHistory(move.x, move.y, depth, currentPlayer);
                searchOptimizations.storeTable(hash, depth, winScore, 'exact', move);
                return winScore;
            }

            // 递归搜索
            const childScore = alphaBetaForConnect6(
                depth - 1, alpha, beta, false, opponentPlayer, aiPlayer, startTime, maxTime
            );

            board[move.y][move.x] = 0;

            if (childScore === null) return null;  // 超时

            // 更新最大值
            if (childScore > maxScore) {
                maxScore = childScore;
                bestMove = move;
                if (childScore > alpha) {
                    alpha = childScore;
                    entryType = 'exact';
                }
            }

            // ========== Alpha-Beta 剪枝 ==========
            if (beta <= alpha) {
                searchOptimizations.updateHistory(move.x, move.y, depth, currentPlayer);
                entryType = 'lower';
                break;
            }
        }

        searchOptimizations.storeTable(hash, depth, maxScore, entryType, bestMove);
        return maxScore;
    } else {
        // ========== 最小化玩家（对手视角） ==========
        let minScore = Infinity;
        let bestMove = null;
        let entryType = 'lower';

        for (const move of movesToSearch) {
            board[move.y][move.x] = currentPlayer;

            if (checkWin(move.x, move.y, currentPlayer)) {
                board[move.y][move.x] = 0;
                const winScore = isAI ? 1000000 : -1000000;
                searchOptimizations.updateHistory(move.x, move.y, depth, currentPlayer);
                searchOptimizations.storeTable(hash, depth, winScore, 'exact', move);
                return winScore;
            }

            const childScore = alphaBetaForConnect6(
                depth - 1, alpha, beta, true, opponentPlayer, aiPlayer, startTime, maxTime
            );

            board[move.y][move.x] = 0;

            if (childScore === null) return null;

            if (childScore < minScore) {
                minScore = childScore;
                bestMove = move;
                if (childScore < beta) {
                    beta = childScore;
                    entryType = 'exact';
                }
            }

            if (beta <= alpha) {
                searchOptimizations.updateHistory(move.x, move.y, depth, currentPlayer);
                entryType = 'upper';
                break;
            }
        }

        searchOptimizations.storeTable(hash, depth, minScore, entryType, bestMove);
        return minScore;
    }
}

/**
 * 获取深度搜索的候选落点
 * 考虑AI视角和对手视角
 */
function getDeepSearchCandidates(aiPlayer, perspectivePlayer) {
    const candidates = [];
    const searchDist = aiDifficulty === 'master' ? 5 : (aiDifficulty === 'hard' ? 4 : 3);
    
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            
            // 检查是否有邻近棋子
            let hasNeighbor = false;
            for (let dy = -searchDist; dy <= searchDist && !hasNeighbor; dy++) {
                for (let dx = -searchDist; dx <= searchDist && !hasNeighbor; dx++) {
                    const nx = x + dx;
                    const ny = y + dy;
                    if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE) {
                        if (board[ny][nx] !== 0) {
                            hasNeighbor = true;
                        }
                    }
                }
            }
            
            if (hasNeighbor) {
                // 计算基础分数
                const score = evaluatePositionForQuick(x, y, perspectivePlayer);
                candidates.push({ x, y, score });
            }
        }
    }
    
    // 排序并返回
    candidates.sort((a, b) => b.score - a.score);
    return candidates;
}

/**
 * 快速评估函数（用于搜索排序）
 */
function evaluatePositionForQuick(x, y, player) {
    const opponentPlayer = player === 1 ? 2 : 1;
    let score = 0;
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    
    board[y][x] = player;
    
    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, player, dx, dy);
        score += scorePattern(pattern, aiDifficulty, false);
    }
    
    board[y][x] = opponentPlayer;
    for (const [dx, dy] of directions) {
        const pattern = evaluateLinePattern(x, y, opponentPlayer, dx, dy);
        score -= scorePattern(pattern, aiDifficulty, true) * 0.8;
    }
    
    board[y][x] = 0;
    
    return score;
}

/**
 * 评估整个棋盘局面（用于深度搜索叶子节点）
 */
function evaluateBoardPosition(aiPlayer) {
    let score = 0;
    const opponentPlayer = aiPlayer === 1 ? 2 : 1;
    
    const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
    
    // 遍历整个棋盘
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === aiPlayer) {
                for (const [dx, dy] of directions) {
                    const pattern = evaluateLinePattern(x, y, aiPlayer, dx, dy);
                    score += scorePattern(pattern, aiDifficulty, false);
                }
            } else if (board[y][x] === opponentPlayer) {
                for (const [dx, dy] of directions) {
                    const pattern = evaluateLinePattern(x, y, opponentPlayer, dx, dy);
                    score -= scorePattern(pattern, aiDifficulty, true);
                }
            }
        }
    }
    
    // 中心距离加成
    const centerX = Math.floor(BOARD_SIZE / 2);
    const centerY = Math.floor(BOARD_SIZE / 2);
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === aiPlayer) {
                const distToCenter = Math.abs(x - centerX) + Math.abs(y - centerY);
                score += Math.max(0, 20 - distToCenter);
            }
        }
    }
    
    return score;
}

// ========== 深度搜索优化组件 ==========
// 答辩要点：这个类实现了两大核心优化技术
// 1. 置换表(Transposition Table)：缓存已搜索过的局面
// 2. 历史启发式(History Heuristic)：记录好的落子位置，用于排序
class SearchOptimizations {
    constructor() {
        this.transpositionTable = new Map();  // 置换表：局面哈希 -> 搜索结果
        this.historyHeuristic = {};          // 历史启发表：记录每个位置的历史表现
        this.maxTableSize = 100000;          // 置换表最大容量
        this.nodesSearched = 0;              // 统计搜索节点数
        this.cacheHits = 0;                  // 统计缓存命中次数
    }

    // ========== 置换表相关方法 ==========
    
    // 计算棋盘局面的哈希值（使用简单的多项式哈希）
    getBoardHash() {
        let hash = 0;
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                hash = (hash * 3 + board[y][x]) % 0xFFFFFFFF;
            }
        }
        return hash;
    }

    // 查找置换表
    // 答辩要点：只有当缓存的深度 >= 当前搜索深度时，才使用缓存结果
    lookupTable(hash, depth) {
        const entry = this.transpositionTable.get(hash);
        if (entry && entry.depth >= depth) {
            this.cacheHits++;
            return entry;
        }
        return null;
    }

    // 存储到置换表
    // 答辩要点：使用FIFO策略淘汰旧条目，避免内存溢出
    storeTable(hash, depth, score, type, bestMove) {
        if (this.transpositionTable.size >= this.maxTableSize) {
            this.transpositionTable.delete(this.transpositionTable.keys().next().value);
        }
        this.transpositionTable.set(hash, {
            depth,
            score,
            type, // 'exact'(精确值), 'lower'(下界), 'upper'(上界)
            bestMove,
            timestamp: Date.now()
        });
    }

    // ========== 历史启发式相关方法 ==========
    
    // 更新历史启发表：当某个位置导致剪枝时，增加其权重
    // 答辩要点：权重与搜索深度的平方成正比，深度越深的剪枝越重要
    updateHistory(x, y, depth, player) {
        const key = `${player}-${x}-${y}`;
        if (!this.historyHeuristic[key]) {
            this.historyHeuristic[key] = 0;
        }
        this.historyHeuristic[key] += depth * depth;
    }

    // 获取某个位置的历史启发分数
    getHistoryScore(x, y, player) {
        const key = `${player}-${x}-${y}`;
        return this.historyHeuristic[key] || 0;
    }

    // 清空所有缓存和统计
    clear() {
        this.transpositionTable.clear();
        this.historyHeuristic = {};
        this.nodesSearched = 0;
        this.cacheHits = 0;
    }

    // 获取统计信息（用于调试和性能分析）
    getStats() {
        return {
            tableSize: this.transpositionTable.size,
            cacheHits: this.cacheHits,
            nodesSearched: this.nodesSearched,
            hitRate: this.nodesSearched > 0 ? (this.cacheHits / this.nodesSearched * 100).toFixed(2) : 0
        };
    }
}

let searchOptimizations = new SearchOptimizations();

function iterativeDeepeningSearch(player, maxDepth) {
    searchOptimizations.clear();
    
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (let depth = 1; depth <= maxDepth; depth++) {
        const result = alphaBetaWithHistory(player, depth, -Infinity, Infinity, true);
        
        if (result.bestMove) {
            bestMove = result.bestMove;
            bestScore = result.score;
        }
        
        if (result.score >= 1000000) {
            break;
        }
        
        console.log(`深度 ${depth} 搜索完成，最佳得分: ${bestScore}`);
    }
    
    const stats = searchOptimizations.getStats();
    console.log(`搜索统计: 节点数 ${stats.nodesSearched}, 缓存命中 ${stats.cacheHits} (${stats.hitRate}%)`);
    
    return { bestMove, bestScore };
}

function alphaBetaWithHistory(player, depth, alpha, beta, maximizingPlayer) {
    searchOptimizations.nodesSearched++;
    
    const hash = searchOptimizations.getBoardHash();
    const cached = searchOptimizations.lookupTable(hash, depth);
    if (cached) {
        if (cached.type === 'exact') {
            return { score: cached.score, bestMove: cached.bestMove };
        } else if (cached.type === 'lower' && cached.score >= beta) {
            return { score: cached.score, bestMove: cached.bestMove };
        } else if (cached.type === 'upper' && cached.score <= alpha) {
            return { score: cached.score, bestMove: cached.bestMove };
        }
    }
    
    if (depth === 0) {
        const score = evaluatePositionForSearch(player);
        return { score, bestMove: null };
    }
    
    const candidateMoves = getCandidateMovesForSearch(player);
    if (candidateMoves.length === 0) {
        return { score: 0, bestMove: null };
    }
    
    const opponentPlayer = player === 1 ? 2 : 1;
    let bestMove = null;
    let bestScore = maximizingPlayer ? -Infinity : Infinity;
    
    for (const move of candidateMoves) {
        makeMoveForSearch(move.x, move.y, maximizingPlayer ? player : opponentPlayer);
        
        if (checkWin(move.x, move.y, maximizingPlayer ? player : opponentPlayer)) {
            const score = maximizingPlayer ? 1000000 : -1000000;
            undoMoveForSearch(move.x, move.y);
            
            if (maximizingPlayer) {
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
                alpha = Math.max(alpha, score);
            } else {
                if (score < bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
                beta = Math.min(beta, score);
            }
            
            searchOptimizations.storeTable(hash, depth, bestScore, 'exact', bestMove);
            return { score: bestScore, bestMove };
        }
        
        const result = alphaBetaWithHistory(player, depth - 1, alpha, beta, !maximizingPlayer);
        undoMoveForSearch(move.x, move.y);
        
        if (maximizingPlayer) {
            if (result.score > bestScore) {
                bestScore = result.score;
                bestMove = move;
            }
            alpha = Math.max(alpha, result.score);
        } else {
            if (result.score < bestScore) {
                bestScore = result.score;
                bestMove = move;
            }
            beta = Math.min(beta, result.score);
        }
        
        if (beta <= alpha) {
            searchOptimizations.updateHistory(move.x, move.y, depth, player);
            break;
        }
    }
    
    let entryType = 'exact';
    if (bestScore <= alpha) {
        entryType = 'upper';
    } else if (bestScore >= beta) {
        entryType = 'lower';
    }
    
    searchOptimizations.storeTable(hash, depth, bestScore, entryType, bestMove);
    
    return { score: bestScore, bestMove };
}

function getCandidateMovesForSearch(player) {
    const candidates = [];
    const opponentPlayer = player === 1 ? 2 : 1;
    
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] !== 0) continue;
            
            const hasNeighbor = hasNearbyPiece(x, y, 2);
            if (!hasNeighbor && totalMoveCount > 5) continue;
            
            const historyScore = searchOptimizations.getHistoryScore(x, y, player);
            const evalScore = evaluatePosition(x, y, { playerOverride: player });
            
            candidates.push({
                x, y,
                score: evalScore + historyScore * 0.1,
                historyScore
            });
        }
    }
    
    candidates.sort((a, b) => b.score - a.score);
    
    const limit = aiDifficulty === 'master' ? 15 : aiDifficulty === 'hard' ? 12 : 8;
    return candidates.slice(0, limit);
}

function nullMovePruning(player, depth, alpha, beta) {
    if (depth < 3) return false;
    
    makeMoveForSearch(-1, -1, player);
    const result = alphaBetaWithHistory(player, depth - 3, alpha, beta, false);
    undoMoveForSearch(-1, -1);
    
    return result.score >= beta;
}

function aiMove() {
    if (gameOver || !isAITurn() || isBrowsingHistory() || aiThinking) return;

    aiThinking = true;
    aiAbortSignal = { aborted: false };
    document.getElementById('aiBtn').disabled = true;
    showMessage('AI正在思考...');
    startTimer();
    showAIProgress(0);

    const thinkDelay = AI_THINK_DELAY[aiDifficulty] || 1000;
    const moveDelay = aiDifficulty === 'master' ? 1500 : 800;

    if (aiMode === 'rl') {
        setTimeout(() => {
            const rlMove = getRLMove();
            
            if (rlMove) {
                makeMove(rlMove.x, rlMove.y);
                showAIProgress(-1);
                lastAIDecision = '使用强化学习模型';
                
                if (!gameOver && isAITurn() && moveCount < 2) {
                    setTimeout(() => {
                        const rlMove2 = getRLMove();
                        if (rlMove2 && (rlMove2.x !== rlMove.x || rlMove2.y !== rlMove.y)) {
                            makeMove(rlMove2.x, rlMove2.y);
                        }
                        aiThinking = false;
                        updateActionButtons();
                    }, moveDelay);
                } else {
                    aiThinking = false;
                    updateActionButtons();
                }
            } else {
                aiThinking = false;
                showAIProgress(-1);
                updateActionButtons();
            }
        }, thinkDelay);
        return;
    }
    
    asyncFindBestTurnMoves((plannedMoves) => {
        aiThinking = false;
        
        if (gameOver || !isAITurn()) {
            updateActionButtons();
            showAIProgress(-1);
            return;
        }

        if (plannedMoves && plannedMoves.length > 0) {
            setTimeout(() => {
                makeMove(plannedMoves[0].x, plannedMoves[0].y);
                showAIProgress(-1);

                if (!gameOver && isAITurn() && moveCount < 2 && plannedMoves[1]) {
                    setTimeout(() => {
                        makeMove(plannedMoves[1].x, plannedMoves[1].y);
                        updateActionButtons();
                    }, moveDelay);
                } else {
                    updateActionButtons();
                }
            }, thinkDelay);
        } else {
            showAIProgress(-1);
            updateActionButtons();
        }
    }, aiAbortSignal);
}

function showAIProgress(progress) {
    const progressBar = document.getElementById('aiProgressBar');
    const progressContainer = document.getElementById('aiProgressContainer');
    
    if (progressBar && progressContainer) {
        if (progress < 0) {
            progressContainer.style.display = 'none';
        } else {
            progressContainer.style.display = 'block';
            progressBar.style.width = `${Math.min(progress, 100)}%`;
        }
    }
}

function asyncFindBestTurnMoves(callback, abortSignal) {
    const aiPlayer = getAITurnPlayer();
    const profile = getCurrentAIProfile();
    const stonesLeft = totalMoveCount === 0 ? 1 : (2 - moveCount);
    const useSearch = aiDifficulty === 'master' || aiDifficulty === 'hard' || (aiDifficulty === 'medium' && totalMoveCount >= 8);
    
    if (totalMoveCount < 6 && aiDifficulty !== 'easy') {
        const openingMove = getOpeningBookMove();
        if (openingMove) {
            lastAIDecision = '使用开局库策略';
            callback([{ x: openingMove.x, y: openingMove.y }]);
            return;
        }
    }

    let firstMoves;
    try {
        firstMoves = useSearch ? findBestMovesWithSearch() : findBestMoves();
    } catch (e) {
        firstMoves = [];
    }
    
    if (firstMoves.length === 0) {
        callback([]);
        return;
    }

    if (stonesLeft <= 1) {
        if (useSearch && aiDifficulty === 'master') {
            performDeepSearchAsync(firstMoves, aiPlayer, getSearchDepth(), (result) => {
                if (result && result.bestMove) {
                    callback([result.bestMove]);
                } else {
                    callback([firstMoves[0]]);
                }
            });
        } else {
            callback([firstMoves[0]]);
        }
        return;
    }

    let topFirstMoves;
    if (aiDifficulty === 'master') topFirstMoves = firstMoves.slice(0, 8);
    else if (aiDifficulty === 'hard') topFirstMoves = firstMoves.slice(0, 7);
    else if (aiDifficulty === 'medium') topFirstMoves = firstMoves.slice(0, 6);
    else topFirstMoves = firstMoves.slice(0, 5);

    let bestPlan = null;
    let bestScore = -Infinity;
    let processedCount = 0;
    const totalCount = topFirstMoves.length;

    function processNext(firstIndex) {
        if (abortSignal && abortSignal.aborted) {
            callback(null);
            return;
        }

        if (firstIndex >= topFirstMoves.length) {
            if (bestPlan) {
                callback(bestPlan);
            } else {
                callback([firstMoves[0]]);
            }
            return;
        }

        const firstMove = topFirstMoves[firstIndex];
        
        setTimeout(() => {
            if (abortSignal && abortSignal.aborted) {
                callback(null);
                return;
            }

            board[firstMove.y][firstMove.x] = aiPlayer;

            let secondMoves;
            try {
                secondMoves = useSearch 
                    ? findBestMovesWithSearch({ firstMove }).filter(move => move.x !== firstMove.x || move.y !== firstMove.y)
                    : findBestMoves({ firstMove }).filter(move => move.x !== firstMove.x || move.y !== firstMove.y);
            } catch (e) {
                secondMoves = [];
            }

            board[firstMove.y][firstMove.x] = 0;

            let topSecondMoves;
            if (aiDifficulty === 'master') topSecondMoves = secondMoves.slice(0, 8);
            else if (aiDifficulty === 'hard') topSecondMoves = secondMoves.slice(0, 7);
            else if (aiDifficulty === 'medium') topSecondMoves = secondMoves.slice(0, 6);
            else topSecondMoves = secondMoves.slice(0, 5);

            function processSecond(secondIndex) {
                if (abortSignal && abortSignal.aborted) {
                    board[firstMove.y][firstMove.x] = 0;
                    callback(null);
                    return;
                }

                if (secondIndex >= topSecondMoves.length) {
                    board[firstMove.y][firstMove.x] = 0;
                    processedCount++;
                    showAIProgress(Math.round((processedCount / totalCount) * 100));
                    processNext(firstIndex + 1);
                    return;
                }

                requestAnimationFrame(() => {
                    const secondMove = topSecondMoves[secondIndex];
                    let pairScore = scoreMovePair(firstMove, secondMove, aiPlayer);

                    if (aiDifficulty === 'master' || aiDifficulty === 'hard') {
                        board[secondMove.y][secondMove.x] = aiPlayer;
                        const responseRisk = evaluatePairResponseRisk(firstMove, secondMove, aiPlayer);
                        board[secondMove.y][secondMove.x] = 0;

                        const penaltyMultiplier = aiDifficulty === 'master' ? 1.5 : 1;
                        if (responseRisk >= 100000) {
                            pairScore -= profile.pairLossPenalty * penaltyMultiplier;
                        } else {
                            pairScore -= Math.floor(responseRisk * profile.pairRiskWeight * penaltyMultiplier);
                        }
                    } else if (aiDifficulty === 'medium') {
                        const responseRisk = evaluatePairResponseRisk(firstMove, secondMove, aiPlayer);
                        if (responseRisk >= 100000) {
                            pairScore -= profile.pairLossPenalty * 0.7;
                        }
                    }

                    if (pairScore > bestScore) {
                        bestScore = pairScore;
                        bestPlan = [firstMove, secondMove];
                    }

                    processSecond(secondIndex + 1);
                });
            }

            processSecond(0);
        });
    }

    processNext(0);
}

// ========== 对局控制 ==========
function initGame() {
    board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
    currentPlayer = 1;
    gameOver = false;
    winner = null;
    moveCount = 0;
    totalMoveCount = 0;
    turnNumber = 1;
    lastMove = null;
    moveHistory = [];
    viewedHistoryIndex = null;

    resetTimer();
    renderBoard();
    updateStatus();
    updateStats();
    updateHistory();
    updateActionButtons();
    updateHistoryModeBadge();
    updateAIInsight('等待本回合分析');
    updateCoachSuggestion('当前暂无建议。');
}

function restartGame() {
    initGame();

    if (isAITurn()) {
        setTimeout(() => aiMove(), 300);
    }

    showMessage('游戏已重置');
}

function setDifficulty(difficulty, button) {
    aiDifficulty = difficulty;
    updateModeButtons(button, ['difficultyEasyBtn', 'difficultyMediumBtn', 'difficultyHardBtn', 'difficultyMasterBtn']);
    const difficultyNames = {
        easy: '简单',
        medium: '中等',
        hard: '困难',
        master: '大师'
    };
    showMessage(`AI难度已设置为${difficultyNames[difficulty] || '未知'}`);
}

function setGameMode(mode, button) {
    gameMode = mode;
    updateModeButtons(button, ['gameModeAiBtn', 'gameModeHumanBtn']);

    const aiRelatedElements = [
        document.getElementById('humanFirstBtn')?.closest('.setting-row'),
        document.getElementById('aiBtn')
    ].filter(Boolean);

    if (mode === 'ai') {
        aiRelatedElements.forEach(el => {
            el.style.display = 'block';
        });
    } else {
        aiRelatedElements.forEach(el => {
            el.style.display = 'none';
        });
    }

    restartGame();
    showMessage(`游戏模式已设置为${mode === 'ai' ? '人机对战' : '人人对战'}`);
}

function setFirstPlayer(player) {
    firstPlayer = player;
    document.getElementById('humanFirstBtn').classList.toggle('active', player === 'human');
    document.getElementById('aiFirstBtn').classList.toggle('active', player === 'ai');
    restartGame();
    showMessage(`先后手已设置为${player === 'human' ? '人类先手' : 'AI先手'}`);
}

function setAIProfile(profile, button) {
    aiProfile = profile;
    updateModeButtons(button, ['profileBalancedBtn', 'profileSolidBtn', 'profileAggressiveBtn']);
    showMessage(`AI策略已切换为${profile === 'balanced' ? '均衡' : profile === 'solid' ? '稳健' : '进攻'}模式`);
}

function undoMove() {
    if (moveHistory.length === 0) return;

    stopTimer();
    moveHistory.pop();
    viewedHistoryIndex = null;
    rebuildStateFromHistory(moveHistory);

    renderBoard();
    updateStatus();
    updateStats();
    updateHistory();
    showMessage('已悔棋一步');
}

function jumpToMove(index) {
    stopTimer();
    viewedHistoryIndex = index;
    rebuildStateFromHistory(moveHistory, index);

    renderBoard();
    updateStatus();
    updateStats();
    updateHistory();
    showMessage(`已跳转到第${index + 1}步`);
}

function continueFromHistory() {
    if (!isBrowsingHistory()) return;

    stopTimer();
    moveHistory = moveHistory.slice(0, viewedHistoryIndex + 1);
    viewedHistoryIndex = null;
    rebuildStateFromHistory(moveHistory);

    renderBoard();
    updateStatus();
    updateStats();
    updateHistory();
    showMessage('已从当前历史节点继续对局');
}

function returnToLatest() {
    if (!isBrowsingHistory()) return;

    stopTimer();
    viewedHistoryIndex = null;
    rebuildStateFromHistory(moveHistory);

    renderBoard();
    updateStatus();
    updateStats();
    updateHistory();
    showMessage('已返回最新局面');
}

function exportGame() {
    const exportText = buildExportText();
    const blob = new Blob([exportText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = buildExportFilename();
    a.click();

    URL.revokeObjectURL(url);
    showMessage('棋谱已按规范导出');
}

function importGame() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt';

    input.onchange = function (e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (event) {
            try {
                const gameData = parseImportedText(event.target.result);

                initGame();
                moveHistory = gameData.moveHistory;
                viewedHistoryIndex = null;
                rebuildStateFromHistory(moveHistory);
                document.getElementById('blackTeamInput').value = gameData.blackTeam;
                document.getElementById('whiteTeamInput').value = gameData.whiteTeam;

                renderBoard();
                updateStatus();
                updateStats();
                updateHistory();
                showMessage('规范棋谱已导入');
            } catch (error) {
                showMessage('导入失败: ' + error.message);
            }
        };
        reader.readAsText(file);
    };

    input.click();
}

function toggleNotation() {
    notationEnabled = !notationEnabled;
    renderBoard();
    showMessage(notationEnabled ? '棋子标记已启用' : '棋子标记已关闭');
}

// ========== 事件绑定 ==========
function bindEvents() {
    document.getElementById('difficultyEasyBtn')?.addEventListener('click', function () {
        setDifficulty('easy', this);
    });
    document.getElementById('difficultyMediumBtn')?.addEventListener('click', function () {
        setDifficulty('medium', this);
    });
    document.getElementById('difficultyHardBtn')?.addEventListener('click', function () {
        setDifficulty('hard', this);
    });
    document.getElementById('difficultyMasterBtn')?.addEventListener('click', function () {
        setDifficulty('master', this);
    });

    document.getElementById('gameModeAiBtn')?.addEventListener('click', function () {
        setGameMode('ai', this);
    });
    document.getElementById('gameModeHumanBtn')?.addEventListener('click', function () {
        setGameMode('human', this);
    });

    document.getElementById('humanFirstBtn')?.addEventListener('click', function () {
        setFirstPlayer('human');
    });
    document.getElementById('aiFirstBtn')?.addEventListener('click', function () {
        setFirstPlayer('ai');
    });

    document.getElementById('profileBalancedBtn')?.addEventListener('click', function () {
        setAIProfile('balanced', this);
    });
    document.getElementById('profileSolidBtn')?.addEventListener('click', function () {
        setAIProfile('solid', this);
    });
    document.getElementById('profileAggressiveBtn')?.addEventListener('click', function () {
        setAIProfile('aggressive', this);
    });

    document.getElementById('aiBtn')?.addEventListener('click', aiMove);
    document.getElementById('restartBtn')?.addEventListener('click', restartGame);
    document.getElementById('exportBtn')?.addEventListener('click', exportGame);
    document.getElementById('importBtn')?.addEventListener('click', importGame);
    document.getElementById('notationBtn')?.addEventListener('click', toggleNotation);
    document.getElementById('continueBtn')?.addEventListener('click', continueFromHistory);
    document.getElementById('returnLatestBtn')?.addEventListener('click', returnToLatest);
    document.getElementById('undoBtn')?.addEventListener('click', undoMove);

    document.addEventListener('keydown', function (event) {
        if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'k') {
            event.preventDefault();
            toggleCoachPanel();
        }
    });
}

// ========== 入口 ==========
bindEvents();
initGame();

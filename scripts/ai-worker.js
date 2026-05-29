const BOARD_SIZE = 19;
let board = [];

self.onmessage = function(e) {
    const { type, data } = e.data;
    
    if (type === 'init') {
        board = data.board;
    } else if (type === 'search') {
        const result = performSearch(data.player, data.searchDepth, data.candidateMoves);
        self.postMessage({ type: 'result', data: result });
    }
};

function performSearch(player, searchDepth, candidateMoves) {
    const transpositionTable = new Map();
    let nodesSearched = 0;
    let cacheHits = 0;
    
    function getBoardHash() {
        let hash = 0;
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                hash = (hash * 3 + board[y][x]) % 0xFFFFFFFF;
            }
        }
        return hash;
    }
    
    function alphaBeta(depth, alpha, beta, maximizingPlayer, currentPlayer) {
        nodesSearched++;
        
        const hash = getBoardHash();
        const cached = transpositionTable.get(hash);
        if (cached && cached.depth >= depth) {
            cacheHits++;
            if (cached.type === 'exact') return cached.score;
            if (cached.type === 'lower' && cached.score >= beta) return cached.score;
            if (cached.type === 'upper' && cached.score <= alpha) return cached.score;
        }
        
        if (depth === 0) {
            return evaluatePositionForSearch(player);
        }
        
        const moves = getSearchMoves(maximizingPlayer ? currentPlayer : (currentPlayer === 1 ? 2 : 1));
        if (moves.length === 0) return 0;
        
        let bestScore = maximizingPlayer ? -Infinity : Infinity;
        const opponent = currentPlayer === 1 ? 2 : 1;
        
        for (const move of moves) {
            board[move.y][move.x] = maximizingPlayer ? currentPlayer : opponent;
            
            if (checkWin(move.x, move.y, maximizingPlayer ? currentPlayer : opponent)) {
                board[move.y][move.x] = 0;
                const score = maximizingPlayer ? 1000000 : -1000000;
                
                if (transpositionTable.size > 50000) {
                    transpositionTable.delete(transpositionTable.keys().next().value);
                }
                transpositionTable.set(hash, { depth, score, type: 'exact' });
                return score;
            }
            
            const score = alphaBeta(depth - 1, alpha, beta, !maximizingPlayer, currentPlayer);
            board[move.y][move.x] = 0;
            
            if (maximizingPlayer) {
                bestScore = Math.max(bestScore, score);
                alpha = Math.max(alpha, score);
            } else {
                bestScore = Math.min(bestScore, score);
                beta = Math.min(beta, score);
            }
            
            if (beta <= alpha) break;
        }
        
        let entryType = 'exact';
        if (bestScore <= alpha) entryType = 'upper';
        else if (bestScore >= beta) entryType = 'lower';
        
        if (transpositionTable.size > 50000) {
            transpositionTable.delete(transpositionTable.keys().next().value);
        }
        transpositionTable.set(hash, { depth, score: bestScore, type: entryType });
        
        return bestScore;
    }
    
    function getSearchMoves(currentPlayer) {
        const candidates = [];
        
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] !== 0) continue;
                
                let hasNeighbor = false;
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] !== 0) {
                            hasNeighbor = true;
                            break;
                        }
                    }
                    if (hasNeighbor) break;
                }
                
                if (hasNeighbor) {
                    candidates.push({ x, y, score: evaluatePositionForSearchMove(x, y, currentPlayer) });
                }
            }
        }
        
        candidates.sort((a, b) => b.score - a.score);
        return candidates.slice(0, 12);
    }
    
    function evaluatePositionForSearchMove(x, y, currentPlayer) {
        const opponent = currentPlayer === 1 ? 2 : 1;
        let score = 0;
        const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
        
        for (const [dx, dy] of directions) {
            let aiCount = 0, oppCount = 0;
            let aiOpen = 0, oppOpen = 0;
            
            for (let dir = -1; dir <= 1; dir += 2) {
                let nx = x + dx * dir;
                let ny = y + dy * dir;
                while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === currentPlayer) {
                    aiCount++;
                    nx += dx * dir;
                    ny += dy * dir;
                }
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === 0) {
                    aiOpen++;
                }
            }
            
            for (let dir = -1; dir <= 1; dir += 2) {
                let nx = x + dx * dir;
                let ny = y + dy * dir;
                while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === opponent) {
                    oppCount++;
                    nx += dx * dir;
                    ny += dy * dir;
                }
                if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === 0) {
                    oppOpen++;
                }
            }
            
            if (aiCount >= 5) score += 500000;
            else if (aiCount === 4 && aiOpen >= 2) score += 200000;
            else if (aiCount === 4 && aiOpen === 1) score += 80000;
            else if (aiCount === 3 && aiOpen >= 2) score += 30000;
            else if (aiCount === 3 && aiOpen === 1) score += 8000;
            
            if (oppCount >= 5) score += 450000;
            else if (oppCount === 4 && oppOpen >= 2) score += 180000;
            else if (oppCount === 4 && oppOpen === 1) score += 70000;
            else if (oppCount === 3 && oppOpen >= 2) score += 25000;
        }
        
        return score;
    }
    
    function evaluatePositionForSearch(currentPlayer) {
        const opponent = currentPlayer === 1 ? 2 : 1;
        let score = 0;
        const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
        
        for (const [dx, dy] of directions) {
            for (let y = 0; y < BOARD_SIZE; y++) {
                for (let x = 0; x < BOARD_SIZE; x++) {
                    if (board[y][x] !== currentPlayer) continue;
                    
                    let count = 1;
                    let openEnds = 0;
                    
                    let nx = x + dx;
                    let ny = y + dy;
                    while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === currentPlayer) {
                        count++;
                        nx += dx;
                        ny += dy;
                    }
                    if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === 0) {
                        openEnds++;
                    }
                    
                    nx = x - dx;
                    ny = y - dy;
                    while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === currentPlayer) {
                        count++;
                        nx -= dx;
                        ny -= dy;
                    }
                    if (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === 0) {
                        openEnds++;
                    }
                    
                    if (count >= 6) score += 5000000;
                    else if (count === 5 && openEnds >= 2) score += 800000;
                    else if (count === 5 && openEnds === 1) score += 400000;
                    else if (count === 4 && openEnds >= 2) score += 200000;
                    else if (count === 4 && openEnds === 1) score += 80000;
                    else if (count === 3 && openEnds >= 2) score += 30000;
                }
            }
        }
        
        return score;
    }
    
    function checkWin(x, y, player) {
        const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
        
        for (const [dx, dy] of directions) {
            let count = 1;
            
            let nx = x + dx;
            let ny = y + dy;
            while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === player) {
                count++;
                nx += dx;
                ny += dy;
            }
            
            nx = x - dx;
            ny = y - dy;
            while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE && board[ny][nx] === player) {
                count++;
                nx -= dx;
                ny -= dy;
            }
            
            if (count >= 6) return true;
        }
        
        return false;
    }
    
    let bestMove = null;
    let bestScore = -Infinity;
    
    for (let depth = 1; depth <= searchDepth; depth++) {
        for (const move of candidateMoves) {
            board[move.y][move.x] = player;
            
            if (checkWin(move.x, move.y, player)) {
                board[move.y][move.x] = 0;
                return { bestMove: { x: move.x, y: move.y }, score: 1000000, depth, nodesSearched, cacheHits };
            }
            
            const score = alphaBeta(depth - 1, -Infinity, Infinity, false, player);
            board[move.y][move.x] = 0;
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = { x: move.x, y: move.y };
            }
        }
        
        if (bestScore >= 1000000) break;
    }
    
    return { bestMove, score: bestScore, depth: searchDepth, nodesSearched, cacheHits };
}
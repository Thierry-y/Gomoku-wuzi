from alpha_beta import evaluation  
from val_net_ai import Model_new
from ai import Model  

class searcher(object):
    def __init__(self, ann_model_path='go_model_alphazero.pth'):
        # Initialize evaluation functions
        self.evaluator = evaluation()
        self.ann_evaluator = Model_new(ann_model_path)
        
        # Initialize an empty 19x19 Go board
        self.board = [[0 for _ in range(19)] for _ in range(19)]
        
        # Game state variables
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3  # Maximum search depth
        self.neighbor_radius = 1  # Radius for checking nearby moves
        self.center_pos = [(9, 9)]  # Default first move position

    def hasNeighbor(self, x, y, radius):
        # Check if there are any stones within the given radius
        start_x, end_x = max(0, x-radius), min(18, x+radius)
        start_y, end_y = max(0, y-radius), min(18, y+radius)
        
        for i in range(start_y, end_y+1):
            for j in range(start_x, end_x+1):
                if i == y and j == x:  # Ignore the current position
                    continue
                if self.board[i][j] != 0:
                    return True  # Found a neighbor stone
        return False

    def genmove(self, turn):
        # Generate possible moves for the current turn
        board = self.board
        POSES = self.evaluator.POS  # Position evaluation values
        
        moves = []
        for i in range(19):
            for j in range(19):
                if board[i][j] == 0:  # Empty position
                    if self.hasNeighbor(j, i, self.neighbor_radius):  # Check for nearby stones
                        score = POSES[i][j]  # Get evaluation score
                        moves.append((score, i, j))
        
        # If no moves are found, prioritize the center position if available
        if not moves and self.board[9][9] == 0:
            moves.append((POSES[9][9], 9, 9))
        
        # Sort moves by evaluation score in descending order
        moves.sort(reverse=True, key=lambda x: x[0])
        return moves
    
    # Recursive search: returns the best score for the current turn
    def __search(self, turn, depth, alpha=-0x7fffffff, beta=0x7fffffff):
        if depth <= 0:
            # Use a mix of ANN-based and traditional evaluation functions
            mix_score = self.ann_evaluator.get_score_ANN(self.board, turn) * 0.5 + self.evaluator.evaluate(self.board, turn) * 0.5
            return mix_score
        
        # if depth <= 0:
        #     # Use a ANN-based evaluation functions
        #     mix_score = self.ann_evaluator.get_score_ANN(self.board, turn)
        #     return mix_score

        # if depth <= 0:
        #     score = self.evaluator.evaluate(self.board, turn)
        #     return score
        
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score  # Early termination if the position is a win/loss
        
        moves = self.genmove(turn)
        bestmove = None
        
        for score_move, row, col in moves:
            self.board[row][col] = turn  # Make move
            nturn = 2 if turn == 1 else 1  # Switch turn
            score_eval = -self.__search(nturn, depth - 1, -beta, -alpha)  # Minimax search
            self.board[row][col] = 0  # Undo move
            
            if score_eval > alpha:
                alpha = score_eval
                bestmove = (row, col)
                if alpha >= beta:
                    break  # Beta cut-off (pruning)
        
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove  # Store best move at max depth
        return alpha

    # Public interface: search for the best move and return its score and coordinates
    def search(self, turn, depth=3):
        self.maxdepth = depth
        self.bestmove = None
        
        score = self.__search(turn, depth)
        
        # If a winning move is found, refine the search with depth=1 for immediate moves
        if abs(score) > 8000:
            self.maxdepth = depth
            score = self.__search(turn, 1)
        
        row, col = self.bestmove
        return score, row, col

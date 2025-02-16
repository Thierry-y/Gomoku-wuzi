from alpha_beta import evaluation

class searcher(object):
    def __init__(self):
        self.evaluator = evaluation()
        # Create an empty board (19x19)
        self.board = [[0 for _ in range(19)] for _ in range(19)]
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3

    # Generate all possible moves based on positional weights
    def genmove(self, turn):
        moves = []
        board = self.board
        POSES = self.evaluator.POS
        for i in range(19):
            for j in range(19):
                if board[i][j] == 0:
                    score = POSES[i][j]
                    moves.append((score, i, j))
        moves.sort()
        moves.reverse()
        return moves

    # Recursive search: returns the best score for the current turn
    def __search(self, turn, depth, alpha=-0x7fffffff, beta=0x7fffffff):
        if depth <= 0:
            score = self.evaluator.evaluate(self.board, turn)
            return score
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score
        moves = self.genmove(turn)
        bestmove = None
        for score_move, row, col in moves:
            self.board[row][col] = turn
            nturn = 2 if turn == 1 else 1
            score_eval = - self.__search(nturn, depth - 1, -beta, -alpha)
            self.board[row][col] = 0
            if score_eval > alpha:
                alpha = score_eval
                bestmove = (row, col)
                if alpha >= beta:
                    break
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove
        return alpha

    # Public interface: search for the best move and return its score and coordinates
    def search(self, turn, depth=3):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(turn, depth)
        if abs(score) > 8000:
            self.maxdepth = depth
            score = self.__search(turn, 1)
        row, col = self.bestmove
        return score, row, col
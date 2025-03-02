import pygame
import torch
from board import Board
from mcts import AlphaZeroMCTS
from pol_val_net import PolicyValueNet


# Initialize Pygame
pygame.init()

# Set the screen size
screen_size = 600
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Gomoku Game')

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)

# Board size
board_len = 9
cell_size = screen_size // (board_len + 1)  

# Load the model
model_path = 'model/final_policy_value_net_2700.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_value_net = torch.load(model_path, map_location=device)
policy_value_net.eval()
mcts = AlphaZeroMCTS(policy_value_net, c_puct=4, n_iters=500, is_self_play=False) # Reduce n_iters to speed up

# Create the board
board = Board(board_len, n_feature_planes=6)

def draw_board():
    screen.fill(BROWN)
    for i in range(board_len):
        pygame.draw.line(screen, BLACK, ((i + 1) * cell_size, cell_size), ((i + 1) * cell_size, screen_size - cell_size))
        pygame.draw.line(screen, BLACK, (cell_size, (i + 1) * cell_size), (screen_size - cell_size, (i + 1) * cell_size))

def draw_stones():
    for action, player in board.state.items():
        row, col = divmod(action, board_len)
        x = (col + 1) * cell_size
        y = (row + 1) * cell_size
        radius = cell_size // 2 - 2  
        if player == Board.BLACK:
            pygame.draw.circle(screen, BLACK, (x, y), radius)
        elif player == Board.WHITE:
            pygame.draw.circle(screen, WHITE, (x, y), radius)
            pygame.draw.circle(screen, BLACK, (x, y), radius, 1)

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                col = round(x / cell_size) - 1  
                row = round(y / cell_size) - 1
                if 0 <= row < board_len and 0 <= col < board_len:
                    action = row * board_len + col
                    if action in board.available_actions:
                        board.do_action(action)
                        if board.is_game_over()[0]:
                            print("Player (Black) wins!")
                            running = False
                        else:
                            if mcts.is_self_play:
                                action, _ = mcts.get_action(board)
                            else:
                                action = mcts.get_action(board)
                            board.do_action(action)
                            if board.is_game_over()[0]:
                                print("AI (White) wins!")
                                running = False

        draw_board()
        draw_stones()
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()

import pygame
import sys

# Constants
GRID_SIZE = 30  # Grid size
BOARD_SIZE = 15  # Board size
SCREEN_SIZE = GRID_SIZE * BOARD_SIZE  # Window size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)  # Background color
LINE_COLOR = (0, 0, 0)
WHITE_CHESS = (255, 255, 255)
BLACK_CHESS = (0, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Gomoku")

# Game menu screen
def game_menu():
    screen.fill(BROWN)
    font = pygame.font.Font(None, 40)
    title_text = font.render("Gomoku Game", True, WHITE)
    player_vs_player_text = font.render("Player vs Player", True, WHITE)
    player_vs_ai_text = font.render("Player vs AI (Coming Soon)", True, WHITE)
    
    pvp_rect = player_vs_player_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2 - 30))
    ai_rect = player_vs_ai_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 30))
    
    screen.blit(title_text, (SCREEN_SIZE // 3, SCREEN_SIZE // 4))
    screen.blit(player_vs_player_text, pvp_rect)
    screen.blit(player_vs_ai_text, ai_rect)
    
    pygame.display.flip()
    
    selecting = True
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if pvp_rect.collidepoint(x, y):
                    selecting = False  # Enter player vs player mode

# Draw game board
def draw_board():
    screen.fill(BROWN)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (i * GRID_SIZE, 0), (i * GRID_SIZE, SCREEN_SIZE))
        pygame.draw.line(screen, LINE_COLOR, (0, i * GRID_SIZE), (SCREEN_SIZE, i * GRID_SIZE))

game_menu()
draw_board()

# Track board state, 0 = empty, 1 = black piece, 2 = white piece
board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
turn = 1  # 1 = black, 2 = white

# Check win condition
def check_win(x, y, player):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for d in (-1, 1):
            step = 1
            while True:
                nx, ny = x + step * dx * d, y + step * dy * d
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
                    count += 1
                    step += 1
                else:
                    break
        if count >= 5:
            return True
    return False

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            row, col = round(x / GRID_SIZE), round(y / GRID_SIZE)
            if board[row][col] == 0:
                board[row][col] = turn
                color = BLACK_CHESS if turn == 1 else WHITE_CHESS
                pygame.draw.circle(screen, color, (row * GRID_SIZE, col * GRID_SIZE), GRID_SIZE // 2 - 2)
                if check_win(row, col, turn):
                    print(f"Player {turn} wins!")
                    running = False
                turn = 3 - turn  # Switch turn
    pygame.display.flip()

pygame.quit()
sys.exit()

# main

import pygame
import sys
import time
from search import searcher

# Constants
GRID_SIZE = 30      # Size of each grid cell
BOARD_SIZE = 19     # Number of rows/columns on the board
SCREEN_SIZE = GRID_SIZE * BOARD_SIZE  # Window size
BROWN = (139, 69, 19)     # Board background color
LINE_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Gomoku")

def game_menu():
    screen.fill(BROWN)
    font = pygame.font.Font(None, 40)
    title_text = font.render("Gomoku Game", True, WHITE_COLOR)
    pvp_text = font.render("Player vs Player", True, WHITE_COLOR)
    pvai_text = font.render("Player vs AI", True, WHITE_COLOR)
    
    pvp_rect = pvp_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2 - 30))
    pvai_rect = pvai_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 30))
    
    screen.blit(title_text, (SCREEN_SIZE // 3, SCREEN_SIZE // 4))
    screen.blit(pvp_text, pvp_rect)
    screen.blit(pvai_text, pvai_rect)
    
    pygame.display.flip()
    
    selecting = True
    mode = None
    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if pvp_rect.collidepoint(x, y):
                    mode = "pvp"
                    selecting = False
                elif pvai_rect.collidepoint(x, y):
                    mode = "pvai"
                    selecting = False
    return mode

def draw_board():
    screen.fill(BROWN)
    for i in range(BOARD_SIZE):
        # Draw vertical grid lines
        pygame.draw.line(screen, LINE_COLOR,
                         (i * GRID_SIZE + GRID_SIZE//2, GRID_SIZE//2),
                         (i * GRID_SIZE + GRID_SIZE//2, SCREEN_SIZE - GRID_SIZE//2), 1)
        # Draw horizontal grid lines
        pygame.draw.line(screen, LINE_COLOR,
                         (GRID_SIZE//2, i * GRID_SIZE + GRID_SIZE//2),
                         (SCREEN_SIZE - GRID_SIZE//2, i * GRID_SIZE + GRID_SIZE//2), 1)
    pygame.display.flip()

def check_win(row, col, player, board):
    # Check winning condition from the given position by scanning in four directions
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        # Extend in both directions
        for sign in (-1, 1):
            r, c = row, col
            while True:
                r += sign * dr
                c += sign * dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                    count += 1
                else:
                    break
        if count >= 5:
            return True
    return False

def main():
    mode = game_menu()  # Display menu and choose game mode
    draw_board()

    # Initialize board state
    if mode == "pvai":
        ai_engine = searcher()
        board = ai_engine.board  # Use the same board as in the AI engine
        human_player = 1
        ai_player = 2
    else:
        board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    turn = 1  # 1 for Black, 2 for White
    running = True
    while running:
        # In Player vs AI mode, when it's the AI's turn, let the AI move automatically
        if mode == "pvai" and turn == ai_player:
            pygame.time.delay(500)  # Delay to make the AI move observable
            score, ai_row, ai_col = ai_engine.search(turn, depth=2)
            if board[ai_row][ai_col] == 0:
                board[ai_row][ai_col] = turn
                pos = (ai_col * GRID_SIZE + GRID_SIZE//2, ai_row * GRID_SIZE + GRID_SIZE//2)
                pygame.draw.circle(screen, WHITE_COLOR, pos, GRID_SIZE//2 - 2)
                pygame.display.flip()
                if check_win(ai_row, ai_col, turn, board):
                    print("AI (White) wins!")
                    running = False
                turn = 3 - turn
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # In PvAI mode, only process clicks when it is the human's turn (Black)
                if mode == "pvai" and turn != human_player:
                    continue
                x, y = event.pos
                col = x // GRID_SIZE
                row = y // GRID_SIZE
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0:
                    board[row][col] = turn
                    color = BLACK_COLOR if turn == 1 else WHITE_COLOR
                    pos = (col * GRID_SIZE + GRID_SIZE//2, row * GRID_SIZE + GRID_SIZE//2)
                    pygame.draw.circle(screen, color, pos, GRID_SIZE//2 - 2)
                    pygame.display.flip()
                    if check_win(row, col, turn, board):
                        print(f"Player {turn} wins!")
                        running = False
                        break
                    turn = 3 - turn
        pygame.display.flip()

    time.sleep(2)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
# Proviamo a fare una partita

import numpy as np
from IPython.display import clear_output
import Forza4Methods as f4
import pandas as pd

ROW_COUNT = 6
COLUMN_COUNT = 7


def create_board():
    board = np.zeros((6, 7))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    # if this condition is true we will let the use drop piece here.
    # if not true that means the col is not vacant
    return board[5][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print('*** FORZA 4 BOT ***')
    print('\n')
    print(np.flip(board, 0))
    clear_output(wait=True)


board = create_board()
print_board(board)
game_over = False
turn = 0

while not game_over:
    # Ask for player 1 input
    if turn == 0:
        col = int(input("Player, Make your Selection(0-6):"))
        # Player 1 will drop a piece on the board
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 2)

    # Ask for player 2 input
    else:
        # Qui facciamo giocare l'algoritmo

        col = f4.algoMove(f4.convertMatchtoSeries(board), model = 'KNN', return_column=True)

        # col = int(input("Player 2, Make your Selection(0-6):"))
        # Player 2 will drop a piece on the board
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 1)

    print_board(board)

    turn += 1
    turn = turn % 2


# Definisci una giocata qualsiasi

import time
import pandas as pd
import Forza4Methods as f4
import random
import numpy as np

baseData = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\10kGamesSimulator.xlsx")

ind = 34  # 630 ha una vittoria obliqua destra e una sinistra

winInd = baseData['Win'][ind]
if winInd == 1:
    print('Win: Yes')
if winInd == 0:
    print('Win: No')

print('\n')

game = baseData.loc[:, baseData.columns != 'Win'].iloc[ind]

sGame = f4.convertToVisualGame(game)

print(sGame)

#hardCodeHor = f4.hardCodeHorizontalWins(game)
#hardCodeVert = f4.hardCodeVerticalWins(game)
#print(hardCodeVert)
#hardcodeDR = f4.hardCodeDiagonalRightWins(game)
#hardcodeDL = f4.hardCodeDiagonalLeftWins(game)

print('\n')

#if len(hardCodeHor) != 0:
#    print('BLOCCA la vittoria (ORIZZONTALE) avversaria in:', hardCodeHor, '!')
#if len(hardCodeVert) != 0:
#    print('BLOCCA la vittoria (VERTICALE) avversaria in:', hardCodeVert, '!')
#if len(hardcodeDR) != 0:
#    print('BLOCCA la vittoria (OBLIQUA DESTRA) avversaria in:', hardcodeDR, '!')
#if len(hardcodeDL) != 0:
#    print('BLOCCA la vittoria (OBLIQUA SINISTRA) avversaria in:', hardcodeDL, '!')

game = pd.DataFrame(game).set_axis([0], axis = 1)

#a = f4.fastHorizontalWinIdentification(game)
#b = f4.fastDiagonalRightWinIdentifier(game)
#c = f4.fastDiagonalLeftWinIdentifier(game)
#d = f4.fastVerticalWinIdentifier(game)
#
#print('Horizontal:', a)
#print('Diagonal Right:', b)
#print('Diagonal Left:', c)
#print('Vertical:', d)

# Identifichiamo l'ultima riga scritta (cioè il l'ultima lettera scritta)

letters = ['F', 'E', 'D', 'C', 'B', 'A']

gameNoZero = game

prLetter = list()
for letter in letters:
    gameNoZero1 = pd.concat([gameNoZero, pd.DataFrame(gameNoZero.index.str.contains(letter)).set_index(gameNoZero.index)],
                           axis = 1).set_axis([0, 1], axis = 1)
    gameNoZero1 = gameNoZero1[0][gameNoZero1[1] == True]
    prLetter.append(gameNoZero1)

preWin = list()
for series in prLetter:
    dist = series[series != 0]
    if len(dist) > 3 and len(dist) < 7:
        preWin.append(series)

# Get last value of Index (=last open cell)

openCols = list()
for series1 in preWin:
    openCols.append(series1.index[0][0])

finalDec = random.choice()





















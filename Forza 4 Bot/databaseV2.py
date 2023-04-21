# Database diverso per discriminare meglio le vittorie delle 'X'
# In questo caso il modello che scegliamo è lo stesso ma inseriamo solo le vittorie "secche" di 1 (quindi escludiamo
# le giocate che hanno una vittoria dei 2.

import pandas as pd
import Forza4Methods as f4
import numpy as np

base = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSet.xlsx")[0:100]
compare = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSetV2.xlsx")[0:100]

oldWins = base['Win']

Hor = list()
Ver = list()
DiagR = list()
DiagL = list()

for iteration, column in enumerate(base.transpose().columns):
    eachGame = base.iloc[column]

    Hor.append(f4.fastHorizontalWinIdentification(eachGame, default=2))
    Ver.append(f4.fastVerticalWinIdentifier(eachGame, default=2))
    DiagR.append(f4.fastDiagonalRightWinIdentifier(eachGame, default=2))
    DiagL.append(f4.fastDiagonalLeftWinIdentifier(eachGame, default=2))

    # Stato di Avanzamento
    print('\n')
    print('Game', iteration, 'analyzed, out of', len(base['A1']))
    print('Progress', round(iteration / len(base), 2) * 100, '%')

wins = pd.concat([pd.Series(Hor), pd.Series(Ver), pd.Series(DiagR), pd.Series(DiagL)], axis=1).set_axis(
    ['Horizontal Win',
     'Vertical Win', 'Right Diagonal Win', 'Left Diagonal Win'], axis=1)

dfWithWins = pd.concat([base, oldWins, wins], axis=1)

print('\n')
print('Total Games:', len(dfWithWins['Vertical Win']))
print('Vertical Wins:', len(dfWithWins['Vertical Win'][dfWithWins['Vertical Win'] == 1]))
print('Horizontal Wins:', len(dfWithWins['Horizontal Win'][dfWithWins['Horizontal Win'] == 1]))
print('Diagonal Wins (Right):', len(dfWithWins['Right Diagonal Win'][dfWithWins['Right Diagonal Win'] == 1]))
print('Diagonal Wins (Left):', len(dfWithWins['Left Diagonal Win'][dfWithWins['Left Diagonal Win'] == 1]))

dfWithWins.loc[(dfWithWins['Horizontal Win'] == 1) | (dfWithWins['Vertical Win'] == 1) |
                   (dfWithWins['Right Diagonal Win'] == 1) | (dfWithWins['Left Diagonal Win'] == 1), 'Win2'] = 1
dfWithWins['Win2'] = dfWithWins['Win2'].fillna(0).astype(int)

dfWithWins.loc[(dfWithWins['Win2'] == 1), 'Win'] = 0

del [dfWithWins['Horizontal Win']]
del [dfWithWins['Vertical Win']]
del [dfWithWins['Right Diagonal Win']]
del [dfWithWins['Left Diagonal Win']]

dfWithWins.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSetV3.xlsx")

print('\n')
print('--UPDATED DATABASE--')
print('\n')

print('TOTAL GAMES:', len(dfWithWins['Win']))
print('TOTAL WINS:', len(dfWithWins['Win'][dfWithWins['Win'] == 1]))
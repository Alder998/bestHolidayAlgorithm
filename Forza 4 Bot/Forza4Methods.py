def diagonalWinsCombinations(initialCoord):
    import pandas as pd

    # Inizalizza le coordinate numeriche e alfabetiche

    char = ['A', 'B', 'C', 'D', 'E', 'F']
    num = ['1', '2', '3', '4', '5', '6', '7']

    # Inizializza le coordinate di partenza

    coord0 = str(initialCoord)

    baseCR = list()
    for character in coord0:
        baseCR.append(character)

    firstLetter = baseCR[0]
    firstNumber = int(baseCR[1])

    charEncodedA = pd.concat([pd.Series(['A', 'B', 'C', 'D', 'E', 'F']), pd.Series([1, 2, 3, 4, 5, 6])],
                             axis=1).set_axis(['L', 'L_enc'], axis=1)

    firstInput = str(firstLetter) + str(firstNumber)

    rightDiag = [firstInput]
    while len(rightDiag) <= 3:

        if firstNumber == 7:
            nextNumber = 7
        else:
            nextNumber = firstNumber + 1

        relativeN = charEncodedA['L_enc'][charEncodedA['L'] == firstLetter].reset_index()['L_enc'][0]

        if firstLetter == 'A':
            nextLetter = 'A'
        else:
            nextLetter = (charEncodedA['L'][charEncodedA['L_enc'] == relativeN - 1]).reset_index()['L'][0]

        nextValueDiagonal = str(nextLetter) + str(nextNumber)

        firstLetter = nextLetter
        firstNumber = nextNumber

        # print('Starting:', firstValue)
        # print('Resulting:', nextValueDiagonal)
        # print ('\n')

        rightDiag.append(nextValueDiagonal)

    # Hard-code per fare in modo che tutte le diagonali siano espresse

    rightDiag = pd.Series(rightDiag)
    hardCodeRight = pd.DataFrame(rightDiag.str.split('', expand=True)).set_axis(['cancella1', 'lettera', 'numero',
                                                                                 'cancella2'], axis=1)
    del [hardCodeRight['cancella1']]
    del [hardCodeRight['cancella2']]

    hardCodeRight = hardCodeRight.drop_duplicates(subset=['numero'])
    hardCodeRight = hardCodeRight.drop_duplicates(subset=['lettera'])

    hardCodeRight['comp'] = hardCodeRight['lettera'] + hardCodeRight['numero']

    rightDiag = list(hardCodeRight['comp'])

    # print('Right Diagonal:', rightDiag)

    baseCL = list()
    for character in coord0:
        baseCL.append(character)

    firstLetter = baseCL[0]
    firstNumber = int(baseCL[1])

    leftDiag = [firstInput]
    while len(leftDiag) <= 3:

        if firstNumber == 7:
            nextNumber = 7
        else:
            nextNumber = firstNumber + 1

        relativeN = charEncodedA['L_enc'][charEncodedA['L'] == firstLetter].reset_index()['L_enc'][0]

        if firstLetter == 'F':
            nextLetter = 'F'
        else:
            nextLetter = (charEncodedA['L'][charEncodedA['L_enc'] == relativeN + 1]).reset_index()['L'][0]

        nextValueDiagonal = str(nextLetter) + str(nextNumber)

        firstLetter = nextLetter
        firstNumber = nextNumber

        leftDiag.append(nextValueDiagonal)

    # Hard-code per fare in modo che tutte le diagonali siano espresse

    leftDiag = pd.Series(leftDiag)
    hardCodeLeft = pd.DataFrame(leftDiag.str.split('', expand=True)).set_axis(['cancella1', 'lettera', 'numero',
                                                                               'cancella2'], axis=1)
    del [hardCodeLeft['cancella1']]
    del [hardCodeLeft['cancella2']]

    hardCodeLeft = hardCodeLeft.drop_duplicates(subset=['numero'])
    hardCodeLeft = hardCodeLeft.drop_duplicates(subset=['lettera'])

    hardCodeLeft['comp'] = hardCodeLeft['lettera'] + hardCodeLeft['numero']

    leftDiag = list(hardCodeLeft['comp'])

    # print('Left Diagonal:', leftDiag)

    finalDiag = pd.concat([pd.Series(rightDiag), pd.Series(leftDiag)], axis=1).set_axis(['Right', 'Left'], axis=1)

    return finalDiag


def allWinningDiagonal():
    import pandas as pd

    columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
               'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7',
               'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
               'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
               'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

    diagWinL = list()
    diagWinR = list()

    for coord in columns:

        b = diagonalWinsCombinations(coord)

        if len(b['Right'].dropna()) == 4:
            diagWinR.append(b['Right'])
        if len(b['Left'].dropna()) == 4:
            diagWinL.append(b['Left'])

    totalDiag = pd.concat([pd.Series(diagWinR), pd.Series(diagWinL)], axis=1).set_axis(['Right', 'Left'], axis=1)

    return totalDiag


def identifyHorizontalWin(gameSeries):
    import pandas as pd
    import numpy as np

    eachGame = gameSeries[gameSeries != 0].reset_index()

    # VITTORIA ORIZZONTALE: la stessa lettera per quattro valori diversi. Isoliamo le singole lettere delle coordinate
    # nella singola giocata

    char = ['A', 'B', 'C', 'D', 'E', 'F']

    winH = list()

    for letter in char:

        cont = pd.concat([eachGame, eachGame['index'].str.contains(letter)], axis=1).set_axis(['1', '2', '3'], axis=1)
        cont = cont[cont['3'] == True]

        # A questo punto (per semplificarci la vita) prendiamo la vittoria delle X come unica vittoria.
        # Applichiamo una media rolling. per proprietà della media, se è ancora 1 al quarto valore, allora vuol dire
        # che i valori precedenti erano 1 (nel nostro setting, altrimenti un 2 e uno 0 sarebbero stati 1 lo stesso).

        winIndex = cont['2'].rolling(window=4, min_periods=1).mean().reset_index()
        del [winIndex['index']]

        # print(winIndex)

        # Definiamo (finalmente) la vincita orizzontale, tenendo anche conto degli Empty, altrimenti non riusciamo a
        # concatenare il database iniziale

        realWin = list()

        for possibleWin in winIndex:

            if (len(winIndex['2']) < 4) | (winIndex.empty):
                winH.append(0)

            if len(winIndex['2']) >= 4:
                if winIndex['2'][3] == 1:
                    winH.append(1)

            if len(winIndex['2']) >= 5:
                if winIndex['2'][4] == 1:
                    winH.append(1)

            if len(winIndex['2']) >= 6:
                if winIndex['2'][5] == 1:
                    winH.append(1)

            if len(winIndex['2']) >= 7:
                if winIndex['2'][6] == 1:
                    winH.append(1)

            if len(winIndex['2']) >= 4:
                if winIndex['2'][3] != 1:
                    winH.append(0)

    if len(pd.Series(winH).unique()) == 2:
        realWin.append(1)
    if len(pd.Series(winH).unique()) != 2:
        realWin.append(0)

    for valueHorizontal in realWin:
        return valueHorizontal


def identifyVerticalWin(gameSeries):
    import pandas as pd
    import numpy as np

    # VITTORIA VERTICALE: lo stesso numero per quattro valori diversi. Isoliamo i singoli numeri delle coordinate
    # nella singola giocata

    eachGame = gameSeries[gameSeries != 0].reset_index()

    num = ['1', '2', '3', '4', '5', '6', '7']

    winV = list()
    for letter in num:
        cont = pd.concat([eachGame, eachGame['index'].str.contains(letter)], axis=1).set_axis(['1a', '2a', '3a'],
                                                                                              axis=1)
        cont = cont[cont['3a'] == True]

        # A questo punto (per semplificarci la vita) prendiamo la vittoria delle X come unica vittoria.
        # Applichiamo una media rolling. per proprietà della media, se è ancora 1 al quarto valore, allora vuol dire
        # che i valori precedenti erano 1 (nel nostro setting, altriemnti un 2 e uno 0 sarebbero stati 1 lo stesso).

        winIndex = cont['2a'].rolling(window=4, min_periods=1).mean().reset_index()
        del [winIndex['index']]

        # Definiamo (finalmente) la vincita verticale, tenendo anche conto degli Empty, altrimenti non riusciamo a
        # concatenare il database iniziale

        if (len(winIndex['2a']) < 4) | (winIndex.empty):
            winV.append(0)

        if len(winIndex['2a']) >= 4:
            if winIndex['2a'][3] == 1:
                winV.append(1)

        if len(winIndex['2a']) >= 5:
            if winIndex['2a'][4] == 1:
                winV.append(1)

        if len(winIndex['2a']) >= 6:
            if winIndex['2a'][5] == 1:
                winV.append(1)

        if len(winIndex['2a']) >= 7:
            if winIndex['2a'][6] == 1:
                winV.append(1)

        if len(winIndex['2a']) >= 4:
            if winIndex['2a'][3] != 1:
                winV.append(0)

    realWinV = list()
    if len(pd.Series(winV).unique()) == 2:
        realWinV.append(1)
    if len(pd.Series(winV).unique()) != 2:
        realWinV.append(0)

    for valueVertical in realWinV:
        return valueVertical


def identifyRightDiagonalWin(gameSeries):

    import pandas as pd

    rightDiag = list()

    eachGame = gameSeries.reset_index()

    # VITTORIA OBLIQUA - DIAGONALE DESTRA. Nella cella superiore abbiamo costruito un oggetto che definisce per ogni
    # coordinata tutte le possibili conbinazioni di vittorie oblique. Confrontiamo questa con ogni nostra giocata
    # e verifichiamo il pattern.

    diagR = list()
    for diag in allWinningDiagonal()['Right']:
        cont = pd.concat([eachGame, eachGame['index'].isin(diag)], axis=1).set_axis(['1b', '2b', '3b'],
                                                                                    axis=1)
        cont = cont[cont['3b'] == True]
        diagR.append(cont)

    rightInd = list()
    for series in diagR:

        if series.empty == False:
            if len(series['2b']) == 4:
                if (series['2b'].reset_index()['2b'][0]) == 1:
                    if len(series['2b'].unique()) == 1:
                        # print(series)
                        rightInd.append(1)

    rightDiag.append(rightInd)

    for rdV in rightDiag:
        if len(rdV) != 0:
            return (pd.Series(rdV).unique()[0])
        if len(rdV) == 0:
            return 0


def identifyLeftDiagonalWin(gameSeries):

    import pandas as pd

    leftDiag = list()

    eachGame = gameSeries.reset_index()

    # VITTORIA OBLIQUA - DIAGONALE SINISTRA. Nella cella superiore abbiamo costruito un oggetto che definisce per ogni
    # coordinata tutte le possibili conbinazioni di vittorie oblique. Confrontiamo questa con ogni nostra giocata
    # e verifichiamo il pattern.

    diagL = list()
    for diagLf in allWinningDiagonal()['Left']:
        cont = pd.concat([eachGame, eachGame['index'].isin(diagLf)], axis=1).set_axis(['1c', '2c', '3c'],
                                                                                      axis=1)
        cont = cont[cont['3c'] == True]
        diagL.append(cont)

    leftInd = list()
    for series in diagL:

        if series.empty == False:
            if len(series['2c']) == 4:
                if (series['2c'].reset_index()['2c'][0]) == 1:
                    if len(series['2c'].unique()) == 1:
                        # print(series)
                        leftInd.append(1)

    leftDiag.append(leftInd)

    for ldV in leftDiag:
        if len(ldV) != 0:
            return (pd.Series(ldV).unique()[0])
        if len(ldV) == 0:
            return 0


def playCells (gameSeries):

    import pandas as pd

    game = gameSeries.reset_index().set_axis(['index', 'value'], axis=1)

    # Dividiamo per COLONNA (== Per numero)
    num = ['1', '2', '3', '4', '5', '6', '7']

    nextValues = list()
    for column in num:
        gameProxy = pd.concat([game, game['index'].str.contains(column)], axis=1).set_axis(['1aa', '2aa', '3aa'],
                                                                                           axis=1)
        gameProxy = gameProxy[(gameProxy['3aa'] == True) & (gameProxy['2aa'] == 0)].reset_index()

        if gameProxy.empty == False:
            nextValues.append(gameProxy['1aa'][0])

    return nextValues


def generateGameScenarios(gameSeries):
    import pandas as pd
    import numpy as np

    game = gameSeries.reset_index().set_axis(['index', 'value'], axis=1)

    # Diviamo per COLONNA (== Per numero)
    num = ['1', '2', '3', '4', '5', '6', '7']

    nextValues = list()
    for column in num:
        gameProxy = pd.concat([game, game['index'].str.contains(column)], axis=1).set_axis(['1aa', '2aa', '3aa'],
                                                                                           axis=1)
        gameProxy = gameProxy[(gameProxy['3aa'] == True) & (gameProxy['2aa'] == 0)].reset_index()

        if gameProxy.empty == False:
            nextValues.append(gameProxy['1aa'][0])

    # Adesso bisogna passare alla definizione degli SCENARI. Uno scenario è una partita dove UNA CASELLA Ammissibile sia
    # fillata con una 'X'

    # Ci tocca farlo in modo molto stupido, lento, noioso

    scenarios = list()
    for casella in nextValues:
        gameCells = game[game['index'] == casella]

        # Per evitare quel fastidiosissimo SettingWithCopyWarning
        gameCells = gameCells.copy()
        gameCells['value'] = 1

        preScenario = pd.concat([game, gameCells], axis=0)
        preScenario = preScenario.drop_duplicates(subset='index', keep='last').sort_index()

        preScenario['scenario'] = np.full(len(preScenario['index']), casella)

        preScenario = preScenario.set_index('index')

        scenarios.append(preScenario)

    return scenarios


def convertToVisualGame(gameSeries):
    import pandas as pd
    import numpy as np

    provaForza4Serie = gameSeries

    provaForza4Serie = provaForza4Serie.sort_index()
    provaForza4Serie = provaForza4Serie.copy()
    provaForza4Serie = provaForza4Serie.astype(float)

    # Mettiamo questa serie come vogliamo la tavola per giocare

    # dividiamo per colonna (== per numero)

    num = ['1', '2', '3', '4', '5', '6', '7']

    matrixBase = list()
    for numero in num:
        jnk = provaForza4Serie[provaForza4Serie.index.str.contains(numero)].reset_index().set_axis(['index', 'valori'],
                                                                                                   axis=1)['valori']
        matrixBase.append(jnk)

    finalMatrix = np.array(pd.concat([row for row in matrixBase], axis=1))
    # print(finalMatrix)

    return finalMatrix


def convertMatchtoSeries(matchBoard):
    import pandas as pd
    import numpy as np

    # Convertiamo una partita in serie
    # sGame = convertToVisualGame(matchBoard)

    # print(sGame)

    df = pd.DataFrame(matchBoard)

    columns = ['F1', 'E1', 'D1', 'C1', 'B1', 'A1',
               'F2', 'E2', 'D2', 'C2', 'B2', 'A2',
               'F3', 'E3', 'D3', 'C3', 'B3', 'A3',
               'F4', 'E4', 'D4', 'C4', 'B4', 'A4',
               'F5', 'E5', 'D5', 'C5', 'B5', 'A5',
               'F6', 'E6', 'D6', 'C6', 'B6', 'A6',
               'F7', 'E7', 'D7', 'C7', 'B7', 'A7']

    colonna = list()
    for value in range(7):
        colonna.append(df[value])

    colonna = pd.concat([series for series in colonna], axis=0)

    colonna = pd.DataFrame(colonna).set_index(pd.Series(columns))

    # print('\n')
    # print(convertToVisualGame(colonna))

    return colonna


def algoMove(gameSeriesFormat, model = 'SVM', version = 'V2', return_column=False):

    import pandas as pd
    import numpy as np
    import pickle
    import random

    # sGame = convertToVisualGame(game)

    # Hard Coding per evitare la vittoria su due fronti nelle prime giocate

    correctIndex = gameSeriesFormat[gameSeriesFormat[0] != 0]
    w = pd.Series(['F3', 'F4', 'F5'])
    correct = pd.concat([w, w.isin(pd.Series(playCells(gameSeriesFormat)))], axis = 1)
    correct = correct[correct[1] == True]

    # print(sGame)

    FW = smartForceWin(gameSeriesFormat)
    HC = blockOpponentWin(gameSeriesFormat)

    # Verifica se si può fare una ultima mossa per vincere

    if len(correctIndex[0]) == 1:
        finalDec = random.choice(list(correct[0]))
        print('Decision: Kick-Off')

    elif FW != 0:
        finalDec = FW
        print('Decision: Force Win', '(', FW, ')')

    # Vediamo se c'è qualche mossa avversaria da bloccare con un Hard-Coding

    elif HC != 0:
        finalDec = HC
        print('Decision: Hard Code', '(', HC, ')')

    else:

    # Importiamo i modelli

        if model == 'SVM':
            model = pickle.load(
                open(r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\SVM_model' + version + '.sav', 'rb'))

        if model == 'Logistic':
            model = pickle.load(
                open(r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Logistic_model' + version + '.sav', 'rb'))

        if model == 'KNN':
            model = pickle.load(
                open(r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\KNN_model' + version + '.sav', 'rb'))

        # Creiamo gli scenari possibili per la giocata

        scenarios = generateGameScenarios(gameSeriesFormat)

        valueScenario = list()
        winChanceSVM = list()
        for singleGame in scenarios:

            # Prepariamo ogni partita per essere processata

            test = np.array(singleGame['value']).reshape(1, -1)
            predictionSVM = model.predict_proba(test)[0][1]

            winChanceSVM.append(predictionSVM)
            valueScenario.append(singleGame['scenario'].unique()[0])

        decisionDf = pd.concat([pd.Series(winChanceSVM), pd.Series(valueScenario)], axis=1).set_axis(['% win',
                                                                                                      'scenario'],
                                                                                                     axis=1)
        # Decisione Finale

        # print('\n')
        # print('Where to play:', decisionDf['scenario'][decisionDf['% win'] == decisionDf['% win'].max()].reset_index()['scenario'][0])

        finalDec = decisionDf['scenario'][decisionDf['% win'] == decisionDf['% win'].max()].reset_index()['scenario'][0]

        # Questo è un test che serve per evitare che l'algoritmo regali una vittoria all'avversario pensando solo a se
        # da bravo egoista quale è

        gameSeriesFormat[0][finalDec] = 1

        if blockOpponentWin(gameSeriesFormat) != 0:
             filter = decisionDf[decisionDf['% win'] != decisionDf['% win'].max()]
             finalDec = filter['scenario'][filter['% win'] == filter['% win'].max()].reset_index()['scenario'][0]

             print('Decision: Algorithm (second best)')

        else:
             print('Decision: Algorithm (first best)')

    # Facciamo "Giocare il modello"

    gameSeriesFormat[0][finalDec] = 1
    fGame = convertToVisualGame(gameSeriesFormat[0])

    # test: proviamo a salvare ogni partita che fa
    #runningData = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\runningData1.xlsx")
    #runningData = pd.concat([runningData, gameSeriesFormat[0]], axis = 0)
    #runningData = runningData.transpose()
    #runningData.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\runningData1.xlsx")

    if return_column == True:
        return int(finalDec[1]) - 1

    else:
        return fGame


def createModel (database, trainSize, version = 'V2'):

    import numpy as np
    import pandas as pd
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    import pickle


    cutOff = trainSize

    trainSet = database[:cutOff]
    testSet = database[cutOff:len(database['Win'])]

    print('\n')
    print('TOTAL GAMES (TRAIN):', len(trainSet['Win']))
    print('TOTAL WINS (TRAIN):', len(trainSet['Win'][trainSet['Win'] == 1]))

    print('\n')
    print('TOTAL GAMES (TEST):', len(testSet['Win']))
    print('TOTAL WINS (TEST):', len(testSet['Win'][testSet['Win'] == 1]))
    print('\n')

    # Trainiamo il modello sulla variabile WIN

    # Aggiustiamo le variabili di input - TRAIN

    y_train = np.array(trainSet['Win']).reshape(-1, 1)
    y_train = np.ravel(y_train)
    X_train = np.array(trainSet.loc[:, trainSet.columns != 'Win'])

    # Aggiustiamo le variabili di input - TEST

    y_test = np.array(testSet['Win']).reshape(-1, 1)
    y_test = np.ravel(y_test)
    X_test = np.array(testSet.loc[:, testSet.columns != 'Win'])

    # Definiamo il modello - testiamo più modelli di classificazione

    # SVM

    print('Fitting SVM...')
    print('\n')

    modelSVC = svm.SVC(kernel='rbf', probability=True)
    modelSVCFit = modelSVC.fit(X_train, y_train)

    # Logistic

    print('Fitting Logistic...')
    print('\n')

    modelLogistic = LogisticRegression(multi_class='ovr', fit_intercept=True)
    modelLogisticFit = modelLogistic.fit(X_train, y_train)

    # KNN

    print('Fitting KNN...')
    print('\n')

    possibleK = np.arange(5, 25)
    scoreList = list()
    KList = list()
    for k in possibleK:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        scoreP = clf.score(X_test, y_test)
        scoreList.append(scoreP)
        KList.append(k)

    scoreList = pd.Series(scoreList)
    KList = pd.Series(KList)
    scoreBase = pd.concat([KList, scoreList], axis=1).set_axis(['K-Value', 'Score'], axis=1)

    bestK = scoreBase['K-Value'][scoreBase['Score'] == scoreBase['Score'].max()].reset_index()['K-Value'][0]

    clf = KNeighborsClassifier(n_neighbors=bestK)
    clfFit = clf.fit(X_train, y_train)

    test_score = clfFit.score(X_test, y_test)

    # Vediamo l'accuratezza tramite uno score R2 sul test set

    print('Model Accuracy (SVM):', round(modelSVCFit.score(X_test, y_test) * 100, 2), '%')
    print('Model Accuracy (Logistic):', round(modelLogisticFit.score(X_test, y_test) * 100, 2), '%')
    print('Model Accuracy (KNN, k =', bestK, ') : ', round(test_score * 100, 2), '%')

    # Storiamo i modelli per non dovere fare train ogni volta che usiamo il modello

    filenameSVM = r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\SVM_model' + version + '.sav'
    filenameLogistic = r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Logistic_model' + version + '.sav'
    filenameKNN = r'C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\KNN_model' + version + '.sav'

    pickle.dump(modelSVC, open(filenameSVM, 'wb'))
    pickle.dump(modelLogistic, open(filenameLogistic, 'wb'))
    pickle.dump(clf, open(filenameKNN, 'wb'))

    print('\n')
    print('The Models has been Trained and Stored')


def hardCodeHorizontalWins (gameSeries):

    import pandas as pd
    import Forza4Methods as f4

    # definiamo un metodo per fare un hard-code che vada a bloccare le vincite (possibili) per un avversario. Intanto
    # la vincita per essere tale deve appartenere ai possibili scenari di giocata. Quindi definiamo tutte le possibili
    # giocate in ciascuna partita. Se tutte le possibili giocate da tre hanno uno 0, questo viene considerato una vincita
    # potenziale SOLO se è in questo insieme

    ad = f4.playCells(gameSeries)  # Ci serve per dopo

    # Iniziamo con la vittoria in ORIZZONTALE partendo dal codice IdentifyHorizontalWin

    eachGame = gameSeries.reset_index()

    # identifichiamo le lettere

    char = ['A', 'B', 'C', 'D', 'E', 'F']

    winH = list()

    for letter in char:

        cont = pd.concat([eachGame, eachGame['index'].str.contains(letter)], axis=1).set_axis(['1', '2', '3'], axis=1)
        cont = cont[cont['3'] == True].reset_index()

        winIndex = pd.DataFrame(cont['2'].rolling(window=3, min_periods=1).mean()).set_index(cont['1'])[2:]

        possibleWins = list()
        for value in winIndex[winIndex == 2].dropna().index:

            # Questa però equivale alla cella dove è PRESENTE l'ultimo numero giocato, a noi serve quello DOPO

            nextCol = int(value[1]) + 1
            if nextCol > 7:
                nextCol = 7

            nextCol = str(nextCol)

            # Ora codifichiamo anche il caso che ci sia uno zero anche PRIMA

            beforeCol = int(value[1]) - 3
            if beforeCol < 1:
                beforeCol = 1

            beforeCol = str(beforeCol)

            possibleWins.append(value[0] + nextCol)
            possibleWins.append(value[0] + beforeCol)

        winH.append(possibleWins)

    almostHor = list()
    for listH in winH:
        for value in listH:
            almostHor.append(value)

    almostHor = pd.Series(almostHor, dtype=pd.StringDtype())

    # Adesso compariamo questo codice con quello delle giocate ammissibili definite dagli scenari sopra

    hardCodeBase = almostHor[almostHor.isin(ad)].reset_index()
    del[hardCodeBase['index']]

    if hardCodeBase.empty == False:
        playIn = hardCodeBase[0][0]
        return playIn

    if hardCodeBase.empty == True:
        return []


def hardCodeVerticalWins (gameSeries):

    import pandas as pd
    import Forza4Methods as f4

    # Adesso sviluppiamo un hard code che ci permetta di EVITARE LA VITTORIA VERTICALE dell'avversario
    # Il codice è molto simile a quello che abbiamo creato prima per bloccare la vittoria orizzontale, con il solo fatto
    # che passare da una lettera all'altra è più difficile che passare da un numero all'altro, e quindi per forza di cose
    # servirà un mapping

    ad = f4.playCells(gameSeries)  # Ci serve per dopo

    eachGame = gameSeries.reset_index()  # Anche qua gli zeri ci servono

    num = ['1', '2', '3', '4', '5', '6', '7']

    # Creiamo il mapping per muoversi in mezzo alle lettere

    mappingChar = ['A', 'B', 'C', 'D', 'E', 'F']
    mappingNum = ['1', '2', '3', '4', '5', '6']

    mapping = pd.concat([pd.Series(mappingNum).astype(int), pd.Series(mappingChar)], axis=1).set_axis(['num', 'lett'],
                                                                                                      axis=1)

    winV = list()
    for letter in num:
        cont = pd.concat([eachGame, eachGame['index'].str.contains(letter)], axis=1).set_axis(['1a', '2a', '3a'],
                                                                                              axis=1)
        cont = cont[cont['3a'] == True]

        winIndex = pd.DataFrame(cont['2a'].rolling(window=3, min_periods=1).mean()).set_index(cont['1a'])[2:]

        # Adesso bisogna definire la casella che si trova tre caselle prima del 2.0, e quella che sta dopo, per questo
        # Usiamo il mapping che abbiamo definito sopra

        possibleWins = list()
        for value in winIndex[winIndex == 2].dropna().index:
            numberFixed = value[1]

            # numero relativo alla lettera

            try:
                numberRelative = mapping['num'][mapping['lett'] == value[0]].reset_index()['num'][0]
                # aggiorniamo il numero
                numberTargetAfter = numberRelative - 1
                if numberTargetAfter > 8:
                    numberTargetAfter = 7
            except:
                numberRelative = None

            # Prendiamo la lettera sulla base del numero aggiornato

            try:
                PWLetterAfter = mapping['lett'][mapping['num'] == numberTargetAfter].reset_index()['lett'][0]
                possibleWins.append(PWLetterAfter + numberFixed)
            except:
                PWLetterAfter = None

        winV.append(possibleWins)

    almostVert = list()
    for listV in winV:
        for value in listV:
            almostVert.append(value)

    almostVert = pd.Series(almostVert, dtype=pd.StringDtype())

    # Adesso compariamo questo codice con quello delle giocate ammissibili definite dagli scenari sopra

    hardCodeBase = almostVert[almostVert.isin(ad)].reset_index()
    del [hardCodeBase['index']]

    if hardCodeBase.empty == False:
        playIn = hardCodeBase[0][0]
        return playIn

    if hardCodeBase.empty == True:
        return []


def hardCodeDiagonalRightWins(gameSeries):

    import pandas as pd
    import Forza4Methods as f4
    import numpy as np

    ad = f4.playCells(gameSeries)

    # HARD-CODE PER VITTORIA OBLIQUA (DESTRA) - il metodo che si deve usare qua è diverso, e molto più simile alla funzione
    # per definire la vittoria

    eachGame = gameSeries.reset_index()
    diagR = list()
    for diag in f4.allWinningDiagonal()['Right']:
        cont = pd.concat([eachGame, eachGame['index'].isin(diag)], axis=1).set_axis(['1b', '2b', '3b'],
                                                                                    axis=1)
        cont = cont[cont['3b'] == True]
        diagR.append(cont)

    possibleWin = list()
    for series in diagR:
        series = series.reset_index()
        lastCell = series['1b'][len(series['2b']) - 1]
        series = series[:3]
        #print(series)
        del [series['index']]
        # Scremiamo per le possibili vittorie sulla destra, quindi le seire per cui la media è 2 (media di 3 volte 2 == 2,
        # per proprietà della media). Prendiamo poi l'ultima osservazione (quella che manca al conto) e la concateniamo
        # alla media della serie dei primi 3, di modo che sia chiaro DOVE GIOCARE, e SE GIOCARE.

        identWin = pd.concat([pd.Series(series['2b'].mean()), pd.Series(np.full(1, lastCell))], axis=1)

        # In questo modo abbiamo un dataset con due colonne: la media delle osservazioni nei primi 3 valori della serie, e
        # dove giocare

        possibleWin.append(identWin)

    possibleWin = pd.concat([series for series in possibleWin], axis=0)

    #print(possibleWin)

    # Selezioniamo la vittoria, e dove giocare per fermarla

    isWin = possibleWin[1][possibleWin[0] == 2]
    isWin = pd.concat([isWin, isWin.isin(pd.Series(ad))], axis = 1).set_axis([0, 1], axis = 1)
    isWin = isWin[0][isWin[1] == True]

    if isWin.empty == False:
        hardCode = isWin.reset_index()
        del [hardCode['index']]

        return hardCode[0][0]

    if isWin.empty:
        return []


def hardCodeDiagonalLeftWins(gameSeries):

    import pandas as pd
    import Forza4Methods as f4
    import numpy as np

    # Manca solo la diagonale sinistra, per cui copiamo da zero quello della destra e cambiamo il puntamento della funzione
    # su 'Left'

    ad = f4.playCells(gameSeries)

    eachGame = gameSeries.reset_index()
    diagR = list()
    for diag in f4.allWinningDiagonal()['Left']:
        cont = pd.concat([eachGame, eachGame['index'].isin(diag)], axis=1).set_axis(['1b', '2b', '3b'],
                                                                                    axis=1)
        cont = cont[cont['3b'] == True]
        diagR.append(cont)

    possibleWin = list()
    for series in diagR:
        series = series.reset_index()
        lastCell = series['1b'][len(series['2b']) - 1]
        series = series[:3]
        #print(series)
        del [series['index']]
        # Scremiamo per le possibili vittorie sulla destra, quindi le seire per cui la media è 2 (media di 3 volte 2 == 2,
        # per proprietà della media). Prendiamo poi l'ultima osservazione (quella che manca al conto) e la concateniamo
        # alla media della serie dei primi 3, di modo che sia chiaro DOVE GIOCARE, e SE GIOCARE.

        identWin = pd.concat([pd.Series(series['2b'].mean()), pd.Series(np.full(1, lastCell))], axis=1)

        # In questo modo abbiamo un dataset con due colonne: la media delle osservazioni nei primi 3 valori della serie, e
        # dove giocare

        possibleWin.append(identWin)

    possibleWin = pd.concat([series for series in possibleWin], axis=0)

    #print(possibleWin)

    # Selezioniamo la vittoria, e dove giocare per fermarla

    isWin = possibleWin[1][possibleWin[0] == 2]
    isWin = pd.concat([isWin, isWin.isin(pd.Series(ad))], axis = 1).set_axis([0, 1], axis = 1)
    isWin = isWin[0][isWin[1] == True]

    if isWin.empty == False:
        hardCode = isWin.reset_index()
        del [hardCode['index']]

        return hardCode[0][0]

    if isWin.empty:
        return []


def blockOpponentWin (gameSeries):

    import pandas as pd
    import Forza4Methods as f4
    import numpy as np

    # serve che l'hard-coding sia tra le giocate Ammissibili
    # Questa cosa viene fuori soprattutto quando si ha un hard-coding Obliquo

    ad = playCells(gameSeries)

    hardCodeHor = f4.smartHardCodingHorizontal(gameSeries)
    hardCodeVert = f4.smartHardCodingVerticalWin(gameSeries)
    hardcodeDR = f4.smartHardCodingRight(gameSeries)
    hardcodeDL = f4.smartHardCodingLeft(gameSeries)

    hardCodeNecessary = [hardCodeHor, hardCodeVert, hardcodeDR, hardcodeDL]

    # Check che sia una giocata possibile, altrimenti non va bene per il gioco in toto

    hardCodeNecessary = pd.concat([pd.Series(hardCodeNecessary), pd.Series(hardCodeNecessary).isin(pd.Series(ad))],
                                  axis = 1).set_axis([0, 1], axis = 1)
    hardCodeNecessary = list(hardCodeNecessary[0][hardCodeNecessary[1] == True])

    hardCodeNeeded = list()
    for value in hardCodeNecessary:
         if len(value) != 0:
             hardCodeNeeded.append(value)

    if len(hardCodeNeeded) != 0:
        return hardCodeNeeded[0]
    if len(hardCodeNeeded) == 0:
        return 0


def updateDataBase (nGame):

    # GENERIAMO E AGGIORNIAMO IL DATABASE

    import pandas as pd
    import random
    import numpy as np
    from IPython.display import clear_output

    # Genera giocate random

    columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
               'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7',
               'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
               'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
               'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

    eligible = [1, 2]
    gamesNumber = nGame

    game = list()
    for nGames in range(gamesNumber):

        fullCells = int(np.random.uniform(0, 42, 1))

        it = 0
        while it < fullCells:

            for casella in range(fullCells):
                SGame = random.choice(eligible)
                game.append(SGame)

                it += 1

        for casella in range(len(columns) - fullCells):
            game.append(0)

    gameSeries = np.array(game).reshape(gamesNumber, len(columns))

    gameSeries = pd.DataFrame(gameSeries, columns=columns)

    legalGames = list()
    for col in gameSeries.transpose().columns:

        gameSeriesT = (gameSeries.transpose()[col])
        gs1 = gameSeriesT[gameSeriesT == 1].count()
        gs2 = gameSeriesT[gameSeriesT == 2].count()

        if gs1 == gs2:
            legalGames.append(gameSeries.transpose()[col])

    legalGames = pd.concat([s for s in legalGames], axis=1).transpose().drop_duplicates().reset_index()
    del [legalGames['index']]

    # Ora dobbiamo definire quando avviene una vittoria

    # Isoliamo ogni singola riga (== Ogni singola giocata)

    print('Admissible Games:', len(legalGames))

    Hor = list()
    Ver = list()
    DiagR = list()
    DiagL = list()

    for iteration, column in enumerate(legalGames.transpose().columns):
        eachGame = legalGames.iloc[column]

        Hor.append(fastHorizontalWinIdentification(eachGame))
        Ver.append(fastVerticalWinIdentifier(eachGame))
        DiagR.append(fastDiagonalRightWinIdentifier(eachGame))
        DiagL.append(fastDiagonalLeftWinIdentifier(eachGame))

        # Stato di Avanzamento
        print('\n')
        print('Game', iteration, 'analyzed, out of', len(legalGames['A1']))
        print('Progress', round(iteration / len(legalGames), 2) * 100, '%')
        clear_output(wait=True)

    wins = pd.concat([pd.Series(Hor), pd.Series(Ver), pd.Series(DiagR), pd.Series(DiagL)], axis=1).set_axis(
        ['Horizontal Win',
         'Vertical Win', 'Right Diagonal Win', 'Left Diagonal Win'], axis=1)

    dfWithWins = pd.concat([legalGames, wins], axis=1)
    # dfWithWins = pd.concat([dfWithWins, finalConcat], axis = 1)

    # Riassunto della creazione del Database

    print('\n')
    print('Total Games:', len(dfWithWins['Vertical Win']))
    print('Vertical Wins:', len(dfWithWins['Vertical Win'][dfWithWins['Vertical Win'] == 1]))
    print('Horizontal Wins:', len(dfWithWins['Horizontal Win'][dfWithWins['Horizontal Win'] == 1]))
    print('Diagonal Wins (Right):', len(dfWithWins['Right Diagonal Win'][dfWithWins['Right Diagonal Win'] == 1]))
    print('Diagonal Wins (Left):', len(dfWithWins['Left Diagonal Win'][dfWithWins['Left Diagonal Win'] == 1]))

    dfWithWins.loc[(dfWithWins['Horizontal Win'] == 1) | (dfWithWins['Vertical Win'] == 1) |
                   (dfWithWins['Right Diagonal Win'] == 1) | (dfWithWins['Left Diagonal Win'] == 1), 'Win'] = 1
    dfWithWins['Win'] = dfWithWins['Win'].fillna(0).astype(int)

    del [dfWithWins['Horizontal Win']]
    del [dfWithWins['Vertical Win']]
    del [dfWithWins['Right Diagonal Win']]
    del [dfWithWins['Left Diagonal Win']]

    baseData = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSet.xlsx")

    final = pd.concat([baseData, dfWithWins], axis=0)
    final = final.drop_duplicates()

    final.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSet.xlsx")

    print('\n')
    print('--UPDATED DATABASE--')
    print('\n')

    print('TOTAL GAMES:', len(final['Win']))
    print('TOTAL WINS:', len(final['Win'][final['Win'] == 1]))


def forceWin (gameSeriesFormat, default=1):

    import pandas as pd
    import Forza4Methods as f4
    import numpy as np

    # Algoritmo che ritorna 0 se non si deve forzare una vittoria e il valore di una cella se c'è da forzare una vittoria

    scenarios = f4.generateGameScenarios(gameSeriesFormat)

    forceWin = list()
    for scenario in scenarios:
        a = f4.fastVerticalWinIdentifier(scenario['value'], default=default)
        b = f4.fastDiagonalRightWinIdentifier(scenario['value'], default=default)
        c = f4.fastDiagonalLeftWinIdentifier(scenario['value'], default=default)
        d = f4.fastHorizontalWinIdentification(scenario['value'], default=default)
        sumUp = pd.concat(
            [pd.Series(a), pd.Series(b), pd.Series(c), pd.Series(d), pd.Series(scenario['scenario'].unique())],
            axis=1)
        forceWin.append(sumUp)

    forceWin = pd.concat([series for series in forceWin], axis=0)

    whereToPlay = forceWin[4][(forceWin[0] == 1) | (forceWin[1] == 1) | (forceWin[2] == 1) | (forceWin[3] == 1)]

    if whereToPlay.empty == False:
        return whereToPlay.reset_index()[4][0]

    if whereToPlay.empty:
        return 0


def fastHorizontalWinIdentification(gameSeries, default = 1):

    import pandas as pd
    import numpy as np

    char = ['A', 'B', 'C', 'D', 'E', 'F']
    HorComb = list()
    for letter in char:
        i = 0
        while i < 4:
            coordSeries = list()
            for k in np.arange(0, 4):
                firstS = letter + str(k + 1)
                ongoing = letter + str(int(firstS[1]) + i)
                coordSeries.append(ongoing)
            i += 1
            HorComb.append(coordSeries)

    win = list()
    for combination in HorComb:
        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if len(indexGame[indexGame == default]) == 4:
            win.append(1)

    if pd.Series(win, dtype=pd.StringDtype()).unique() == 1:
        return 1
    else:
        return 0

def fastDiagonalRightWinIdentifier (gameSeries, default = 1):

    import Forza4Methods as f4
    import pandas as pd

    win = list()
    for combination in f4.allWinningDiagonal()['Right']:

        combination = list(combination)

        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if len(indexGame[indexGame == default]) == 4:
            win.append(1)

    if pd.Series(win, dtype=pd.StringDtype()).unique() == 1:
        return 1
    else:
        return 0


def fastDiagonalLeftWinIdentifier(gameSeries, default = 1):

    import Forza4Methods as f4
    import pandas as pd

    win = list()
    for combination in f4.allWinningDiagonal()['Left']:

        combination = list(combination)

        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if len(indexGame[indexGame == default]) == 4:
            win.append(1)

    if pd.Series(win, dtype=pd.StringDtype()).unique() == 1:
        return 1
    else:
        return 0


def fastVerticalWinIdentifier(gameSeries, default = 1):

    import pandas as pd
    import numpy as np
    import Forza4Methods as f4

    lett = ['A', 'B', 'C', 'D', 'E', 'F']
    mapping = [1, 2, 3, 4, 5, 6]
    mappingC = pd.concat([pd.Series(lett), pd.Series(mapping)], axis=1).set_axis(['lettera', 'numero'], axis=1)
    comb = list()
    for numero in mapping:
        i = 0
        while i < 3:
            coordSeries = list()
            for k in np.arange(0, 4):
                firstS = mappingC['lettera'][mappingC['numero'] == (k + 1)].reset_index()['lettera'][0] + str(numero)
                relNum = mappingC['numero'][mappingC['lettera'] == firstS[0]].reset_index()['numero'][0]
                ongoing = mappingC['lettera'][mappingC['numero'] == int(relNum) + i].reset_index()['lettera'][0] + str(
                    numero)
                coordSeries.append(ongoing)
            i += 1
            comb.append(coordSeries)

    #print(comb)

    win = list()
    for combination in comb:
        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if len(indexGame[indexGame == default]) == 4:
            win.append(1)

    if pd.Series(win, dtype=pd.StringDtype()).unique() == 1:
        return 1
    else:
        return 0


def smartHardCodingHorizontal(gameSeries, default = 2.0):

    # HARD CODING ORIZZONTALE

    import pandas as pd
    import numpy as np

    char = ['A', 'B', 'C', 'D', 'E', 'F']
    HorComb = list()
    for letter in char:
        i = 0
        while i < 4:
            coordSeries = list()
            for k in np.arange(0, 4):
                firstS = letter + str(k + 1)
                ongoing = letter + str(int(firstS[1]) + i)
                coordSeries.append(ongoing)
            i += 1
            HorComb.append(coordSeries)

    win = list()
    for combination in HorComb:
        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if (len(indexGame[indexGame[0] == default]) == 3) & (len(indexGame[indexGame[0] == 0]) == 1):
            win.append(indexGame)


    if len(win) != 0:
        for series in win:
            return series[series[0] == 0].reset_index()['index'][0]
    else:
        return []


def smartHardCodingVerticalWin (gameSeries, default = 2.0):

    import pandas as pd
    import numpy as np
    import Forza4Methods as f4

    lett = ['A', 'B', 'C', 'D', 'E', 'F']
    mapping = [1, 2, 3, 4, 5, 6]
    numeroS = [1, 2, 3, 4, 5, 6, 7]
    mappingC = pd.concat([pd.Series(lett), pd.Series(mapping)], axis=1).set_axis(['lettera', 'numero'], axis=1)
    comb = list()
    for numero in numeroS:
        i = 0
        while i < 3:
            coordSeries = list()
            for k in np.arange(0, 4):
                firstS = mappingC['lettera'][mappingC['numero'] == (k + 1)].reset_index()['lettera'][0] + str(numero)
                relNum = mappingC['numero'][mappingC['lettera'] == firstS[0]].reset_index()['numero'][0]
                ongoing = mappingC['lettera'][mappingC['numero'] == int(relNum) + i].reset_index()['lettera'][0] + str(
                    numero)
                coordSeries.append(ongoing)
            i += 1
            comb.append(coordSeries)

    win = list()
    for combination in comb:
        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if (len(indexGame[indexGame[0] == default]) == 3) & (len(indexGame[indexGame[0] == 0]) == 1):
            win.append(indexGame)

    if len(win) != 0:
        for series in win:
            return series[series[0] == 0].reset_index()['index'][0]
    else:
        return []


def smartHardCodingLeft (gameSeries, default = 2.0):

    import Forza4Methods as f4
    import pandas as pd

    win = list()
    for combination in f4.allWinningDiagonal()['Left']:

        combination = list(combination)

        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if (len(indexGame[indexGame[0] == default]) == 3) & (len(indexGame[indexGame[0] == 0]) == 1):
            win.append(indexGame)

    if len(win) != 0:
        for series in win:
            return series[series[0] == 0].reset_index()['index'][0]
    else:
        return []


def smartHardCodingRight (gameSeries, default = 2.0):

    import Forza4Methods as f4
    import pandas as pd

    win = list()
    for combination in f4.allWinningDiagonal()['Right']:

        combination = list(combination)

        combination = pd.Series(combination)
        indexGame = gameSeries[gameSeries.index.isin(combination)]

        if (len(indexGame[indexGame[0] == default]) == 3) & (len(indexGame[indexGame[0] == 0]) == 1):
            win.append(indexGame)

    if len(win) != 0:
        for series in win:
            return series[series[0] == 0].reset_index()['index'][0]
    else:
        return []


def smartForceWin (gameSeries, default = 1):

    ad = playCells(gameSeries)

    # Serve un Force Win decisamente più veloce di quello che abbiamo. Quello attuale fa 28 calcoli per arrivare alla soluzione
    # Dobbiamo trovarne uno che sia meno computationally expensive. Quello di adesso di base itera per tutti gli scenari
    # possibili e per ogni scenario cerca delle vittorie. Per velocizzarlo si potrebbe utilizzare un metodo che sfrutta la
    # Tecnologia degli Hard-Coding che abbiamo creato prima. Di base identifica quasi-vittorie e gioca nella casella mancante
    # Se viene codificato per i due blocca le vittorie avversarie, ma se lo si codifica per gli 1, allora completa la
    # giocata che è iniziata. Quindi partiamo dal codice di BlockOpponentWin

    import pandas as pd
    import Forza4Methods as f4
    import numpy as np

    hardCodeHor = f4.smartHardCodingHorizontal(gameSeries, default = default)
    hardCodeVert = f4.smartHardCodingVerticalWin(gameSeries, default = default)
    hardcodeDR = f4.smartHardCodingRight(gameSeries, default = default)
    hardcodeDL = f4.smartHardCodingLeft(gameSeries, default = default)

    hardCodeNecessary = [hardCodeHor, hardCodeVert, hardcodeDR, hardcodeDL]

    # Vediamo se il forceWin è negli scenari possibili, perchè in questo modo evitiamo che entri da solo
    # ed eviti, ad esempio, un hard-coding

    hardCodeNecessary = pd.concat([pd.Series(hardCodeNecessary),
                                   pd.Series(hardCodeNecessary).isin(pd.Series(ad))],
                                  axis = 1).set_axis([0, 1], axis = 1)

    hardCodeNecessary = list(hardCodeNecessary[0][hardCodeNecessary[1] == True])

    hardCodeNeeded = list()
    for value in hardCodeNecessary:
         if len(value) != 0:
             hardCodeNeeded.append(value)

    if len(hardCodeNeeded) != 0:
        return hardCodeNeeded[0]
    if len(hardCodeNeeded) == 0:
        return 0
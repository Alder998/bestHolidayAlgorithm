# Raccogliamo i metodi per la predizione della vacanza ideale

def getRegressors(dataset):
    import pandas as pd

    dataClean = dataset

    # Good Choice non va mai messa tra i regressors

    if len(pd.Series(dataClean.columns).str.contains('Good Choice', case=False).unique()) == 2:
        del [dataClean['Good Choice']]

    IntRegressors = list()
    FloatRegressors = list()
    cl = list()
    for col in dataClean.columns:
        FloatRegressors.append(isinstance(dataClean[col].iloc[dataClean[col].first_valid_index()], float))
        IntRegressors.append(isinstance(dataClean[col].iloc[dataClean[col].first_valid_index()], int))
        cl.append(col)
    numCol = pd.concat([pd.Series(cl), pd.Series(FloatRegressors), pd.Series(IntRegressors)], axis=1)
    rCol = numCol[0][(numCol[1] == True) | (numCol[2] == True)]

    regressors = list()
    for regr in rCol:
        regressors.append(regr)

    return dataClean[regressors]


def getAirBnBExperienceData(place, checkIn, checkOut):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    # Settiamo gli Headers

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base

    target_url = "https://www.airbnb.com/s/" + place + "/experiences?refinement_paths%5B%5D=%2Fexperiences&tab_id=experience_tab&checkin=" + checkIn + "&checkout=" + checkOut + "&flexible_trip_lengths%5B%5D=one_week&rank_mode=default&date_picker_type=calendar&source=structured_search_input_header&search_type=filter_change"

    # settiamo la ricerca con BeautifulSoup

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('div')

    fs = list()
    for i in a_tag:
        fs.append(i)

    a = pd.Series(fs).astype(str)

    # a.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Prova.xlsx")

    # Filtriamo per dove sono gli hotel

    base = a[a.str.contains('<a aria-label="')].reset_index()

    # Estraiamo il titolo -------------------------------------------------------------------------------------
    nomeEsp = list()
    for value in range(len(base[0])):
        counterStart = base[0][value].find('aria-label') + (len('aria-label') + 2)
        nomeEsp.append(base[0][value][counterStart: (base[0][value][counterStart:]).find('"') + 22])
    nomeEsp = pd.DataFrame(nomeEsp).set_axis(['Experience'], axis=1)

    nomeEsp = nomeEsp[~nomeEsp['Experience'].str.contains('"')]

    # Estraiamo il Prezzo --------------------------------------------------------------------------------------

    priceG = list()
    for value in range(len(base[0])):
        counterStart = base[0][value].find('n6k4iv6 dir dir-ltr') + 62
        priceG.append(base[0][value][counterStart: (base[0][value])[counterStart:].find('<') + counterStart])
    priceG = pd.DataFrame(priceG).set_axis(['Price'], axis=1)

    # Estraiamo il rating -----------------------------------------------------------------------------------------

    judge = list()
    for value in range(len(base[0])):
        counterStart = base[0][value].find('k1conusl k8yrq8q dir dir-ltr') + 48
        judge.append(base[0][value][counterStart: (base[0][value])[counterStart:].find('"') + counterStart])

    judge = pd.DataFrame(judge).set_axis(['RevGr'], axis=1)

    # Scorporiamo rating e numero di recensioni

    # Rating

    rating = list()
    for value in range(len(base[0])):
        rating.append(judge['RevGr'][value][0:judge['RevGr'][value].find(' ')])

    rating = pd.DataFrame(rating).set_axis(['rating'], axis=1)

    # Numero di recensioni

    Nrec = list()
    for value in range(len(base[0])):
        Nrec.append(
            judge['RevGr'][value][judge['RevGr'][value].find('out') + 26: judge['RevGr'][value].find(' reviews')])

    numRev = pd.DataFrame(Nrec).set_axis(['Reviews (number)'], axis=1)

    # Inseriamo una colonna che ricapitola di che paese stiamo parlando

    placeC = pd.DataFrame(np.full(len(nomeEsp['Experience']), place)).set_axis(['Place'], axis=1)
    placeC = placeC.set_index(nomeEsp['Experience'].index)

    # Mettiamo tutto insieme

    df = pd.concat([nomeEsp, placeC, priceG, rating, numRev], axis=1).dropna()

    # per eliminare le caselle vuote

    df.loc[df['Experience'].str.len() == 0, 'Da Eliminare'] = 'Si'
    df = df[df['Da Eliminare'] != 'Si'].reset_index()
    del [df['index']]
    del [df['Da Eliminare']]

    # Rimuoviamo i duplicati (le esperienze con esattamente lo stesso numero di recensioni, lo stesso rating, lo stesso prezzo)

    df = df.drop_duplicates(subset=['Price', 'rating', 'Reviews (number)'], keep='last').reset_index()
    del [df['index']]

    # Sostituiamo le colonne di modo che rappresentino qualcosa di significativo

    df['rating'] = df['rating'].str.replace('.', ',', regex=False)
    df['Reviews (number)'] = df['Reviews (number)'].str.replace(',', '', regex=False)

    return df


def getExperienceList(granularity, checkIn, checkOut, get=False):
    import pandas as pd
    import numpy as np

    if get == False:

        tutteLeCitta = pd.read_excel(
            r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\worldcities.xlsx")

        # Teniamo lat e long per provare a localizzare i posti su una mappa

        cityList = tutteLeCitta[['city_ascii', 'country', 'population', 'lat', 'lng']]

        # Importiamo una lista dei paesi dell'Europa continentale

        europeanCountries = pd.read_excel(
            r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\List of European Countries.xlsx")[
            'Country']

        # filtriamo per le città dell'Europa

        eligibleCities = list()
        for country in europeanCountries:
            eligibleCities.append(cityList[cityList['country'] == country])

        eligibleCities = pd.concat([df for df in eligibleCities], axis=0).reset_index()
        del [eligibleCities['index']]

        # Abbiamo trovato le città che possiamo scegliere

        # filtriamo per la popolazione (sennò è troppo lungo)

        eligibleCities = eligibleCities[eligibleCities['population'] > granularity]

        print('\n')
        print('Found', len(eligibleCities), 'Locations')

        final = list()
        for countryI in europeanCountries:

            babyDF = eligibleCities['city_ascii'][eligibleCities['country'] == countryI]

            if len(babyDF) > 0:

                print('\n')
                print('Test DataSet length:', len(babyDF))

                checkIn = checkIn
                checkOut = checkOut

                print('\n')
                print('Creating Database...')
                print('\n')

                bigDB = list()
                for place in babyDF:
                    bigDB.append(getAirBnBExperienceData(place, checkIn, checkOut))
                    print(place, 'Analyzed')

                bigDB = pd.concat([df for df in bigDB], axis=0)
                bigDB = bigDB.drop_duplicates(subset='Experience').reset_index()
                del [bigDB['index']]

                # Inseriamo una colonna con il paese che scegliamo

                countryCol = pd.DataFrame(np.full(len(bigDB['Experience']), countryI)).set_axis(['Country'], axis=1)

                bigDB = pd.concat([bigDB, countryCol], axis=1)

                # bigDB.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Experience" + countryI + ".xlsx")

            final.append(bigDB)

        finalDF = pd.concat([df for df in final], axis=0)
        finalDF = finalDF.dropna(subset=['Reviews (number)']).reset_index()
        del [finalDF['index']]

        finalDF.to_excel(
            r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot.xlsx")

        return finalDF

    if get == True:
        getDf = pd.read_excel(
            r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot - Backup1.xlsx")

        return getDf


def createTrainDatasetAirBnB(dataset, granularity):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    df = dataset

    df = df.dropna(subset=['Reviews (number)']).reset_index()

    # Eliminiamo gli outliers sul prezzo e sul numero di reviews

    df['Z Rev'] = np.abs(stats.zscore(df['Reviews (number)']))
    df = df[df['Z Rev'] < 3]
    del [df['Z Rev']]
    del [df['index']]

    df['Z Price'] = np.abs(stats.zscore(df['Price']))
    df = df[df['Z Price'] < 3].reset_index()
    del [df['Z Price']]
    del [df['index']]

    # Abbiamo tutti i dati adesso (3189 esperienze in una trentina di paesi)
    # Dobbiamo CREARE il train set (nessuno ci dice quale posto è effettivamente migliore degli altri)
    # Un modo per farlo può essere clusterizzare (con K-Means) le osservazioni. Il Cluster che rappresenta
    # i valori più alti sarà quello che consideremo come valore 1 (== Buona scelta per una vacanza).
    # Questo creerà una nuova variabile, che sarà chiamata "buona scelta"

    # Il prezzo deve essere il minimo, e deve essere massimizzato. Per risolvere questa cosa creiamo una variabile nuova
    # che chiamiamo prezzo_C (== Corrected), ed è 1/Prezzo

    df['Price_C'] = (1 / df['Price']) * 10
    df['Price_CH'] = (1 / df['House Price']) * 10

    dfCluster = df[['Price_C', 'rating', 'Reviews (number)', 'Price_CH']]

    clModel = pd.DataFrame(KMeans(n_clusters=granularity).fit_predict(dfCluster)).set_axis(['Ranking'], axis=1)

    dfBase = pd.concat([df, clModel], axis=1)

    # Isoliamo la classe delle migliori scelte

    winCluster = \
        dfBase['Ranking'][dfBase['Reviews (number)'] == dfBase['Reviews (number)'].max()].reset_index()['Ranking'][0]

    dfBase.loc[dfBase['Ranking'] == winCluster, 'Good Choice'] = 1
    dfBase['Good Choice'] = dfBase['Good Choice'].fillna(0)

    # ELiminiamo le colonne superflue

    del [dfBase['Ranking']]
    del [dfBase['Price_C']]
    del [dfBase['Price_CH']]

    dfBase.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot - Prova.xlsx")

    print('Train Creation Analytics')
    print('\n')
    print('Number of Good Choices out of Total: ',
          ((dfBase['Good Choice'][dfBase['Good Choice'] == 1].count()) / (len(dfBase['rating']))) * 100, '%')
    print('Average Price Good Choice: ', ((dfBase['Price'][dfBase['Good Choice'] == 1].mean())), ', Total Sample: ',
          dfBase['Price'].mean())
    print('Average Rating Good Choice: ', ((dfBase['rating'][dfBase['Good Choice'] == 1].mean())), ', Total Sample: ',
          dfBase['rating'].mean())
    print('Average Number of Reviews Good Choice: ', ((dfBase['Reviews (number)'][dfBase['Good Choice'] == 1].mean())),
          ', Total Sample: ', dfBase['Reviews (number)'].mean())

    return dfBase


def defineModel(model, dataset, trainDim):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    dfBase = dataset

    split = train_test_split(dfBase, test_size=trainDim, random_state=1893)

    trainSet = split[1].reset_index()
    del [trainSet['index']]
    testSet = split[0].reset_index()
    del [testSet['index']]

    # Definiamo il Train Set
    y_train = np.array(trainSet['Good Choice']).reshape(-1, 1)
    y_train = np.ravel(y_train)
    X_train = np.array(getRegressors(trainSet))

    # Definiamo il Test Set
    y_test = np.array(testSet['Good Choice']).reshape(-1, 1)
    y_test = np.ravel(y_test)
    X_test = np.array(getRegressors(testSet))

    if model == 'Logistic':
        print('\n')
        print('--------------------- MODEL: Logistic ---------------------')
        print('\n')
        print('Regressors:', getRegressors(trainSet).columns)

        clf = LogisticRegression(multi_class='ovr', fit_intercept=False).fit(X_train, y_train)

        print('\n')
        print('Model Performance:', round(clf.score(X_test, y_test) * 100, 2), '%')

    if model == 'KNN':

        print('\n')
        print('--------------------- MODEL: KNN Algorithm ---------------------')
        print('\n')
        print('Regressors:', getRegressors(trainSet).columns)

        # Prendiamo tra i 5 e i 25 neighbors, selezioniamo quello con lo score R2 maggiore

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
        clf.fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        print('\n')
        print('Model Score ( k =', bestK, ') : ', round(test_score * 100, 2), '%')

    print('\n')
    probab = list()
    for prob in (clf.predict_proba(X_test)):
        probab.append(prob[1])

    probabilityToBeGoodChoice = pd.DataFrame(probab).set_axis(['Probability'], axis=1)

    # Creiamo il DataFrame Finale

    dfFinal = pd.concat([testSet, probabilityToBeGoodChoice], axis=1)

    # Creiamo un Output che sia una top ten dei posti dove andare

    TopTen = dfFinal.sort_values(by='Probability', ascending=False)
    TopTen = TopTen.drop_duplicates(subset='Place', keep='first').reset_index()
    del [TopTen['index']]

    print('Top 10 Best places to go holiday this summer:')
    print('\n')

    topTenCounter = np.arange(1, 10)
    for value in topTenCounter:
        print(TopTen['Place'][value], '(', TopTen['Country'][value], ')', '== Probability to be a good choice:',
              TopTen['Probability'][value])

    dfFinal.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot - ProvaProb.xlsx")

    return dfFinal


def bestDecisionsMap(probabilityDataset, visualize=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objs as go
    import plotly.io as pio

    # proviamo a vedere se riusciamo a rappresentare i punti senza i contorni della mappa assegnata

    tutteLeCitta = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\worldcities.xlsx")

    # Teniamo lat e long per provare a localizzare i posti su una mappa

    cityList = tutteLeCitta[['city_ascii', 'country', 'population', 'lat', 'lng']]

    # Importiamo una lista dei paesi dell'Europa continentale

    europeanCountries = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\List of European Countries.xlsx")[
        'Country']

    # filtriamo per le città dell'Europa

    eligibleCities = list()
    for country in europeanCountries:
        eligibleCities.append(cityList[cityList['country'] == country])

    eligibleCities = pd.concat([df for df in eligibleCities], axis=0).reset_index()
    del [eligibleCities['index']]

    # Modifichiamo le colonne per riuscire ad avere dati reali

    eligibleCities['lat'] = eligibleCities['lat'].astype(str)
    eligibleCities['lng'] = eligibleCities['lng'].astype(str)

    trueLat = list()
    for valueLat in eligibleCities['lat']:
        trueLat.append(valueLat[:-4] + '.' + valueLat[2:])

    trueLat = pd.Series(trueLat).astype(float)

    trueLon = list()
    for valueLon in eligibleCities['lng']:
        trueLon.append(valueLon[:-6] + '.' + valueLon[len(valueLon[:-5]): len(valueLon) - 2])

    trueLon = pd.Series(trueLon)

    trueLon = trueLon.str.replace('..', '.', regex=False)
    trueLon = trueLon.str.replace('-.0.', '0', regex=False)
    trueLon = trueLon.str.replace('.0.', '0', regex=False)
    trueLon = trueLon.str.replace('.-0.', '0', regex=False)
    trueLon = trueLon.str.replace('.-0', '0', regex=False)

    trueLon = trueLon.astype(float)

    trueLat = pd.DataFrame(trueLat).set_axis(['lat'], axis=1)
    trueLon = pd.DataFrame(trueLon).set_axis(['lng'], axis=1)

    eligibleCities['lat'] = trueLat['lat']
    eligibleCities['lng'] = trueLon['lng']

    eligibleCities = eligibleCities.set_axis(['Place', 'country', 'population', 'lat', 'lng'], axis=1).set_index(
        'Place')

    places = probabilityDataset.set_index('Place')

    placesWithCoord = places.join(eligibleCities, rsuffix='_d')
    # print(placesWithCoord)

    print('\n')

    if visualize == True:
        fig = px.scatter_geo(placesWithCoord, lat=placesWithCoord['lat'], lon=placesWithCoord['lng'],
                             color=placesWithCoord['Probability'])
        fig.update_layout(
            title='Eligible Cities',
            geo_scope='europe',
        )

        fig.update_traces(marker=dict(sizemode='diameter', sizeref=0.05, size=placesWithCoord['Probability']))
        subfig = go.Scattergeo(lat=placesWithCoord['lat'], lon=placesWithCoord['lng'],
                               marker=dict(size=3, color='darkblue'))

        # Add the small markers trace to the plot
        fig.add_trace(subfig)

        # fig.update_traces(marker=dict(size=2))
        fig.show()


def getAirBnBHousePrice(place, checkIn, checkOut, adults):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    # Settiamo gli Headers

    place = place
    checkIn = checkIn
    checkOut = checkOut
    adults = adults

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base

    target_url = "https://www.airbnb.com/s/" + place + "/homes?adults=" + str(
        adults) + "&refinement_paths%5B%5D=%2Fhouses&tab_id=house_tab&checkin=" + checkIn + "&checkout=" + checkOut

    # settiamo la ricerca con BeautifulSoup

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('div')

    fs = list()
    for i in a_tag:
        fs.append(i)

    a = pd.Series(fs).astype(str)

    # a.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Prova.xlsx")

    # Filtriamo per dove sono gli hotel

    base = a[a.str.contains('Apartment')].reset_index()

    if base.empty == False:

        # Estraiamo il nome della casa --------------------------------------------------------------------------------------

        nameG = list()
        for value in range(len(base[0])):
            counterStart = base[0][value].find('t6mzqp7 dir dir-ltr') + 63
            nameG.append(base[0][value][counterStart: (base[0][value])[counterStart:].find('<') + counterStart])
        nameG = pd.DataFrame(nameG).set_axis(['Apartment Name'], axis=1)

        nameG = nameG[~nameG['Apartment Name'].str.contains('title')]
        nameG = nameG.drop_duplicates()

        # Estraiamo il prezzo della casa --------------------------------------------------------------------------------------

        priceG = list()
        for value in range(len(base[0])):
            counterStart = base[0][value].find('_tyxjp1') + 11
            priceG.append(base[0][value][counterStart: (base[0][value])[counterStart:].find('<') + counterStart])
        priceG = pd.DataFrame(priceG).set_axis(['Apartment Price'], axis=1)

        priceG = priceG[~priceG['Apartment Price'].str.contains('=')]
        priceG = priceG.drop_duplicates()

        # Mettiamo insieme tutto

        dfHouses = pd.concat([nameG, priceG], axis=1).reset_index()
        del [dfHouses['index']]

        dfHouses = dfHouses.dropna()

        dfHouses['Apartment Price'] = dfHouses['Apartment Price'].astype(str)
        dfHouses['Apartment Price'] = dfHouses['Apartment Price'].str.replace(',', '').astype(float)

        return dfHouses

    else:
        return pd.DataFrame(list())


def getAveragePrice(place, checkIn, checkOut, adults):
    import pandas as pd

    base = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot - Backup.xlsx")
    base = base[base['Country'] == 'Switzerland']

    hh = list()
    town = list()
    for place in base['Place'].unique():
        a = getAirBnBHousePrice(place, checkIn, checkOut, adults)
        houseAv = a['Apartment Price'].mean()

        hh.append(houseAv)
        town.append(place)

        print('Average Price', place, ':', round(a['Apartment Price'].mean(), 2))

    hh = pd.Series(hh)
    town = pd.Series(town)

    houseDf = pd.concat([town, hh], axis=1).set_axis(['Place', 'Average House Price'], axis=1)

    # houseDf.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\HousePricesAirBnB - Test.xlsx")

    return houseDf

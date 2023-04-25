# Qui raccogliamo qualche metodo per fare web scraping

def getSingleHotelDataWithFeatures(typeAcc, nome, checkin, checkout, adults, children, numberOfRooms, currency):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    typeAcc = typeAcc
    checkin = checkin
    checkout = checkout
    hotelName = nome
    adults = str(adults)
    children = str(children)
    numberOfRooms = str(numberOfRooms)
    currency = currency

    l = list()
    g = list()
    o = {}
    k = {}
    fac = []
    fac_arr = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    target_url = "https://www.booking.com/hotel/it/" + hotelName + ".it.html?checkin=" + checkin + "&checkout=" + checkout + "&group_adults=" + adults + "&group_children=" + children + "&no_rooms=" + numberOfRooms + "&selected_currency=" + currency + "&lang=en"

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        o["name"] = soup.find("h2", {"class": "pp-header__title"}).text
    except:
        o["name"] = None

    try:
        o["address"] = soup.find("span", {"class": "hp_address_subtitle"}).text.strip("\n")
    except:
        o["address"] = None

    try:
        o["rating"] = soup.find("div", {"class": "d10a6220b4"}).text
    except:
        o["rating"] = None

    try:
        o["reviews"] = soup.find("div", {"class": "db63693c62"}).text
    except:
        o["reviews"] = None

    fac = soup.find_all("div", {"class": "important_facility"})
    for i in range(0, len(fac)):
        fac_arr.append(fac[i].text.strip("\n"))
    ids = list()
    targetId = list()
    try:
        tr = soup.find_all("tr")
    except:
        tr = None
    for y in range(0, len(tr)):
        try:
            id = tr[y].get('data-block-id')
        except:
            id = None
        if (id is not None):
            ids.append(id)
    print("ids are ", len(ids))
    for i in range(0, len(ids)):
        try:
            allData = soup.find("tr", {"data-block-id": ids[i]})
            try:
                rooms = allData.find("span", {"class": "hprt-roomtype-icon-link"})
            except:
                rooms = None
            if (rooms is not None):
                last_room = rooms.text.replace("\n", "")
            try:
                k["room"] = rooms.text.replace("\n", "")
            except:
                k["room"] = last_room
            price = allData.find("div", {
                "class": "bui-price-display__value prco-text-nowrap-helper prco-inline-block-maker-helper prco-f-font-heading"})
            k["price"] = price.text.replace("\n", "")

            g.append(k)
            k = {}
        except:
            k["room"] = None
            k["price"] = None
    l.append(g)
    l.append(o)
    l.append(fac_arr)

    #print(l)

    camere = list()
    prezziCamere = list()
    rating = list()

    if len(l[0]) != 0:

        for camera in l[0]:
            camere.append(pd.Series(camera['room']))
            prezziCamere.append(pd.Series(camera['price'][2:len(camera['price'])]))

        features = l[2]

        #print(features)

        if pd.Series(features).empty:
            print('Warning! No features found')
        else:
            print('Found', len(features), 'features')

        i = 1
        while i < 3:
            print('Retying to get the features...')
            features = getSingleHotelDataList(nome, checkin, checkout, adults, children, numberOfRooms, currency,
                                              features=True)
            if pd.Series(features).empty == False:
                break

            i += 1

        print('Found', len(features), 'features')

        features = " ".join([str(f) for f in features])

        # Abbiamo le features, che altro non sono che una semplice lista di cose. L'idea per collocarle in modo che siano mangiabili
        # da un modello sarebbe metterle in formato dummy variables. Quindi, se un hotel ha n features, allora ci saranno n colonne
        # popolate da un 1 se quell'hotel le ha. Farlo per un Hotel singolo non è difficile.

        rating = l[1]['rating']
        reviews = l[1]['reviews']

        try:
            reviews = reviews[0:reviews.find(' ')]
        except:
            reviews = None

        camere = pd.concat([series for series in camere], axis=0)
        prezziCamere = pd.concat([series for series in prezziCamere], axis=0)

        camereConPrezzi = pd.concat([camere, prezziCamere], axis=1).set_axis(['Tipo Camera', 'Prezzo'],
                                                                             axis=1).reset_index()

        address = pd.Series(np.full(len(camere), l[1]['address']))
        nome = pd.Series(np.full(len(camere), l[1]['name']))
        typeAcc = pd.Series(np.full(len(camere), typeAcc))
        checkIn = pd.Series(np.full(len(camere), checkin))
        checkOut = pd.Series(np.full(len(camere), checkout))
        rating = pd.Series(np.full(len(camere), rating))
        reviews = pd.Series(np.full(len(camere), reviews))

        featuresApp = pd.Series(np.full(len(camere), features))

        camereConPrezzi = pd.concat([nome, typeAcc, address, checkIn, checkOut, camereConPrezzi, rating, reviews, featuresApp],
                                    axis=1).set_axis(['Name', 'Type of Accomodation', 'Area1', 'Check-in', 'Check-out',
                                                      'index', 'Type', 'Price', 'Rating', 'Reviews',
                                                      'Features (appoggio)'],
                                                     axis=1)

        areaUpd = camereConPrezzi['Area1'].str.split(', ', 4, expand=True)

        if len(areaUpd.columns) == 5:
            areaUpd = areaUpd.set_axis(['Street', 'Address', 'Area', 'Postal code/City', 'Country'], axis=1)

            postalCode = areaUpd['Postal code/City'].str.split(' ', 2, expand=True)

            if len(postalCode.columns) >= 2:
                postalCode = postalCode.set_axis(['Postal Code', 'city'], axis=1)

            camereConPrezzi = pd.concat([camereConPrezzi, areaUpd['Area'], postalCode], axis=1)

            del [camereConPrezzi['index']]
            #del [camereConPrezzi['Address']]
            del [camereConPrezzi['Area1']]

        if len(areaUpd.columns) == 4:
            areaUpd = areaUpd.set_axis(['Street', 'Area', 'Postal code/City', 'Country'], axis=1)

            postalCode = areaUpd['Postal code/City'].str.split(' ', 2, expand=True)

            if len(postalCode.columns) == 2:
                postalCode = postalCode.set_axis(['Postal Code', 'city'], axis=1)

            camereConPrezzi = pd.concat([camereConPrezzi, areaUpd['Area'], postalCode], axis=1)

            del [camereConPrezzi['index']]
            del [camereConPrezzi['Area1']]

        # Mettiamo a posto le colonne, convertiamo le virgole in punti, e viceversa

        camereConPrezzi['Price'] = camereConPrezzi['Price'].str.replace(',', '')
        # camereConPrezzi['Rating'] = camereConPrezzi['Rating'].str.replace('.', ',')
        # camereConPrezzi['Reviews'] = camereConPrezzi['Reviews'].astype(str).str.replace(',', '.')

        # Creiamo le colonne con le features

        camereConPrezzi.loc[
            camereConPrezzi['Features (appoggio)'].str.contains('Fabulous breakfast'), 'Fabulous Breakfast'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('WiFi'), 'WiFi'] = int(1)
        camereConPrezzi.loc[
            camereConPrezzi['Features (appoggio)'].str.contains('24-hour front desk'), '24-hour front desk'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('Bar'), 'Bar'] = int(1)
        camereConPrezzi.loc[
            camereConPrezzi['Features (appoggio)'].str.contains('Non-smoking rooms'), 'Non-smoking rooms'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('Pets allowed'), 'Pets allowed'] = int(
            1)
        camereConPrezzi.loc[
            camereConPrezzi['Features (appoggio)'].str.contains('Very good breakfast'), 'Very good breakfast'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('Restaurant'), 'Restaurant'] = int(1)
        camereConPrezzi.loc[
            camereConPrezzi['Features (appoggio)'].str.contains('Private parking'), 'Private parking'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('Tea/coffee maker in all rooms'),
                            'Tea/coffee maker in all rooms'] = int(1)
        camereConPrezzi.loc[camereConPrezzi['Features (appoggio)'].str.contains('Room service'), 'Room service'] = int(
            1)

        del [camereConPrezzi['Features (appoggio)']]

        #camereConPrezzi.to_excel(
        #    r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBookingTest.xlsx")

    if len(l[0]) != 0:
        return camereConPrezzi


def getSingleHotelDataList(nome, checkin, checkout, adults, children, numberOfRooms, currency, features=False):

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    checkin = checkin
    checkout = checkout
    hotelName = nome
    adults = str(adults)
    children = str(children)
    numberOfRooms = str(numberOfRooms)
    currency = currency

    l = list()
    g = list()
    o = {}
    k = {}
    fac = []
    fac_arr = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    target_url = "https://www.booking.com/hotel/it/" + hotelName + ".it.html?checkin=" + checkin + "&checkout=" + checkout + "&group_adults=" + adults + "&group_children=" + children + "&no_rooms=" + numberOfRooms + "&selected_currency=" + currency

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        o["name"] = soup.find("h2", {"class": "pp-header__title"}).text
    except:
        o["name"] = None

    try:
        o["address"] = soup.find("span", {"class": "hp_address_subtitle"}).text.strip("\n")
    except:
        o["address"] = None

    try:
        o["rating"] = soup.find("div", {"class": "d10a6220b4"}).text
    except:
        o["rating"] = None

    fac = soup.find_all("div", {"class": "important_facility"})
    for i in range(0, len(fac)):
        fac_arr.append(fac[i].text.strip("\n"))
    ids = list()
    targetId = list()
    try:
        tr = soup.find_all("tr")
    except:
        tr = None
    for y in range(0, len(tr)):
        try:
            id = tr[y].get('data-block-id')
        except:
            id = None
        if (id is not None):
            ids.append(id)
    #print("Number of Features: ", len(ids))

    for i in range(0, len(ids)):
        try:
            allData = soup.find("tr", {"data-block-id": ids[i]})
            try:
                rooms = allData.find("span", {"class": "hprt-roomtype-icon-link"})
            except:
                rooms = None
            if (rooms is not None):
                last_room = rooms.text.replace("\n", "")
            try:
                k["room"] = rooms.text.replace("\n", "")
            except:
                k["room"] = last_room
            price = allData.find("div", {
                "class": "bui-price-display__value prco-text-nowrap-helper prco-inline-block-maker-helper prco-f-font-heading"})
            k["price"] = price.text.replace("\n", "")

            g.append(k)
            k = {}
        except:
            k["room"] = None
            k["price"] = None
    l.append(g)
    l.append(o)
    l.append(fac_arr)

    if features == True:
        return l[2]
    if features == False:
        return l


def getSingleHotelData(nome, checkin, checkout, adults, children, numberOfRooms, currency):

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    checkin = checkin
    checkout = checkout
    hotelName = nome
    adults = str(adults)
    children = str(children)
    numberOfRooms = str(numberOfRooms)
    currency = currency

    l = list()
    g = list()
    o = {}
    k = {}
    fac = []
    fac_arr = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    target_url = "https://www.booking.com/hotel/it/" + hotelName + ".it.html?checkin=" + checkin + "&checkout=" + checkout + "&group_adults=" + adults + "&group_children=" + children + "&no_rooms=" + numberOfRooms + "&selected_currency=" + currency

    resp = requests.get(target_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        o["name"] = soup.find("h2", {"class": "pp-header__title"}).text
    except:
        o["name"] = None

    try:
        o["address"] = soup.find("span", {"class": "hp_address_subtitle"}).text.strip("\n")
    except:
        o["address"] = None

    try:
        o["rating"] = soup.find("div", {"class": "d10a6220b4"}).text
    except:
        o["rating"] = None

    fac = soup.find_all("div", {"class": "important_facility"})
    for i in range(0, len(fac)):
        fac_arr.append(fac[i].text.strip("\n"))
    ids = list()
    targetId = list()
    try:
        tr = soup.find_all("tr")
    except:
        tr = None
    for y in range(0, len(tr)):
        try:
            id = tr[y].get('data-block-id')
        except:
            id = None
        if (id is not None):
            ids.append(id)
    print("Number of Features: ", len(ids))

    for i in range(0, len(ids)):
        try:
            allData = soup.find("tr", {"data-block-id": ids[i]})
            try:
                rooms = allData.find("span", {"class": "hprt-roomtype-icon-link"})
            except:
                rooms = None
            if (rooms is not None):
                last_room = rooms.text.replace("\n", "")
            try:
                k["room"] = rooms.text.replace("\n", "")
            except:
                k["room"] = last_room
            price = allData.find("div", {
                "class": "bui-price-display__value prco-text-nowrap-helper prco-inline-block-maker-helper prco-f-font-heading"})
            k["price"] = price.text.replace("\n", "")

            g.append(k)
            k = {}
        except:
            k["room"] = None
            k["price"] = None
    l.append(g)
    l.append(o)
    l.append(fac_arr)

    camere = list()
    prezziCamere = list()
    if len(l[0]) != 0:

        for camera in l[0]:
            camere.append(pd.Series(camera['room']))
            prezziCamere.append(pd.Series(camera['price'][2:len(camera['price'])]))

        rating = l[1]['rating']

        camere = pd.concat([series for series in camere], axis=0)
        prezziCamere = pd.concat([series for series in prezziCamere], axis=0)

        camereConPrezzi = pd.concat([camere, prezziCamere], axis=1).set_axis(['Tipo Camera', 'Prezzo'],
                                                                             axis=1).reset_index()

        address = pd.Series(np.full(len(camere), l[1]['address']))
        nome = pd.Series(np.full(len(camere), l[1]['name']))
        checkIn = pd.Series(np.full(len(camere), checkin))
        checkOut = pd.Series(np.full(len(camere), checkout))
        rating = pd.Series(np.full(len(camere), rating))

        camereConPrezzi = pd.concat([nome, address, checkIn, checkOut, camereConPrezzi, rating], axis=1).set_axis(
            ['Nome', 'Indirizzo', 'Check-in', 'Check-out',
             'index', 'Tipo Camera', 'Prezzo', 'Rating'], axis=1)
        del [camereConPrezzi['index']]

    # camereConPrezzi.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBooking.xlsx")

    if len(l[0]) != 0:
       return camereConPrezzi


def aggregaDBHotel (cameraConPrezzi):

    import pandas as pd

    # storiamo in un Excel finale che contiene tutti gli altri dati

    base = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBooking1.xlsx")

    nuovoH = cameraConPrezzi

    DBNuovo = pd.concat([base, nuovoH], axis=0)

    DBNuovo = DBNuovo.drop_duplicates(subset = ['Name', 'Area', 'Check-in', 'Check-out', 'Type', 'Price'], keep = 'last')

    DBNuovo['Reviews'] = DBNuovo['Reviews'].astype(str).str.replace(',', '.')

    # Viene salvato e aggregato solo il dato che ha almeno una feature
    DBNuovo[['Name','Type of Accomodation', 'Area', 'Postal Code', 'city', 'Check-in', 'Check-out', 'Type', 'Price',
             'Rating', 'Reviews', 'Fabulous Breakfast',
             '24-hour front desk', 'Non-smoking rooms', 'Very good breakfast', 'Restaurant', 'Private parking',
             'Tea/coffee maker in all rooms', 'Room service', 'Bar', 'WiFi', 'Pets allowed']].to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBooking1.xlsx")


def getHotelList(country, city):

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    city = city

    city = city.lower().replace(" ", "-")
    country = country

    centre = "https://www.booking.com/district/" + country + "/milan/" + city + "-centre.it.html"
    apartments = "https://www.booking.com/apartments/city/" + country + "/" + city + ".it.html"
    hostels = "https://www.booking.com/hostels/city/" + country + "/" + city + ".it.html"
    general = "https://www.booking.com/city/" + country + "/" + city + ".html"

    linkList = [general, hostels, centre, apartments]

    hotelList = list()

    for iLink in range(len(linkList)):

        resp = requests.get(linkList[iLink])
        soup = BeautifulSoup(resp.text, 'html.parser')

        a_tag = soup.findAll('a', href=True)

        fs = list()
        for i in a_tag:
            fs.append(i)

        a = pd.Series(fs).astype(str)

        a.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Prova.xlsx")

        hotelData = a[
            (a.str.contains('class="bui-card__header_full_link_wrap"') == True)].drop_duplicates().reset_index()

        hotelNames = list()
        for value in range(len(hotelData[0])):
            hotelNames.append(hotelData[0][value][59:hotelData[0][value].find('.html'):])

        hotelNames = pd.DataFrame(hotelNames)

        if iLink == 0:
            hotelNames['Accomodation'] = pd.Series(np.full(len(hotelNames), 'Hotel'))
        if iLink == 1:
            hotelNames['Accomodation'] = pd.Series(np.full(len(hotelNames), 'Hostel'))
        if iLink == 2:
            hotelNames['Accomodation'] = pd.Series(np.full(len(hotelNames), 'Hotel'))
        if iLink == 3:
            hotelNames['Accomodation'] = pd.Series(np.full(len(hotelNames), 'Apartment'))

        hotelList.append(hotelNames)

    hotelList = pd.concat([series for series in hotelList], axis=0).drop_duplicates()
    hotelList = hotelList[~hotelList[0].str.contains('=')].reset_index().dropna()
    del [hotelList['index']]

    corrected = list()
    for string in hotelList[0][hotelList[0].str.contains('.it')]:
        stringC = string.replace(string, string[0:len(string) - 3])
        corrected.append(stringC)

    finalList = pd.concat([hotelList[0][~hotelList[0].str.contains('.it')], pd.Series(corrected)], axis=0).reset_index()

    del [finalList['index']]

    finalList = pd.concat([finalList, hotelList['Accomodation']], axis=1).set_axis(['Accomodation Name',
                                                                                    'Accomodation Type'], axis=1)

    return finalList[finalList['Accomodation Name'] != 'ults']


def getHotelNames ():

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np

    html = 'https://www.booking.com/searchresults.it.html?aid=352050&label=msn-5AYr7bJaEOczZX_NT30WAg-16815849946%3Atikwd-17383121152%3Aloc-93%3Aneo%3Amte%3Alp1847%3Adec%3Aqsostello%2Bbello%2Bmilano&sid=17af2df6ace3521c63aead43b193c4d4&sb=1&sb_lp=1&src=index&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Findex.it.html%3Faid%3D352050%26label%3Dmsn-5AYr7bJaEOczZX_NT30WAg-16815849946%253Atikwd-17383121152%253Aloc-93%253Aneo%253Amte%253Alp1847%253Adec%253Aqsostello%252Bbello%252Bmilano%26sid%3D17af2df6ace3521c63aead43b193c4d4%26sb_price_type%3Dtotal%26%26&ss=Milano%2C+Lombardia%2C+Italia&is_ski_area=0&checkin_year=&checkin_month=&checkout_year=&checkout_month=&group_adults=2&group_children=0&no_rooms=1&b_h4u_keep_filters=&from_sf=1&ss_raw=milano&ac_position=0&ac_langcode=it&ac_click_type=b&ac_meta=GhA0ODY2OWRiYmE3NGQwMTU1IAAoATICaXQ6Bm1pbGFub0AASgBQAA%3D%3D&dest_id=-121726&dest_type=city&iata=MIL&place_id_lat=45.4643&place_id_lon=9.18878&search_pageview_id=48669dbba74d0155&search_selected=true&search_pageview_id=48669dbba74d0155&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0'
    resp = requests.get(html)
    soup = BeautifulSoup(resp.text, 'html.parser')

    a_tag = soup.findAll('a', href=True)

    fs = list()
    for i in a_tag:
        fs.append(i)

    a = pd.Series(fs).astype(str)

    hotelData = a[(a.str.contains('class="bui-card__header_full_link_wrap"') == True)].drop_duplicates().reset_index()

    # print(hotelData[0][0][58:len(hotelData[0][0])])

    hotelNames = list()
    for value in range(len(hotelData[0])):
        hotelNames.append(hotelData[0][value][59:hotelData[0][value].find('.it'):])

    hotelNames = pd.Series(hotelNames)
    hotelNames = hotelNames[~hotelNames.str.contains('=')]

    # a.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Prova.xlsx")

    return hotelNames


def cleanAndEncodeAccomodationDataset (df, keep_names=False, keep_type=False, remove_outliers=False, price_limit=None, exclude_luxury=False,
                                       del_rating_reviews=False, price_per_person=False):

    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    # importiamo il dataset

    df = df

    # prepariamo i dati in modo che vengano processati

    # sostituiamo gli NaN delle varie colonne Features: sono variabili Binarie, quindi dovrebbero prendere il valore di 1 e
    # 0 (se hanno NaN e 1 i modelli non vanno)

    df.loc[(df['Very good breakfast'] == 1) | (df['Fabulous Breakfast'] == 1), 'breakfast'] = 1

    # Rimuoviamo gli Outliers

    if remove_outliers == True:
       df['Z'] = stats.zscore(df['Price'])
       df = df[(df['Z'] < 3)]
       del [df['Z']]

    df = df.fillna(0)

    # Ora modifichiamo i CAP per adattarli a un indice di distanza dal centro. Assumiamo che i CAP aumentino il loro valore
    # man mano che si esce dal centro, escludendo il centro stesso (che ha 3 CAP). Non è proprio corretto, ma in assenza di
    # dati rilevanti e affidabili sulla distanza dal centro, ritengo che sia una buona approssimazione.

    df.loc[df['Area'] != 'Milan City Centre', 'Distance from Centre (index)'] = np.abs(
        df['Postal Code'] - df['Postal Code'].min())
    df.loc[df['Area'] == 'Milan City Centre', 'Distance from Centre (index)'] = 0

    # In questo modo, se un Hotel non è in centro di suo, al crescere di questo indice possiamo avere una proxy della
    # distanza dal centro (al crescere di questa, cresce la distanza dal centro, con grande approssimazione).

    # Adesso bisogna codificare il typo di sistemazione. Faremo qualche semplificazione perché ogni hotel ha diversi tipi
    # di stanza. Ci possono poi essere diversi modi fdi codificare e dare un numero ai diversi tipi di stanza. Nella nostra
    # ottica di dare il prezzo a diverse sistemazioni, la cosa migliore è raggruppare per numero di persone nella stanza.
    # È anche utile nel nostro caso distinguere per dormitorio in ostello, e in camere lusso (suite, camere premium, etc.)

    # Dormitori
    df.loc[(df['Type'].str.contains('Dormitory', case=False)) & (
        df['Type'].str.contains('8')), 'Type (encoded)'] = 1  # dormitorio da 8
    df.loc[(df['Type'].str.contains('Dormitory', case=False)) & (
        df['Type'].str.contains('6')), 'Type (encoded)'] = 2  # dormitorio da 6
    df.loc[(df['Type'].str.contains('Dormitory', case=False)) & (
        df['Type'].str.contains('5')), 'Type (encoded)'] = 3  # dormitorio da 5
    df.loc[(df['Type'].str.contains('Dormitory', case=False)) & (
        df['Type'].str.contains('4')), 'Type (encoded)'] = 4  # dormitorio da 4

    # Stanze
    df.loc[(df['Type'].str.contains('Room', case=False)) & (
        df['Type'].str.contains('single', case=False)), 'Type (encoded)'] = 9  # Camera singola
    df.loc[(df['Type'].str.contains('Room', case=False)) & (
                (df['Type'].str.contains('Double', case=False)) | (df['Type'].str.contains('twin', case=False)) |
                (df['Type'].str.contains('King', case=False)) | (
                    df['Type'].str.contains('Queen', case=False))), 'Type (encoded)'] = 8  # Camera doppia
    df.loc[(df['Type'].str.contains('Room', case=False)) & ((df['Type'].str.contains('Triple', case=False)) |
                                                            (df['Type'].str.contains(
                                                                '3'))), 'Type (encoded)'] = 7  # Camera tripla
    df.loc[((df['Type'].str.contains('Room', case=False)) | (df['Type'].str.contains('Family', case=False))) &
           ((df['Type'].str.contains('Quadruple', case=False)) | (
               df['Type'].str.contains('Family', case=False))), 'Type (encoded)'] = 6  # Camera da 4 o family
    df.loc[(df['Type'].str.contains('Room', case=False)) & (
        df['Type'].str.contains('Sextuple', case=False)), 'Type (encoded)'] = 5  # Camera da 6

    # Appartamenti
    df.loc[(df['Type'].str.contains('Apartment', case=False)) | (df['Type'].str.contains('Loft', case=False)) |
           (df['Type'].str.contains('Studio', case=False)), 'Type (encoded)'] = 10  # Appartamento

    # Premium
    df.loc[(df['Type'].str.contains('Deluxe', case=False)) | (df['Type'].str.contains('Suite', case=False)) |
           (df['Type'].str.contains('Collection', case=False)), 'Type (encoded)'] = 11  # Accomodation di classe Premium

    if price_limit != None:
         df = df[df['Price'] < price_limit]

    if exclude_luxury == True:
        df = df[df['Type (encoded)'] != 11]

    df = df.dropna(subset='Type (encoded)')

    # PREZZO A PERSONA

    if price_per_person == True:
        df.loc[df['Type (encoded)'] == 8, 'Price'] = df['Price'] / 2
        df.loc[df['Type (encoded)'] == 7, 'Price'] = df['Price'] / 3
        df.loc[df['Type (encoded)'] == 6, 'Price'] = df['Price'] / 4
        df.loc[df['Type (encoded)'] == 5, 'Price'] = df['Price'] / 6

    # df.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBooking_encoded.xlsx")

    print('\n')
    print('Sample Size:', len(df['Type']))
    print('Average Price per night on the sample:', df['Price'].mean())
    print('\n')

    # Check veloce della distribuzione delle classi di accomodation

    plt.figure(figsize=(11, 5))
    plt.hist(df['Price'], bins=50)
    # plt.show()

    # Ribadiamo che ci interessa prevedere il prezzo per diversi tipi di accomodation, secondo certe caratteristiche della
    # struttura, date dalle features. Quindi la nostra dipendente sarà la colonna 'Price' mentre le indipendenti
    # saranno tutte le altre, ad eccezione del nome dell'Hotel e del tipo di stanza (che sono variabili non numeriche) e che
    # per ora non ci serve analizzare. Quindi puliamo il DB da queste colonne, e da quelle relative al CAP, che abbiamo
    # codificato come distanza dal centro

    if keep_names == False:
        del [df['Name']]
    del [df['Type of Accomodation']]
    del [df['Area']]
    del [df['Postal Code']]
    del [df['city']]
    del [df['Check-in']]
    del [df['Check-out']]
    if keep_type == False:
        del [df['Type']]
    del [df['Unnamed: 0']]
    del [df['Fabulous Breakfast']]
    del [df['Very good breakfast']]

    # Non possiamo fare assunzioni sul rating o sul numero di reviews perché non sarebbe preciso

    if del_rating_reviews == True:
        del[df['Rating']]
        del[df['Reviews']]

    # Salviamo per memoria

    df.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\CamereMilanoBooking_encoded.xlsx")

    return df


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

    df['rating'] = df['rating'].str.replace('.', ',')
    df['Reviews (number)'] = df['Reviews (number)'].str.replace(',', '')

    return df


def getExperienceList (granularity):

    import pandas as pd
    import numpy as np

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

            checkIn = '2023-08-04'
            checkOut = '2023-08-10'

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

    finalDF.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ExperienceDBTot.xlsx")

    return finalDF


def createTrainDatasetAirBnB (dataset, granuarity):

    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from scipy import stats

    df = dataset

    df = df.dropna(subset=['Reviews (number)']).reset_index()
    del [df['index']]

    # Eliminiamo gli outliers sul prezzo e sul numero di reviews

    df['Z Rev'] = np.abs(stats.zscore(df['Reviews (number)']))
    df = df[df['Z Rev'] < 3].reset_index()
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

    df['Price_C'] = (1 / df['Price'])

    dfCluster = df[['Price_C', 'rating', 'Reviews (number)']]

    clModel = pd.DataFrame(KMeans(n_clusters=granuarity).fit_predict(dfCluster)).set_axis(['Ranking'], axis=1)

    dfBase = pd.concat([df, clModel], axis=1)

    # Isoliamo la classe delle migliori scelte

    winCluster = \
    dfBase['Ranking'][dfBase['Reviews (number)'] == dfBase['Reviews (number)'].max()].reset_index()['Ranking'][0]

    dfBase.loc[dfBase['Ranking'] == winCluster, 'Good Choice'] = 1

    dfBase['Good Choice'] = dfBase['Good Choice'].fillna(0)

    # ELiminiamo le colonne superflue

    del [dfBase['Ranking']]
    del [dfBase['Price_C']]

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

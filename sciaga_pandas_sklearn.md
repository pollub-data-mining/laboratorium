# Ściąga: pandas i scikit-learn na laboratorium

Każdy wpis: co robi, potem przykład. Nazwy zmiennych jak w notatniku. Wpisy są pogrupowane według zadań.

### Słowniczek

| Termin | Co znaczy |
| --- | --- |
| odsetek odejść (churn rate) | jaka część klientów odeszła, ułamek od 0 do 1 |
| zbiór uczący i testowy (train, test) | na uczącym model się uczy, na testowym sprawdzamy go na klientach, których nie widział |
| segment (cluster) | grupa podobnych klientów znaleziona przez K-means |
| inercja (inertia) | suma kwadratów odległości klientów od środków segmentów; im mniej, tym ciaśniej |
| silhouette | od -1 do 1; im wyżej, tym lepiej rozdzielone segmenty |
| dokładność (accuracy) | jaka część wszystkich przewidywań jest trafna |
| precision | z klientów, do których dzwonimy, jaka część naprawdę odchodzi |
| recall | z klientów, którzy naprawdę odchodzą, do jakiej części dzwonimy |
| TP, FP | oflagowany i naprawdę odchodzi; oflagowany, a i tak by został |
| próg (threshold) | od jakiego prawdopodobieństwa odejścia flagujemy klienta |

### Zadanie 1: eksploracja

Wczytanie CSV z adresu URL albo pliku.
`df = pd.read_csv(DATA_URL)`

Rozmiar tabeli (wiersze, kolumny) i pierwsze wiersze.
`df.shape; df.head()`

Udział każdej wartości w kolumnie (ułamki zamiast liczebności).
`df["Contract"].value_counts(normalize=True)`

Odsetek odejść: średnia z kolumny prawda/fałsz.
`(df["Churn"] == "Yes").mean()`

Odsetek odejść w każdym rodzaju umowy.
`(df["Churn"] == "Yes").groupby(df["Contract"]).mean()`

Sortowanie wyniku malejąco.
`wynik.sort_values(ascending=False)`

### Zadanie 2: przygotowanie danych

Tekst na liczby; czego nie da się zamienić, staje się NaN.
`df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")`

Wiersze z brakiem w kolumnie.
`df_clean[df_clean["TotalCharges"].isna()]`

Usunięcie wierszy z brakiem w wybranej kolumnie.
`df_clean = df_clean.dropna(subset=["TotalCharges"])`

Usunięcie kolumny.
`df_clean = df_clean.drop(columns="customerID")`

Zamiana wartości według słownika.
`df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})`

Kolumny kategoryczne na kolumny 0/1, bez pierwszej kategorii.
`df_model = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True, dtype=int)`

Standaryzacja: średnia 0, odchylenie 1 w każdej kolumnie.
`X_cluster = StandardScaler().fit_transform(df_clean[["tenure", "MonthlyCharges", "TotalCharges"]])`

### Zadanie 3: segmentacja

K-means z ustalonym ziarnem; `fit_predict` zwraca numer segmentu dla każdego wiersza.
`kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10); labels = kmeans_model.fit_predict(X_cluster)`

Inercja modelu po nauce.
`kmeans_model.inertia_`

Silhouette dla danych i etykiet.
`silhouette_score(X_cluster, labels)`

Dopisanie elementu do listy w pętli.
`inertias.append(kmeans_model.inertia_)`

Średnie kilku kolumn w grupach.
`df_clean.groupby("segment")[["tenure", "MonthlyCharges"]].mean()`

Liczba klientów w każdym segmencie, w kolejności segmentów.
`df_clean["segment"].value_counts().sort_index()`

### Zadanie 4: klasyfikacja i ewaluacja

Podział na zbiór uczący i testowy z zachowaniem proporcji klas.
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`

Dokładność modelu, który zawsze mówi „zostaje”: udział klasy 0.
`(y_test == 0).mean()`

Drzewo decyzyjne: utworzenie i nauka.
`tree_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)`

Drzewo, w którym błędy na rzadszej klasie ważą więcej.
`DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")`

Dokładność na zbiorze testowym.
`accuracy_score(y_test, tree_model.predict(X_test))`

Macierz pomyłek jako wykres.
`ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["zostaje", "odchodzi"])`

Precision, recall i F1 dla każdej klasy.
`print(classification_report(y_test, y_pred, digits=3))`

Ważność cech, posortowana malejąco.
`pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(ascending=False)`

Rysunek dwóch pierwszych poziomów drzewa.
`plot_tree(tree_model, max_depth=2, feature_names=list(X.columns), filled=True, fontsize=14)`

### Zadanie 5 i 6: próg i zysk

Prawdopodobieństwo klasy 1 (odejście) dla każdego klienta.
`proba = tree_model.predict_proba(X_test)[:, 1]`

Decyzja według własnego progu zamiast 0,5.
`flagged = proba >= 0.3`

Liczba klientów spełniających oba warunki naraz.
`int((flagged & (y_true == 1)).sum())`

Lista wyników dla każdego progu.
`[profit(y_test, proba, t) for t in thresholds]`

Pozycja największej wartości na liście.
`int(np.argmax(profits_plain))`

### Gdy coś nie działa

`NameError`: nie uruchomiono wcześniejszej komórki. Środowisko wykonawcze > Uruchom poprzednie (Ctrl+F8).

`KeyError` z nazwą kolumny: literówka w nazwie, wielkość liter ma znaczenie.

`AssertionError` w samokontroli: komunikat pod spodem mówi, co poprawić.

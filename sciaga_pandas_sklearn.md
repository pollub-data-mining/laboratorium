# Ściąga: pandas i scikit-learn na laboratorium

Każdy wpis: co robi, potem przykład. Nazwy zmiennych jak w notatniku.

Wczytanie CSV z adresu URL albo pliku.
`df = pd.read_csv(DATA_URL)`

Rozmiar tabeli (wiersze, kolumny) i pierwsze wiersze.
`df.shape; df.head()`

Udział każdej wartości w kolumnie (ułamki zamiast liczebności).
`df["Contract"].value_counts(normalize=True)`

Średnia w grupach: odsetek odejść w każdym rodzaju umowy.
`(df["Churn"] == "Yes").groupby(df["Contract"]).mean()`

Średnie kilku kolumn w grupach.
`df_clean.groupby("segment")[["tenure", "MonthlyCharges"]].mean()`

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

K-means z ustalonym ziarnem; `fit_predict` zwraca numer segmentu dla każdego wiersza.
`kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10); labels = kmeans_model.fit_predict(X_cluster)`

Inercja: suma kwadratów odległości od środków segmentów (mniej = ciaśniej).
`kmeans_model.inertia_`

Silhouette: od -1 do 1, im wyżej, tym lepiej rozdzielone segmenty.
`silhouette_score(X_cluster, labels)`

Podział na zbiór uczący i testowy z zachowaniem proporcji klas.
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`

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

Prawdopodobieństwo klasy 1 (odejście) dla każdego klienta.
`proba = tree_model.predict_proba(X_test)[:, 1]`

Decyzja według własnego progu zamiast 0,5.
`flagged = proba >= 0.3`

Ważność cech, posortowana malejąco.
`pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(ascending=False)`

Rysunek dwóch pierwszych poziomów drzewa.
`plot_tree(tree_model, max_depth=2, feature_names=list(X.columns), filled=True, fontsize=14)`

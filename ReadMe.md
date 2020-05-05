# Recommender

    ## Was wird gemacht
    Es wird random eine Datensatz generiert, der eine fiktive User-Item Interaktion beschreibt. 
    Der Datensatz zeigt, mit welchen Item ein User interagiert hat und wie er dieses bewertet hat. 
    Ziel ist es, einem User weitere Items zu empfehlen, mit denen er bisher noch nicht interagiert hat. 
    Dazu werden 3 verschiedene Ansätze ausprobiert: 
        1. Content-Based Method
        2. Collaborativ Filtering - Memory based 
        3. Collaborativ Filtering - Model based

    1. Content Based: 
        - Jedes Item werden bestimmte Features zugeordnet (z.B. Genres zu Filmen)
        - Auf Basis der bisher Ratings eines Items von einem User, können die Features der Items gewichtet werden. 
        - Empfohlen werden dann Items, bei denen die zugehörigen Features am stärksten gewichtet sind
  
    2. Collaborative Filtering - Memory based
        - auf Basis von Ähnlichkeiten zwischen Usern/Items werden Empfehlungen gegeben.
        - Die Ähnlichkeit wird über Cosine-Distance bestimmt 
  
    3. Collaborative Filtering - Model Based 
        - Matrix Factorisation 
        - Die normale User-Item-Interaktion Matrix wird über Singular Value Decompasation durch 3 kleinere Matrizen beschrieben. 
        - Dadurch können alle fehlenden Werte predicted werden
  
  ## How to Start
    1. Führe main.py aus 
    2. Zur Evaluation per Mean Square Error führe Evaluation.py aus

    weiterführende Links: 
    https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
    https://realpython.com/build-recommendation-engine-collaborative-filtering/
    https://beckernick.github.io/matrix-factorization-recommender/
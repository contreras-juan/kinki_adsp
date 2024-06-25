import numpy as np
from sklearn.preprocessing import LabelEncoder
from annoy import AnnoyIndex
from joblib import Parallel, delayed
import pandas as pd

# Generar datos categóricos de ejemplo
np.random.seed(42)
categories = list('abcde')
X = np.random.choice(categories, size=(90_000, 27)).astype(object)

# Introducir valores faltantes aleatoriamente
missing_rate = 0.1
mask = np.random.rand(*X.shape) < missing_rate
X[mask] = np.nan

# Codificar las variables categóricas a numéricas usando LabelEncoder para cada columna
label_encoders = []
X_encoded = np.empty_like(X, dtype=float)

for i in range(X.shape[1]):
    le = LabelEncoder()
    col = X[:, i]
    mask = ~pd.isnull(col)
    X_encoded[mask, i] = le.fit_transform(col[mask])
    X_encoded[~mask, i] = np.nan
    label_encoders.append(le)

# Aproximar vecinos más cercanos usando Annoy
def annoy_knn_impute(X_encoded, n_neighbors=2, n_trees=10):
    f = X_encoded.shape[1]
    t = AnnoyIndex(f, 'hamming')
    for i, row in enumerate(np.nan_to_num(X_encoded)):
        t.add_item(i, row)
    t.build(n_trees)
    
    neighbors = []
    for i, row in enumerate(np.nan_to_num(X_encoded)):
        indices = t.get_nns_by_item(i, n_neighbors)
        neighbors.append(indices)
    
    return neighbors

# Obtener los vecinos más cercanos
neighbors = annoy_knn_impute(X_encoded, n_neighbors=5, n_trees=10)

# Función para imputar valores usando los vecinos más cercanos
def impute_row(i, X_encoded, neighbors):
    row = np.copy(X_encoded[i, :])
    for j in range(row.shape[0]):
        if np.isnan(row[j]):
            # Obtener los vecinos válidos
            valid_neighbors = [X_encoded[neighbor, j] for neighbor in neighbors[i] if not np.isnan(X_encoded[neighbor, j])]
            if valid_neighbors:
                row[j] = np.mean(valid_neighbors)
    return row

# Imputar los valores en paralelo
X_imputed = Parallel(n_jobs=-1)(delayed(impute_row)(i, X_encoded, neighbors) for i in range(X_encoded.shape[0]))
X_imputed = np.array(X_imputed)

# Decodificar los valores imputados de vuelta a las categorías originales usando LabelEncoder para cada columna
X_imputed_decoded = np.empty_like(X, dtype=object)

for i in range(X.shape[1]):
    le = label_encoders[i]
    col = X_imputed[:, i]
    mask = ~np.isnan(col)
    X_imputed_decoded[mask, i] = le.inverse_transform(col[mask].astype(int))
    X_imputed_decoded[~mask, i] = np.nan

# Mostrar una parte del resultado imputado
print(X_imputed_decoded)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

# Casos de uso 
sentences = [
    "Browse the Product List",
    "Browse the Catalog",
    "Searching the Catalog",
    "Browse the items List",
    "Add cart Items",
    "Update items in the Cart",
    "Remove items from cart",
    "View Cart Items",
    "Make payment",
    "List Order Items",
    "View Order Status",
    "Confirm Order",
    "Get Total amount",
    "Change Shipping Info ",
    "Signing Up",
    "Signing In",
    "Signout",
    "Manage Item",
    "Manage Account",
    "Manage Order",
    "Manage product/category"
]

# Etiquetas reales (Gold Standard) según AcmeAir 
# 0: Auth, 1: Booking, 2: Customer, 3: Flight, 4: Main
gold_labels = [
    4, 4, 4, 4,   # catalogo
    2, 2, 2, 2,   # carrito (customer)
    1, 1, 1, 1,   # booking
    1,           # booking
    3,           # vuelo
    0, 0, 0,     # autenticación
    2, 2, 1, 4   # gestión y main
]

# Funcion para evaluar un modelo dado 
def evaluate_model(model_name, model_id, sentences, gold_labels, n_clusters=5):
    print(f"\nEvaluando modelo: {model_name}")
    model = SentenceTransformer(model_id)
    embeddings = model.encode(sentences)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Alinear etiquetas del clustering con el gold standard
    conf_matrix = confusion_matrix(gold_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_preds = [mapping[p] for p in predicted_labels]

    # Calcular metricas
    accuracy = accuracy_score(gold_labels, mapped_preds)
    precision = precision_score(gold_labels, mapped_preds, average='macro')
    recall = recall_score(gold_labels, mapped_preds, average='macro')
    f1 = f1_score(gold_labels, mapped_preds, average='macro')

    return {
        'Modelo': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

# Modelos a comparar
models = {
    "SBERT (bert-base-nli-mean-tokens)": "bert-base-nli-mean-tokens",
    "MPNet (all-mpnet-base-v2)": "all-mpnet-base-v2",
    "MiniLM (all-MiniLM-L6-v2)": "all-MiniLM-L6-v2",
    "MiniLM QA (multi-qa-MiniLM-L6-cos-v1)": "multi-qa-MiniLM-L6-cos-v1",
    "Paraphrase-MiniLM (paraphrase-MiniLM-L12-v2)": "paraphrase-MiniLM-L12-v2"
}

# Ejecutar evaluacion para todos los modelos
results = []
for name, model_id in models.items():
    result = evaluate_model(name, model_id, sentences, gold_labels, n_clusters=5)
    results.append(result)

# Mostrar resultados en una tabla ordenada por F1-Score
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
print("\nComparación final de modelos por desempeño:\n")
print(df_results)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering

# 1. Cargar modelo MPNet
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Endpoints (vista OpenAPI)
endpoints = [
    "GET /products", "GET /catalog", "GET /search?q=item", "GET /items",
    "POST /cart/items", "PUT /cart/items", "DELETE /cart/items/{id}", "GET /cart",
    "POST /payment", "GET /orders", "GET /orders/status", "POST /orders/confirm",
    "GET /orders/total", "PUT /shipping/info", "POST /signup", "POST /signin", "POST /signout",
    "PUT /items/manage", "PUT /account/manage", "PUT /order/manage", "PUT /product/manage"
]
endpoint_embeddings = model.encode(endpoints)
print("Endpoints Endoded!")

# 3. Casos de uso
sentences = [
    "Browse the Product List", "Browse the Catalog", "Searching the Catalog", "Browse the items List",
    "Add cart Items", "Update items in the Cart", "Remove items from cart", "View Cart Items",
    "Make payment", "List Order Items", "View Order Status", "Confirm Order", "Get Total amount",
    "Change Shipping Info ", "Signing Up", "Signing In", "Signout", "Manage Item", "Manage Account",
    "Manage Order", "Manage product/category"
]
usecase_embeddings = model.encode(sentences)
print("Use Cases Endoded!")

# 4. Normalización para similitud basada en distancia euclidiana
X1 = normalize(endpoint_embeddings)
X2 = normalize(usecase_embeddings)

print("Normalize!")

# 5. Generar matriz de similitud dispersa para cada vista
def build_similarity_matrix(X, k=5):
    n = X.shape[0]
    D = euclidean_distances(X, X, squared=True)
    S = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(D[i])[1:k+1]
        di_k1 = D[i][idx[-1]]
        dih_sum = np.sum(D[i][idx])
        for j in idx:
            S[i, j] = (di_k1 - D[i][j]) / (k * di_k1 - dih_sum)
    return S

# 6. Similitud dispersa de cada vista (endpoint y casos de uso)
S_endpoints = build_similarity_matrix(X1, k=5)
S_usecases = build_similarity_matrix(X2, k=5)


# 7. Función para calcular matriz de afinidad unificada U y pesos autoajustables
def fuse_views(S_list, max_iter=10, tol=1e-4):
    n = S_list[0].shape[0]
    m = len(S_list)
    
    # Inicializar matriz U como promedio simple
    U = np.mean(S_list, axis=0)
    
    for iteration in range(max_iter):
        # Actualizar pesos wv
        weights = []
        for Sv in S_list:
            diff = norm(U - Sv, ord=1)
            wv = 1 / (2 * np.sqrt(diff) + 1e-10)
            weights.append(wv)
        weights = np.array(weights)
        weights /= np.sum(weights)  # Normalizar

        # Actualizar U (matriz de afinidad unificada)
        U_new = np.zeros((n, n))
        for wv, Sv in zip(weights, S_list):
            U_new += wv * Sv

        # Verificar convergencia
        if np.linalg.norm(U - U_new) < tol:
            break
        U = U_new
    
    return U, weights

# 8. Fusión de vistas (S_usecases y S_endpoints)
S_list = [S_usecases, S_endpoints]
U, learned_weights = fuse_views(S_list, max_iter=20)

print("Pesos aprendidos por vista:")
for i, w in enumerate(learned_weights):
    print(f"Vista {i+1}: w = {w:.4f}")

# 9. Aplicar clustering espectral sobre U
n_clusters = 5  # Según el gold standard
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
labels = clustering.fit_predict(U)

print("\nEtiquetas predichas:")
print(labels)

import numpy as np
from collections import defaultdict

# 1. Modularidad estructural (SM)
def structural_modularity(adj_matrix, labels):
    K = len(set(labels))
    sigma = np.zeros((K, K))
    mu = np.zeros(K)
    m = np.zeros(K)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if adj_matrix[i, j]:
                if labels[i] == labels[j]:
                    mu[labels[i]] += 1
                else:
                    sigma[labels[i], labels[j]] += 1
        m[labels[i]] += 1

    mu = mu / 2  # Aristas internas contadas doble
    term1 = np.sum(mu / m)
    term2 = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                term2 += sigma[i, j] / (m[i] * m[j])
    sm = (term1 - (term2 / (K * (K - 1) / 2))) / K
    return round(sm, 4)

# 2. Porcentaje de llamadas internas (ICP)
def internal_call_percentage(adj_matrix, labels):
    total_calls = 0
    external_calls = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if adj_matrix[i, j]:
                total_calls += 1
                if labels[i] != labels[j]:
                    external_calls += 1
    if total_calls == 0:
        return 1.0
    return round(1 - external_calls / total_calls, 4)

# 3. Número de interfaces externas (IFN)
def num_interfaces(adj_matrix, labels):
    interfaces_per_service = defaultdict(set)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if adj_matrix[i, j] and labels[i] != labels[j]:
                interfaces_per_service[labels[i]].add(labels[j])
    total_interfaces = sum(len(v) for v in interfaces_per_service.values())
    return round(total_interfaces, 4)

# 4. Distribución no extrema (NED)
def non_extreme_distribution(labels):
    K = len(set(labels))
    ni = np.zeros(K)
    for k in range(K):
        count = np.sum(labels == k)
        if 5 <= count <= 20:
            ni[k] = 1
    return round(np.sum(ni) / K, 4)


adj_matrix = (U > 0.3).astype(int)  # umbral de similitud para definir conexión


# Etiquetas simuladas de microservicios (5 clusters)
labels = np.array([
    0, 0, 0, 0,      # Microservicio 0
    1, 1, 1, 1,      # Microservicio 1
    2, 2, 2, 2,      # Microservicio 2
    3, 3, 3,         # Microservicio 3
    4, 4, 4, 4, 4    # Microservicio 4
])


# Aplicar las métricas
sm = structural_modularity(adj_matrix, labels)
icp = internal_call_percentage(adj_matrix, labels)
ifn = num_interfaces(adj_matrix, labels)
ned = non_extreme_distribution(labels)

# Mostrar resultados
print("Resultados de Métricas:")
print(f"SM (Modularidad Estructural): {sm}")
print(f"ICP (Porcentaje de llamadas internas): {icp}")
print(f"IFN (Número de interfaces): {ifn}")
print(f"NED (Distribución no extrema): {ned}")


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configuración común para ambos heatmaps
def plot_similarity_matrix(matrix, title, labels):
    plt.figure(figsize=(12, 10))
    
    # Crear un colormap personalizado
    colors = ["white", "lightyellow", "gold", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Ordenar por clusters para mejor visualización
    sorted_indices = np.argsort(labels)
    sorted_matrix = matrix[sorted_indices][:, sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Crear el heatmap
    ax = sns.heatmap(
        sorted_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=np.array(endpoints)[sorted_indices] if "Endpoints" in title else np.array(sentences)[sorted_indices],
        yticklabels=np.array(endpoints)[sorted_indices] if "Endpoints" in title else np.array(sentences)[sorted_indices]
    )
    
    # Añadir líneas para separar clusters
    unique_labels = np.unique(sorted_labels)
    for label in unique_labels[:-1]:
        pos = np.where(sorted_labels == label)[0][-1] + 0.5
        ax.axhline(pos, color='black', linewidth=1)
        ax.axvline(pos, color='black', linewidth=1)
    
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

# Graficar matrices
plot_similarity_matrix(S_endpoints, "Matriz de Similitud - Endpoints (Ordenada por Clusters)", labels)
plot_similarity_matrix(S_usecases, "Matriz de Similitud - Casos de Uso (Ordenada por Clusters)", labels)
plot_similarity_matrix(U, "Matriz de Afinidad Unificada (Ordenada por Clusters)", labels)


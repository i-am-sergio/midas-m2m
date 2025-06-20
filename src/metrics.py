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



# Test
np.random.seed(42)
n_nodes = 21
adj_matrix = np.random.randint(0, 2, size=(n_nodes, n_nodes))
np.fill_diagonal(adj_matrix, 0)
adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

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

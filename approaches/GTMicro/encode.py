# --- Instalación de la librería ---
# Ejecuta esta celda para instalar Sentence-Transformers
# !pip install sentence-transformers

# --- Importaciones necesarias ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Importamos numpy para trabajar mejor con los embeddings

# --- Datos de entrada (tus oraciones/casos de uso) ---
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

# --- Carga del modelo SBERT ---
# Se utiliza 'bert-base-nli-mean-tokens' para obtener embeddings de oraciones.
# Puedes probar otros modelos si es necesario (ej. 'all-MiniLM-L6-v2' para más rapidez).
print("Cargando el modelo SBERT...")
model = SentenceTransformer('bert-base-nli-mean-tokens')
print("Modelo cargado exitosamente.")

# --- Generación de embeddings para cada oración ---
print("\nGenerando embeddings para las oraciones...")
sentence_embeddings = model.encode(sentences)
print(f"Dimensiones de los embeddings: {sentence_embeddings.shape}")
# Esto imprimirá (número_de_oraciones, tamaño_del_embedding), por ejemplo (21, 768)

# --- Cálculo y visualización de la similaridad coseno ---
print("\nCalculando y mostrando similaridad coseno para cada oración:")

# Diccionario para almacenar los resultados: {oracion_base: {oracion_comparada: similaridad}}
all_similarities = {}

for x in range(len(sentences)):
    current_sentence = sentences[x]
    # Calcula la similaridad de la oración actual con todas las demás (incluida ella misma)
    similarities = cosine_similarity(
        [sentence_embeddings[x]], # Se pasa como lista para mantener las dimensiones correctas
        sentence_embeddings
    )[0] # Tomamos el primer (y único) array de resultados

    # Almacena las similaridades para la oración actual
    sentence_similarity_dict = {}
    for y in range(len(sentences)):
        sentence_similarity_dict[sentences[y]] = similarities[y]
    all_similarities[current_sentence] = sentence_similarity_dict

    # Imprime las similaridades ordenadas para la oración actual
    print(f"\n--- Similaridad para la oración: '{current_sentence}' ---")
    # Ordena el diccionario por similaridad de forma descendente
    sorted_similarities = sorted(sentence_similarity_dict.items(), key=lambda item: item[1], reverse=True)

    # Imprime solo los resultados más relevantes (ej. similaridad > 0.5 o top N)
    # Aquí imprimimos todas para replicar tu salida, pero puedes filtrar si la lista es muy larga
    for sentence_compared, score in sorted_similarities:
        # Formateamos el score a 6 decimales para una salida limpia
        print(f"  - '{sentence_compared}': {score:.6f}")

print("\nProceso de cálculo de similaridades completado.")
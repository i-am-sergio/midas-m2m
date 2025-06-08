import json
import nltk
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from functools import reduce
from topic import handle_topic_similarity
from response import calc_response_similarity

import warnings

warnings.filterwarnings('ignore')

# file = "/Kanban.json"
# file = "/MoneyTransfer.json"
# file = "/MicroservicesEventSourcing.json"
# file = "/PiggyMetrics.json"
# file = "/SockShop.json"
# file = "/SockShop1.json"
# file = "/Sitewhere.json"
# file = "/Petclinic_single.json"
file = "/Petclinic.json"
# file = "/CargoTracking.json"


def get_clusters_num(apis):
    # keys = reduce(lambda x, y: x + y["keys"], apis, [])
    keys = []
    for api in apis:
        for api_key in api["keys"]:
            filter_keys = list(filter(lambda key: key["name"] == api_key, keys))
            has_key = len(filter_keys) != 0
            if has_key:
                filter_keys[0]["counter"] += 1
            else:
                keys.append({"name": api_key, "counter": 1})
    #  and key["counter"] != 1
    #  Para estimar el número máximo de clústeres.
    return len(list(filter(
        lambda key:key["counter"]!=len(apis) and key["counter"] != len(apis) - 1 and key["counter"] != 1, keys)))


# 返回 [{"url": url, "verb": "get", response: "xxx"  "keys": url.split(" "), "words": words.split(" ")}]
def get_apis(path):
    # with open("./Blog.json") as blogFile:
    with open(path, encoding="utf-8") as apiFile:
        api_doc = json.load(apiFile)
    apis = []
    for k, v in api_doc["paths"].items():
        for k1, v1 in v.items():
            keywords = list(filter(lambda w: w != "", k.split("/")))
            words = " ".join(keywords) + " " + v1["description"]
            apis.append({"url": k1 + " " + k, "verb": k1, "keys": filter_params(keywords), "words": words.split(" "),
                         "responses": v1['responses']})
    return (apis, api_doc["definitions"])


# 过滤参数
def filter_params(words):
    return list(filter(lambda w: not w.startswith("{"), words))


def print_line(list):
    for item in list:
        print(item)


# 获取邻接矩阵
def set_adjacency_matrix(similarities, nums):
    adj_mat = [[]]
    i = 0
    for similarity in similarities:
        if i == nums:
            adj_mat.append([])
            i = 0
        adj_mat[len(adj_mat) - 1].append(similarity["similarity"])
        i += 1
    return adj_mat


# 计算两个 api 之间的整体相似度
def clac_api_similarity(topic_similarity, res_similarity):
    if 'delete' in res_similarity["source"] and 'delete' in res_similarity["target"]:
        return topic_similarity["similarity"]
    factor = 1.0
    return factor * topic_similarity["similarity"] + (1 - factor) * res_similarity["similarity"]


# 计算所有 API 两两之间的整体相似度
def clac_api_similaritis(topic_similarities, response_similarities):
    api_similarity = []
    for topic_similarity in topic_similarities:
        for res_similarity in response_similarities:
            if topic_similarity["source"] == res_similarity["source"] and topic_similarity["target"] == \
                    res_similarity["target"]:
                api_similarity.append({"source": topic_similarity["source"], "target": topic_similarity["target"],
                                       "similarity": clac_api_similarity(topic_similarity, res_similarity)})
                continue
    return api_similarity


def transfer_microservice(ms_list, apis):
    ms = {}
    for index, ms_index in enumerate(ms_list):
        if "ms_" + str(ms_index) in ms:
            ms["ms_" + str(ms_index)].append(apis[index]["url"])
        else:
            ms["ms_" + str(ms_index)] = [apis[index]["url"]]
    return ms


def main():
    # Obtiene las APIs y las definiciones de esquemas del archivo especificado
    (apis, api_definitions) = get_apis("./specifications_dataset" + file)

    print_line(apis)
    # 计算最大聚类数量
    print("**** 计算最大聚类数量")
    print("**** Calculando el número máximo de clústeres")
    max_clusters_num = get_clusters_num(apis)
    print(max_clusters_num)
    print("**** 计算主题相似度")
    print("**** Calculando la similaridad de tópicos")
    topic_similarity = handle_topic_similarity(apis)
    print(topic_similarity[0:8])
    print("**** 计算响应消息相似度")
    print("**** Calculando la similaridad del mensaje de respuesta")
    response_similarity = calc_response_similarity(apis, api_definitions)
    print(response_similarity[0:8])
    print("**** 计算综合相似度")
    print("**** Calculando la similaridad combinada")
    api_similarity = clac_api_similaritis(topic_similarity, response_similarity)
    print(api_similarity[0:8])
    print("**** 计算 matrics")
    print("**** Calculando matrices")
    api_similarity_adj_mat = set_adjacency_matrix(api_similarity, len(apis))

    print("\n**** Matriz de Adyacencia de Similaridad de API (formato 6 decimales) ****")
    for row_index in range(len(api_similarity_adj_mat)):
        formatted_row = [f"{value: .6f}" for value in api_similarity_adj_mat[row_index]]
        print(" ".join(formatted_row))
    print("------------------------------------------------------------------")

    # 使用评分方式选出推荐微服务
    print("**** 使用评分方式选出推荐微服务")
    print("**** Seleccionando microservicios recomendados usando un enfoque de puntuación")
    recommend_microservices = []
    
    for k in range(2, max_clusters_num + 1):
        sc = SpectralClustering(n_clusters=k, affinity='precomputed')
        sc.fit(api_similarity_adj_mat)
        # print(sc.labels_)
        print(sc.affinity_matrix_)
        # metrics.calinski_harabaz_score(api_similarity_adj_mat, sc.labels_) 是用来计算当前的划分出来的图的分值
        # 计算 Calinski 和 Harabaz 得分。方差比标准。被定义为群内分散和群集间分散之间的比率。
        recommend_microservices.append(
            (metrics.calinski_harabasz_score(api_similarity_adj_mat, sc.labels_), sc.labels_.tolist()))

    print_line(recommend_microservices)
    # 找出评分最高的（最适合）的微服务
    print("**** 找出评分最高的（最适合）的微服务")
    print("**** Encontrando el microservicio con la puntuación más alta (el más adecuado)")
    best = (0, 0)
    for microservice in recommend_microservices:
        print("Puntuacion Actual: ",best[0], " - Puntuacion Nueva: " , microservice[0])
        if best[0] < microservice[0]:
            best = microservice

    print("**** 存储微服务")
    print("**** Guardando los microservicios")
    result = transfer_microservice(best[1], apis)

    with open("./result/" + file, "w") as f:
        json.dump(result, f)
    
     # print in json format by console
    print("**** Resultados")
    print(json.dumps(result, indent=2))



if __name__ == '__main__':
    main()

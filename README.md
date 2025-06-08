# MIDAS: MIDAS: A Multiview Graph-Based Approach for Automatic Microservice Extraction Enhanced by Domain Knowledge Using SBERT and Self-Weighted Clustering

## Prerrequisitos
- **Environment:** SO Linux/Debian 13
```sh
# neofetch
       _,met$$$$$gg.          heros@debian 
    ,g$$$$$$$$$$$$$$$P.       ------------ 
  ,g$$P"     """Y$$.".        OS: Debian GNU/Linux 13 (trixie) x86_64 
 ,$$P'              `$$$.     Host: HP Laptop 15-gw0xxx 
',$$P       ,ggs.     `$$b:   Kernel: 6.12.30-amd64 
`d$$'     ,$P"'   .    $$$    Uptime: 22 hours, 57 mins 
 $$P      d$'     ,    $$P    Packages: 2309 (dpkg) 
 $$:      $$.   -    ,d$$'    Shell: bash 5.2.37 
 $$;      Y$b._   _,d$P'      Resolution: 1366x768 
 Y$$.    `.`"Y$$$$P"'         DE: Cinnamon 6.4.10 
 `$$b      "-.__              WM: Mutter (Muffin) 
  `Y$$                        WM Theme: cinnamon (Default) 
   `Y$$.                      Theme: Adwaita-dark [GTK2/3] 
     `$$b.                    Icons: mate [GTK2/3] 
       `Y$$b.                 Terminal: WarpTerminal 
          `"Y$b._             CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx (8) @ 2.100GHz 
              `"""            GPU: AMD ATI Radeon Vega Series / Radeon Vega Mobile Series 
                              Memory: 7813MiB / 13925MiB 
```

- **Version de Python:**

```sh
# python --version
Python 3.13.3
```

- **Version de GCC:**

```sh
# g++ --version
g++ (Debian 14.2.0-19) 14.2.0
Copyright (C) 2024 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

## Ejecución

Esta sección detalla los pasos seguidos para replicar y validar el enfoque propuesto en el paper **_"Expert system for automatic microservices identification using API similarity graph"_** ([MsDecomposer](https://github.com/HduDBSI/MsDecomposer)). Los pasos se alinean con las tres partes principales del sistema descrito en el paper. Antes se instaló las dependencias:

<div align="center">
  <img src="images/1a.png" alt="Instalar dependencias" width="800">
</div>

* **1. Preparación de Datos y Cálculo de Similaridad de API:**
    * Se extrajeron las especificaciones RESTful API del sistema legado a ser descompuesto. Esto incluyó la recopilación de `operationId` para la similaridad de tópicos y esquemas de mensajes de respuesta para su similaridad.
    * Se implementó el cálculo de la similaridad entre estas APIs. Esto se realizó combinando dos medidas: la **candidate topic similarity** (derivada de los nombres de las operaciones o descripciones) y la **response message similarity** (basada en la estructura y contenido de los esquemas de respuesta).
    * La similaridad global de cada par de APIs fue obtenida a través de una combinación ponderada de ambas medidas.

    ![Extracción de datos y similaridad](/images/data_similarity.png)

* **2. Construcción del Grafo de Similaridad de API:**
    * Una vez calculadas todas las similaridades, se construyó un **grafo de similaridad de API**. En este grafo, cada **nodo** representa una API individual del sistema legado.
    * Las **aristas** entre los nodos fueron establecidas basándose en la similaridad global calculada en el paso anterior. El **peso de cada arista** corresponde directamente al valor de similaridad entre las dos APIs conectadas, reflejando la fuerza de su relación semántica.

    ![Construcción del grafo](/images/graph_construction.png)

* **3. Identificación de Microservicios Candidatos mediante Clustering de Grafos:**
    * Finalmente, se aplicó un **algoritmo de clustering basado en grafos** sobre el grafo de similaridad de API. Este algoritmo tiene como objetivo agrupar las APIs que exhiben una alta similaridad entre sí, formando así cúmulos cohesivos.
    * Cada uno de estos cúmulos identificados por el algoritmo representa un **microservicio candidato** para la descomposición del sistema monolítico. Los resultados fueron analizados para evaluar la efectividad de la partición.

    ![Clustering y microservicios](/images/clustering_microservices.png)


<!-- ## Run
- Create virtual environment:
```sh
python -m venv venv
```
- Install dependencies:
```sh
pip install -r requirements.txt
```

- Run code:
```sh
python main.py
```
- `specifications` folder contains openapi specification JSON files.  
- `results` folder is the output filepath   -->
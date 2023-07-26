import networkx as nx
import random


# Função para gerar um grafo com pesos nas arestas e grau médio desejado
def gerar_grafo_com_peso_e_grau_medio(num_vertices, grau_medio):
    # Cálculo do número de arestas desejado para atingir o grau médio
    num_arestas = int(num_vertices * grau_medio / 2)

    # Criação de um grafo vazio
    grafo = nx.Graph()

    # Adicionar vértices ao grafo
    for v in range(num_vertices):
        grafo.add_node(v)

    # Adicionar as arestas ao grafo até atingir o número desejado
    for i in range(num_arestas):
        while 1:
            u = random.randint(0, num_vertices)
            v = random.randint(0, num_vertices)
            if u != v and not grafo.has_edge(u,v):
                peso = random.randint(1, 100)
                grafo.add_edge(u, v, weight=peso)
                break

    return grafo

# Exemplo de uso:
num_vertices = [100000]
dividendo = [5000]
for i in num_vertices:
    for j in dividendo:
        grafo = gerar_grafo_com_peso_e_grau_medio(i, j)
        name = f"grafo_{i}_{j}"
        nx.write_weighted_edgelist(grafo, name)

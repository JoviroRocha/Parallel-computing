import networkx as nx
import random


# Function to generate a graph with edge weights and desired average degree
def generate_graph(num_vertex, middle_degree):
    # Calculation of the desired number of edges to reach the average degree
    num_edges = int(num_vertex * middle_degree / 2)

    # Creating an empty chart
    graph = nx.Graph()

    # Add vertex to the graph
    for v in range(num_vertex):
        graph.add_node(v)

    # Add edges to the graph until you reach the desired number
    for i in range(num_edges):
        while 1:
            u = random.randint(0, num_vertex)
            v = random.randint(0, num_vertex)
            if u != v and not graph.has_edge(u,v):
                weight = random.randint(1, 100)
                graph.add_edge(u, v, weight=weight)
                break

    return graph

# Example:
num_vertex = [100]
multiplier = [13]
for i in num_vertex:
    for j in multiplier:
        graph = generate_graph(i, j)
        name = f"graph_{i}_{j}"
        nx.write_weighted_edgelist(graph, name)

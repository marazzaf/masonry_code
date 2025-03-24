import networkx as nx
import matplotlib.pyplot as plt

#Generate a graph
G = nx.wheel_graph(6)

#Plot graph
nx.draw(G, with_labels=True)
plt.show()

print(list(nx.node_boundary(G, G)))

print(list(nx.node_boundary(G, (3, 4))))

print(list(nx.node_boundary(G, (3, 4), (0, 1, 5))))

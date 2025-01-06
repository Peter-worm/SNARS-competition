from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

##generated with chat gpt
def create_reduced_graph(original_graph, partition):
    # Initialize the reduced graph
    reduced_graph = nx.Graph()
    partition_d = defaultdict(list)
    for ix, val in enumerate(partition):
        partition_d[val].append(ix)
    
    # Step 1: Create nodes for each community in the partition
    community_nodes = {}
    for community_id, community_nodes_list in partition_d.items():
        community_nodes[community_id] = f"Community_{community_id}"
        reduced_graph.add_node(community_nodes[community_id])
    
    # Step 2: Count inter-community edges and add them to the reduced graph
    inter_community_edges = defaultdict(int)
    for u, v in original_graph.edges():
        # Find the communities for nodes u and v
        community_u = None
        community_v = None
        for community_id, community_nodes_list in partition_d.items():
            if u in community_nodes_list:
                community_u = community_id
            if v in community_nodes_list:
                community_v = community_id
        
        # If u and v belong to different communities, count the edge
        if community_u != community_v:
            inter_community_edges[(community_u, community_v)] += 1

    # Add edges between communities in the reduced graph based on inter-community edges
    for (community_u, community_v), weight in inter_community_edges.items():
        if community_u != community_v:
            reduced_graph.add_edge(community_nodes[community_u], community_nodes[community_v], weight=weight)

    # Step 3: Optionally add self-loops for intra-community edges (edges within the same community)
    for community_id, community_nodes_list in partition_d.items():
        # Intra-community edges
        intra_edges_count = 0
        for i in range(len(community_nodes_list)):
            for j in range(i + 1, len(community_nodes_list)):
                if original_graph.has_edge(community_nodes_list[i], community_nodes_list[j]):
                    intra_edges_count += 1
        if intra_edges_count > 0:
            reduced_graph.add_edge(community_nodes[community_id], community_nodes[community_id], weight=intra_edges_count)

    return reduced_graph, inter_community_edges, community_nodes

# Example usage:
def draw_reduced(g, partition):
    # Generate the reduced graph
    reduced_G, inter_community_edges, community_nodes = create_reduced_graph(g, partition)
    
    # Calculate the number of inter-community connections for each community
    community_connections = {node: 0 for node in reduced_G.nodes()}
    for (community_u, community_v), weight in inter_community_edges.items():
        community_connections[community_nodes[community_u]] += weight
        community_connections[community_nodes[community_v]] += weight
    
    # Draw the reduced graph with the number of inter-community connections as node labels
    pos = nx.circular_layout(reduced_G)  # Position nodes using spring layout
    plt.figure(figsize=(8, 6))
    
    # Draw nodes and edges
    nx.draw(reduced_G, pos, node_color='skyblue')

    # Draw node labels (number of inter-community connections for each community)
    # nx.draw_networkx_labels(reduced_G, pos)

    # Draw edge weights (show the number of inter-community connections between communities)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in reduced_G.edges(data=True)}
    nx.draw_networkx_edge_labels(reduced_G, pos, edge_labels=edge_labels)

    # Display the graph
    plt.title("Reduced Graph with Number of Inter-community Connections")
    plt.show()

def map_from_1(partition):
    community_map = {}
    mapping = {}

    for index, val in enumerate(partition):
        if val not in community_map:
            community_map[val] = len(community_map) + 1
        if index + 1 not in mapping:
            mapping[index + 1] = f"number_of_cluster_containing_{community_map[val]}"
    return mapping
    
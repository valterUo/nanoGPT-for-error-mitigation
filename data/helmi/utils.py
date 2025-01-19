import rustworkx as rx
import networkx as nx

def dag_node_to_dict(dag_node):
    return {
        'successors': dag_node.successors if dag_node.successors is not None else [],
        'predecessors': dag_node.predecessors if dag_node.predecessors is not None else [],
        'matchedwith': dag_node.matchedwith if dag_node.matchedwith is not None else [],
        'successorstovisit': dag_node.successorstovisit if dag_node.successorstovisit is not None else [],
        'isblocked': dag_node.isblocked if dag_node.isblocked is not None else False,
        'qindices': dag_node.qindices if dag_node.qindices is not None else [],
        'cindices': dag_node.cindices if dag_node.cindices is not None else []
    }

def rustworkx_to_networkx(rx_graph):
    nx_graph = nx.DiGraph()
    
    for node in rx_graph.node_indices():
        dag_node = rx_graph[node]
        # feature keyword is important for the Graph2Vec library
        nx_graph.add_node(node, feature = dag_node_to_dict(dag_node))

    for edge in rx_graph.edge_list():
        edge_data = rx_graph.get_edge_data(edge[0], edge[1])
        nx_graph.add_edge(edge[0], edge[1], **edge_data)

    return nx_graph
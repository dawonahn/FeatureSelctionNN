import heapq
import numpy as np

def backward_search(samples):
    nodes = make_queue(samples)
    repeated_indices = []
    last = []

    while True:
        if empty(nodes):
            break
        node = remove_front(nodes)
        num = len(node[1])
        heapq.heappush(last, node)

        if len(node[1]) == 0:
            break 
        if set(node[1]) not in repeated_indices:
            repeated_indices.append(set(node[1]))
            child_node = expand(node, samples)
            nodes = queuing_function(nodes, child_node)
            
    final = [(1 - i[0], i[1]) for i in last]
    
    return final

def forward_search(samples):
    nodes = make_queue(samples)
    repeated_indices = []
    last = []
    
    while True:
        if empty(nodes):
            break
        node = remove_front(nodes)
        num = len(node[1])
        heapq.heappush(last, node)

        if len(node[1]) == (samples.shape[1]-1):
            break
        if set(node[1]) not in repeated_indices:
            repeated_indices.append(set(node[1]))
            child_node = expand(node, samples)
            nodes = queuing_function(nodes, child_node)

    final = [(1 - i[0], i[1]) for i in last]
    
    return final
    
    
def make_queue(samples):
    
    f_num = samples.shape[1] - 1
    labels = samples[:, 0]
    features = samples[:, 1:]
    initial_lst = []
    for i in range(0, f_num):
        feature2 = features[:, i]
        acc = nearest_neighbors(labels, feature2)
        heapq.heappush(initial_lst, (1-acc, [i]))

    return [heapq.heappop(initial_lst)]

def empty(nodes):
    if len(nodes) == 0:
        return True

def remove_front(nodes):
    return heapq.heappop(nodes)

def queuing_function(nodes, child_nodes, goal=None):
    # Get the node with the minimium path cost
    for cnode in child_nodes:
        heapq.heappush(nodes, cnode)
    return nodes

def expand(node, samples):
    
    indices = node[1]
    f_len = samples.shape[1] -1
    all_indices = np.arange(f_len)
    
    new_indices = list(set(all_indices).difference(indices))
    
    lst = []
    labels = samples[:, 0]
    features = samples[:, 1:]
    for i in new_indices:
        idxs = []
        idxs.extend(indices)
        idxs.append(i)
        feature2 = features[:, idxs]
        acc = nearest_neighbors(labels, feature2)
        
        heapq.heappush(lst, (acc, idxs))
    return [heapq.heappop(lst)]

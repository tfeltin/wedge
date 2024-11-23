import queue
import numpy as np
from utils.utils import remove_first_layer

def get_next(dnn_list, output):
    for i, layer in enumerate(dnn_list):
        if output in layer['inputs']:
            return i
    return None


def conf_to_matrix(dnnconf, netconf):
    N = len(netconf['nodes'])
    L = len(dnnconf['dnn'])
    c = np.zeros((L, 1))
    K = np.zeros((N, N))
    S = np.zeros((L, L))
    B = np.zeros((N, N))

    for (i, layer) in enumerate(dnnconf['dnn']):
        c[i,0] = layer['consumption']
        for output, output_size in zip(layer['outputs'], layer['output_size']): 
            next = get_next(dnnconf['dnn'], output)
            if next is not None:
                S[i, next] = output_size

    for (i, node) in enumerate(netconf['nodes']):
        K[i,i] = 1 / node['capacity']
    
    for link in netconf['links']:
        src = link['source']
        dst = link['target']
        bw = link['bandwidth']
        B[src, dst] = 1 / bw
        B[dst, src] = 1 / bw

    return c, K, S, B


def complete_pl(pl, S):
    return pl + [pl[-1]] * (S.shape[0] - len(pl))


def plToPlParts(pl):
    nb = 0
    part = [0]
    pla = [pl[0]]
    for i in pl[1:]:
        if i == pla[len(pla)-1]:
            nb += 1
        else:
            pla.append(i)
            part.append(nb)
            nb += 1
    part.append(len(pl))
    return [part, pla]


def plPartsToPl(pl):
    [part, pla] = pl
    placement = []
    for i in range(len(part) - 1):
        l0, l1 = part[i], part[i+1]
        placement += [pla[i]] * (l1 - l0)
    return(placement)


def one_hot(p_list, N):
    P = np.zeros((N, len(p_list)))
    for i, s in enumerate(p_list):
        P[s, i] = 1
    return(P)


def max_TcTt(x, c, K, S, B):
        x_full = complete_pl(x, S)
        P = one_hot(x_full, B.shape[0])
        Tc = K @ P @ c
        Tt = (P @ S @ P.T) * B
        return max(np.max(Tc), np.max(Tt))


def max_Tt(px, c, K, S, B):
    P = one_hot(complete_pl(px, S), B.shape[0])
    Tt = (P @ S @ P.T) * B
    return np.max(Tt)


def count_splits(solution):
    return len([i for i in range(len(solution)-1) if solution[i] != solution[i+1]])


def is_single_threaded(solution):
    if len(solution) < 2:
        return True
    seen = [solution[0]]
    for i in range(1, len(solution)):
        if solution[i] not in seen:
            seen.append(solution[i])
        else: 
            if solution[i] != solution[i-1]:
                return False
    return True


def process(partial_x, c, K, S, B, max_time, best_x):
    L = S.shape[0]
    N = B.shape[0]

    if len(partial_x) >= L:
        partial_x_time = max_TcTt(partial_x, c, K, S, B)
        if partial_x_time < max_time:
            return [], partial_x_time, partial_x
        else:
            return [], max_time, best_x
    
    next_x = [partial_x + [i] for i in range(N)]
    next_process = []
    for nx in next_x:
        maxtt = max_Tt(nx, c, K, S, B)
        if maxtt < max_time:
            next_process.append(nx)
            next_max_time = max_TcTt(nx, c, K, S, B)
            if next_max_time < max_time:
                max_time = next_max_time
                best_x = nx 
    return next_process, max_time, best_x


def bnb(dnnconf, netconf, max_split=-1, parallel=True):
    c, K, S, B = conf_to_matrix(dnnconf, netconf)

    q = queue.Queue()
    for node in range(B.shape[0]):
        q.put([node])
        
    node, best_time = min(enumerate([max_TcTt([n], c, K, S, B) for n in range(B.shape[0])]), key=lambda x: x[1])
    best_placement = [node]

    while not q.empty():
        to_process = q.get_nowait()
        selected, mt, bx = process(to_process, c, K, S, B, best_time, best_placement)
        
        if mt < best_time:
            best_time = mt
            best_placement = bx
        
        for s in selected:
            if (max_split > 0 and count_splits(s) < max_split):
                if parallel or (not parallel and is_single_threaded(s)):
                    q.put(s)
    
    return plToPlParts(complete_pl(best_placement, S)), best_time

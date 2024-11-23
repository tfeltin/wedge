def state_to_netconf(state, dnnconf, config):
    netconf = {'nodes' : [], 'links' : []}
    for node in range(config['N_workers']):
        netconf['nodes'].append({'id': node, 'address' : config['worker_addresses'][node], 'capacity': 0, 'load': 0})
    
    # Node capacity
    for node in range(config['N_workers']):
        compute_times = state['node%s' % node]['time']
        placement = state['node%s' % node]['telemetry_placement']
        if placement is None:
            placement = [[0, len(dnnconf['dnn'])],[node]]
        for start_layer, time in compute_times.items():
            end_layer = placement[0][placement[0].index(start_layer) + 1]
            consumptions = sum([dnnconf['dnn'][layer]['consumption'] for layer in range(start_layer, end_layer)])
            netconf['nodes'][node]['capacity'] += (consumptions / time) / len(compute_times)

    # Link bandwidth
    measured_bandwidths = []
    for n0 in range(config['N_workers']):
        for layer, t in state['node%s' % n0]['transmission_time'].items():
            placement = state['node%s' % n0]['telemetry_placement']
            if placement is None:
                placement = [[0, len(dnnconf['dnn'])],[n0]]
            n1 = placement[1][placement[0].index(layer)]
            output_size = sum(dnnconf['dnn'][layer-1]['output_size'])
            bw = output_size / t
            measured_bandwidths.append([(min(n0, n1), max(n0, n1)), bw])
    in_conf = []
    for n0 in range(config['N_workers']):
        for n1 in range(n0 + 1, config['N_workers']):
            bw = [b[1] for b in measured_bandwidths if b[0]==(n0,n1)]
            if len(bw) > 0:
                netconf['links'].append({'source': n0, 'target': n1, 'symmetric': True, 'bandwidth': sum(bw)/len(bw), 'load': 0})
                in_conf.append((n0,n1))
    for node in range(config['N_workers']):
        for (n0, n1), bw in state['node%s' % node]['bandwidths'].items():
            if (n0,n1) not in in_conf:
                netconf['links'].append({'source': n0, 'target': n1, 'symmetric': True, 'bandwidth': bw, 'load': 0})

    return netconf


def remove_first_layer(new_placement):
    layers, nodes = new_placement
    layers = [l-1 for l in layers]
    if layers[1] == 0:
        layers = layers[1:]
        nodes = nodes[1:]
    else:
        layers[0] = 0
    return [layers, nodes]


def find_bandwidth(netconf, src, dst):
    for link in netconf['links']:
        if (link['source'] == src and link['target'] == dst) or (link['source'] == dst and link['target'] == src):
            return link['bandwidth']


def predicted_times(placement, dnn_conf, netconf):
    times = {}
    for i in range(len(placement[0]) - 1):
        start_layer = placement[0][i]
        end_layer = placement[0][i+1]
        node = placement[1][i]
        consumption = sum([dnn_conf['dnn'][layer]['consumption'] for layer in range(start_layer, end_layer)])
        capacity = netconf['nodes'][node]['capacity']
        times['Tc%s' % i] = consumption / capacity
        if i < len(placement[1]) - 1 :
            size = sum(dnn_conf['dnn'][end_layer]['output_size'])
            bandwidth = find_bandwidth(netconf, node, placement[1][i+1])
            times['Tt%s' % i] = size / bandwidth
    return times


def print_real_times(state, placement, pred_times):
    times = {}
    for i in range(len(placement[0]) - 1):
        start_layer = placement[0][i]
        end_layer = placement[0][i+1]
        node = placement[1][i]
        times['Tc%s' % i] =  state['node%s' % node]['time'][start_layer]
        if i < len(placement[1]) - 1 :
            times['Tt%s' % i] = state['node%s' % node]['transmission_time'][end_layer]
    print("\t".join(["%s : %.6f(%.6f)" % (a,b,pred_times[a]) for (a,b) in list(times.items())]))

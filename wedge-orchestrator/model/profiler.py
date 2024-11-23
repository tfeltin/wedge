import onnxruntime as rt
import networkx as nx
import numpy as np
import statistics
import cpuinfo
import logging
import onnx
import json
import os


SIZES = [-1, 32, 8, 8, 16, 16, 32, 64, -1, 8, 16, 64, 32, 64, 64, 128, 16]


def check_processability(dnn):

    """
    Checks wether the dnn can be processed by the algorithm
    or not. To do so it checks if, for all layer, its inputs are outputs of
    previous layers, in the ordering of the given list
    """
    
    available_inputs = []
    for i in dnn:
        available_inputs += i['outputs']
        for inp in i['inputs']:
            if inp not in available_inputs:
                return False
    return True


def reorder(dnn):

    """
    Reorders the layers of the dnn,
    making it processeable if possible
    """

    available_inputs = []
    process_dnn = dnn.copy()
    new_dnn = []
    while len(new_dnn) < len(dnn):
        for id in range(len(process_dnn)):
            correct = True
            d = process_dnn[id]
            for i in d['inputs']:
                if i not in available_inputs:
                    correct = False
                    break
            if correct:
                new_dnn.append(d.copy())
                available_inputs += d['outputs']
                del process_dnn[id]
                break
        if not correct:
            return new_dnn, False
    return new_dnn, True


def onnx_optimize_graph(model_path, out_path):

    """
    Takes a path to an onnx model,
    optimizes its graph (ORT_ENABLE_BASIC) level
    and outputs it to ./optimized/<model_name>.onnx
    """

    # Determines file for outputing the optimized graph
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set session options (optimization level and output path)
    opt = rt.SessionOptions()
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opt.optimized_model_filepath = out_path

    # Run optimization session
    rt.InferenceSession(model_path, opt, providers=['CPUExecutionProvider'])


def create_random_data(shape, type, minvalue, maxvalue, seed):

    """
    Create a numpy array of given shape and values type and
    fill it with random values between minvalue and maxvalue
    """
    nptype = np.dtype(type)
    np.random.seed(seed)
    ret = ((maxvalue-minvalue)*np.random.sample(shape)+minvalue).astype(nptype)
    return ret


def run_onnx_profiler(m, fixed_inputs={}, fixed_sizes={}):

    """
    Runs the onnx profiler on the given input
    and returns a path to the file containing the output
    """

    # Enables profiling and disable optimization for inference
    opt = rt.SessionOptions()
    opt.enable_profiling = True
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Converts model to string for inference
    onnx_model_str = m.SerializeToString()

    # Creates inference session
    sess = rt.InferenceSession(onnx_model_str, opt, providers=['CPUExecutionProvider'])

    # Retrieves shapes of input
    initializers_name = [i.name for i in m.graph.initializer]
    shape_arrs = [onnx_shape_to_array(i.type.tensor_type.shape) for i in m.graph.input if i.name not in initializers_name]
    names = [i.name for i in m.graph.input if i.name not in initializers_name]

    input_dict = {}

    for i in range(len(shape_arrs)):

        name = names[i]

        # Creates random data with input shape
        if name in fixed_inputs.keys():
            input_dict[name] = [np.float32(i) for i in fixed_inputs[name]]
        else:
            if name in fixed_sizes.keys():
                shape_arr = fixed_sizes[name]
            else:
                shape_arr = shape_arrs[i]
            d = create_random_data(shape_arr, np.float32, 0, 1, None)
            input_dict[names[i]] = d

    # Retrieves output names
    output_names = []
    for o in m.graph.output:
        output_names.append(o.name)

    # Runs the inference
    sess.run(output_names, input_dict)

    # Returns file with profiling infos
    return sess.end_profiling()


def run_onnx_json_profile(profile_file):

    """
    Selects interesting outputs in the profiling results file
    and returns it as a dictionnary
    """

    # Retrieve the profiling results from the profile file
    with open(profile_file, 'r') as f:
        j = json.load(f)

    try:
        os.remove(profile_file)
    except Exception as e:
        logging.warning(e)

    # Select the interesting profiling froms json
    ret = []
    for layer in j:
        if layer['cat'] == 'Node' and 'output_size' in layer['args'].keys():
            ret.append(layer)

    return ret


def profile_onnx_model(model, fixed_inputs={}, fixed_sizes={}):

    """
    Performs the whole onnx profiling work
    """

    # Runs the profiling
    o_profile_file = run_onnx_profiler(model, fixed_inputs, fixed_sizes)

    # Extracts datas from profile file
    ret = run_onnx_json_profile(o_profile_file)

    return ret


def tarjan_step(i, d, visited, depth, low, ap, parent, G):

    """
    adapted Tarjan's algorithm
    """

    visited[i] = True
    depth[i] = d
    low[i] = d
    isArticulation = False

    for ni in G.adj[i]:
        if not visited[ni]:
            parent[ni] = i
            tarjan_step(ni, d+1, visited, depth, low, ap, parent, G)
            if low[ni] >= depth[i]:
                isArticulation = True
            low[i] = min(low[i], low[ni])
        elif ni != parent[i]:
            low[i] = min(low[i], depth[ni])
    if parent[i] is not None and isArticulation:
        ap[i] = True


def find_articulation_points(G, start, end):

    """
    Finds the articulation points that would separate start
    and end in G
    """

    visited = {n: False for n in G.nodes}
    depth = {n: np.inf for n in G.nodes}
    low = {n: np.inf for n in G.nodes}
    ap = {n: False for n in G.nodes}
    parent = {n: None for n in G.nodes}

    tarjan_step(start, 0, visited, depth, low, ap, parent, G.to_undirected())

    return [start] + [n for n in G.nodes if ap[n]] + [end]


def find_non_consecutive_nodes(G, sorted_nodes_list):

    """
    Finds all unconsecutive nodes in a graph G along
    a node list
    """

    non_consecutive_list = []
    for i in range(len(sorted_nodes_list)-1):
        s = sorted_nodes_list[i]
        e = sorted_nodes_list[i+1]

        if not(len(G.adj[s]) == 1 and e in G.adj[s]):
            non_consecutive_list.append((s, e))
    return non_consecutive_list


def find_node_between(G, start, end):

    """
    Finds all the nodes n such that there is a
    path from start to n and from n to end
    """

    res = []
    for n in G.nodes:
        if nx.has_path(G, start, n) and nx.has_path(G, n, end):
            res.append(n)

    return res


def onnx_node_attributes_to_dict(args):

    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """

    def onnx_attribute_to_dict(onnx_attr):

        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """

        if onnx_attr.HasField('t'):
            return onnx.numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def onnx_shape_to_array(onnx_shape):

    """
    Converts an onnx shape object to an array
    """

    shape_arr = []

    for onedim in onnx_shape.dim:
        if onedim.HasField("dim_param"):
            shape_arr.append(1)
        if onedim.HasField("dim_value"):
            shape_arr.append(onedim.dim_value)

    return shape_arr


def get_shapes_and_types(model, inferred_model):

    """
    Retrieve shapes and types of inputs and outputs
    of the layers in the graph that are not initializers
    """

    whole_inputs = model.graph.input
    whole_outputs = model.graph.output

    inferred_value_info = inferred_model.graph.value_info
    initializers = [i.name for i in model.graph.initializer]

    # Adding Constant to initializers
    for n in model.graph.node:
        if n.op_type == 'Constant':
            initializers.append(n.output[0])

    out_values = {}

    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        outp = node.output
        inp = node.input

        if len(inp) != 0:

            inferred_i = [j for j in inferred_value_info if j.name in inp]
            inferred_o = [j for j in inferred_value_info if j.name in outp]

            legit_i = [j for j in whole_inputs if j.name in inp]
            legit_o = [j for j in whole_outputs if j.name in outp]

            all_io = inferred_i + inferred_o + legit_i + legit_o

            for j in all_io:
                if j.name not in out_values.keys() \
                        and j.name not in initializers:

                    out_values[j.name] = {
                        'shape': onnx_shape_to_array(j.type.tensor_type.shape),
                        'dtype': j.type.tensor_type.elem_type}

    return out_values


def create_layers_list_from_onnx_model(onnx_model, inferred_model):

    """
    Create a list of the different layers in the graph of
    an onnx model
    """

    sat = get_shapes_and_types(onnx_model, inferred_model)
    initializers_names = [i.name for i in onnx_model.graph.initializer]
    graph_inputs_name = [i.name for i in onnx_model.graph.input]

    layers = []

    graph_index = 0
    for i in range(len(onnx_model.graph.node)):

        typ = onnx_model.graph.node[i].op_type

        if typ not in ['Constant']:

            outp = [j for j in onnx_model.graph.node[i].output
                    if j in sat.keys()]

            inp = [j for j in onnx_model.graph.node[i].input
                   if j in sat.keys()]
            
            input_initializers = [j for j in onnx_model.graph.node[i].input
                                  if j in initializers_names and j in graph_inputs_name]

            array_out = [sat[i] for i in outp]
            array_in = [sat[i] for i in inp]

            name = onnx_model.graph.node[i].name
            t_attr = onnx_model.graph.node[i].attribute
            attr = onnx_node_attributes_to_dict(t_attr)

            outp_size = []
            for i in range(len(array_out)):
                if len(array_out[i]['shape']) != 0:
                    arr_size = np.prod(array_out[i]['shape'])
                    type_size = SIZES[array_out[i]['dtype']]
                    outp_size.append(arr_size*type_size)

            inp_size = []
            for i in range(len(array_in)):
                if len(array_in[i]['shape']) != 0:
                    arr_size = np.prod(array_in[i]['shape'])
                    type_size = SIZES[array_in[i]['dtype']]
                    inp_size.append(arr_size*type_size)

            layers.append({
                "layer": str(graph_index),
                "name": name,
                "type": typ,
                "inputs": inp,
                "outputs": outp,
                "output_size": outp_size,
                "inp_size": inp_size,
                "input_initializers": input_initializers,
                "odims": array_out,
                "idims": array_in,
                "attr": attr})
            graph_index += 1

    return layers, sat


def regroup_layers_by_index(layers, indexes, input_names, model_output):
    concerned_layers = [layers[j] for j in indexes]
    unconcerned_layers = [layers[j] for j in range(len(layers)) if j not in indexes]
    needed_inputs = {}
    needed_outputs = {}
    for l in concerned_layers:
        for in_l in l['inputs']:
            needed_inputs[in_l] = True
        for out_l in l['outputs']:
            needed_outputs[out_l] = False
    for l in unconcerned_layers:
        for in_l in l['inputs']:
            if in_l in needed_outputs.keys():
                needed_outputs[in_l] = True
    for out_l in needed_outputs.keys():
        if out_l in model_output:
            needed_outputs[out_l] = True
    for i in needed_inputs.keys():
        if i in needed_outputs.keys():
            needed_inputs[i] = False
    
    starts = [i for i in needed_inputs.keys() if needed_inputs[i]]
    ends = [i for i in needed_outputs.keys() if needed_outputs[i]]
    
     # Create combined layer (adding all the layers to group)
    i_s = [0]*len(starts)
    o_s = [0]*len(ends)
    for la in concerned_layers:
        for ind_s in range(len(starts)):
            s = starts[ind_s]
            if s in la['inputs']:
                i_s[ind_s] = la['inp_size'][la['inputs'].index(s)]
        for ind_e in range(len(ends)):
            e = ends[ind_e]
            if e in la['outputs']:
                o_s[ind_e] = la['output_size'][la['outputs'].index(e)]

    initializers = []
    for la in concerned_layers:
        initializers += la['input_initializers']
    comb_layer = {
        "layer": "",
        "name": "",
        "inputs": starts,
        "outputs": ends,
        "inp_size": i_s,
        "output_size": o_s,
        "input_initializers": initializers,
        "consumption": 0
        }

    for la in concerned_layers:
        if 'layer' in la.keys():
            comb_layer['layer'] += la['layer'] + "|"
        if 'name' in la.keys():
            comb_layer['name'] += la['name'] + "|"
        if 'consumption' in la.keys():
            comb_layer['consumption'] += la['consumption']
    while len(comb_layer['layer']) > 0 and comb_layer['layer'][-1] == "|":
        comb_layer['layer'] = comb_layer['layer'][:-1]
    while len(comb_layer['name']) > 0 and comb_layer['name'][-1] == "|":
        comb_layer['name'] = comb_layer['name'][:-1]
    
    ins = None

    inputs_fulfilled = {i: False for i in starts}

    val = True
    for i_f in inputs_fulfilled.keys():
        if i_f in input_names:
            inputs_fulfilled[i_f] = True
        else:
            if not inputs_fulfilled[i_f]:
                val = False
    
    if not val:
        for i in range(len(layers)):
            val = True
            for i_f in inputs_fulfilled.keys():
                if i_f in layers[i]['outputs']:
                    inputs_fulfilled[i_f] = True
                else:
                    if not inputs_fulfilled[i_f]:
                        val = False
            if val:
                ins = i + 1
                break
    else:
        ins = 0
    
    # Add combined layer to layers not to group
    unconcerned_layers.insert(ins, comb_layer)
    return unconcerned_layers


def regroup_consecutive_layers(layers, gr_lay_outs, G, start, ends, input_names = []):

    """
    Takes a list of layers to group, create
    a grouped layer and returns a layers list
    with the grouped layer
    """

    # Create lists of layers to group and layers not to group
    ltg = []
    lntg = []
    for la in layers:
        o_no_st = [i for i in la['outputs'] if i in gr_lay_outs and i not in start]
        o_in_grp = [i for i in la['outputs'] if i in gr_lay_outs]
        i_no_end = [i for i in la['inputs'] if i in gr_lay_outs and i not in ends]
        i_in_grp = [i for i in la['inputs'] if i in gr_lay_outs]
        if (len(o_in_grp) != 0 or len(la['outputs']) == 0) and (len(i_no_end) != 0 or len(la['inputs']) == 0):
            ltg.append(la)
        else:
            lntg.append(la)

    # Create combined layer (adding all the layers to group)
    i_s = [0]*len(start)
    o_s = [0]*len(ends)
    for la in ltg:
        for ind_s in range(len(start)):
            s = start[ind_s]
            if s in la['inputs']:
                i_s[ind_s] = la['inp_size'][la['inputs'].index(s)]
        for ind_e in range(len(ends)):
            e = ends[ind_e]
            if e in la['outputs']:
                o_s[ind_e] = la['output_size'][la['outputs'].index(e)]

    initializers = []
    for la in ltg:
        initializers += la['input_initializers']
    comb_layer = {
        "layer": "",
        "name": "",
        "inputs": start,
        "outputs": ends,
        "inp_size": i_s,
        "output_size": o_s,
        "input_initializers": initializers,
        "consumption": 0
        }

    for la in ltg:
        if 'layer' in la.keys():
            comb_layer['layer'] += la['layer'] + "/"
        if 'name' in la.keys():
            comb_layer['name'] += la['name'] + "/"
        if 'consumption' in la.keys():
            comb_layer['consumption'] += la['consumption']
    
    ins = None

    inputs_fulfilled = {i: False for i in start}

    val = True
    for i_f in inputs_fulfilled.keys():
        if i_f in input_names:
            inputs_fulfilled[i_f] = True
        else:
            if not inputs_fulfilled[i_f]:
                val = False
    
    if not val:
        for i in range(len(layers)):
            val = True
            for i_f in inputs_fulfilled.keys():
                if i_f in layers[i]['outputs']:
                    inputs_fulfilled[i_f] = True
                else:
                    if not inputs_fulfilled[i_f]:
                        val = False
            if val:
                ins = i + 1
                break
    else:
        ins = 0

    # Add combined layer to layers not to group
    lntg.insert(ins, comb_layer)
    return lntg


def create_graph_from_layers(layers):

    """
    Create a graph from a layers list
    using the inputs and outputs as nodes
    and processing layers as edges
    """

    G = nx.DiGraph()
    for la in layers:
        for i in la['inputs']:
            if i not in list(G.nodes):
                G.add_node(i)
        for o in la['outputs']:
            if o not in list(G.nodes):
                G.add_node(o)
        for i in la['inputs']:
            for o in la['outputs']:
                G.add_edge(i, o)
    return G


def create_graph_from_onnx_model(onnx_model, inferred_model):

    """
    Create a networkx graph from an onnx model
    """

    layers, sat = create_layers_list_from_onnx_model(onnx_model, inferred_model)
    return create_graph_from_layers(layers), sat


def find_end_of_cycle(cycle, G):

    """
    Finds the end node of an undirected cycle in a directed graph
    """

    for n in cycle:
        if len([i for i in G.successors(n) if i in cycle]) == 0:
            return n


def find_start_of_cycle(cycle, G):

    """
    Function to find the start node of an undirected cycle in a directed graph
    """

    for n in cycle:
        if len([i for i in G.predecessors(n) if i in cycle]) == 0:
            if len([i for i in G.predecessors(n)]) != 0:
                return n


def find_end_of_graph(G):

    """
    Function that returns a list of nodes
    in a graph that have no successors (i.e. end nodes)
    """

    ret = []
    for n in G.nodes:
        if len([i for i in G.successors(n)]) == 0:
            ret.append(n)
    return ret


def find_start_of_graph(G):

    """
    Function that returns a list of nodes
    in a graph that have no predecessors (i.e. start nodes)
    """

    ret = []
    for n in G.nodes:
        if len([i for i in G.predecessors(n)]) == 0:
            ret.append(n)
    return ret


def find_impossible_nodes(G):

    """
    Finds all the inputs/outputs that are
    in cycles
    """

    cycles = nx.cycle_basis(G.to_undirected())
    impossible = []
    for c in cycles:
        e = find_end_of_cycle(c, G)
        for n in c:
            if n != e and n not in impossible:
                impossible.append(n)
    return impossible


def extract_onnx_dnnconf(model_path, CHAIN_GRAPH=True, _=None, N_iter=50, save_path=None, fixed_inputs={}, fixed_sizes={}, finalNodes = 20):

    """
    Runs multiple profiling iterations and
    returns :
    - a dictionnary containing the profiling
    and layers grouping work in the format
    needed by the B&B algorithm
    - the name of all the graph outputs
    """

    # Prepare model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    model_path_list = model_path.split('/')
    model_path_list[-1] = 'inferred_%s' % model_path_list[-1]
    inferred_model_path = '/'.join(model_path_list)
    onnx.shape_inference.infer_shapes_path(model_path, inferred_model_path)
    inferred_model = onnx.load(inferred_model_path)

    # Verify the number of inputs and outputs of the model
    inputs_name = [i.name for i in onnx_model.graph.input]
    initializers_name = [i.name for i in onnx_model.graph.initializer]
    output_name = [i.name for i in onnx_model.graph.output]

    if CHAIN_GRAPH:
        if len(set(inputs_name) - set(initializers_name)) != 1:
            raise ValueError(
                'Model has %d inputs instead of 1.'
                % len(set(inputs_name) - set(initializers_name)))

        if len(output_name) != 1:
            raise ValueError(
                'Model has %d outputs instead of 1.'
                % len(onnx_model.graph.output))

    # Retrieve layers list and graph from it
    layers, sat = create_layers_list_from_onnx_model(onnx_model, inferred_model)
    G = create_graph_from_layers(layers)
    Gstarts = find_start_of_graph(G)
    Gends = find_end_of_graph(G)

    Gstart = Gstarts[0]
    Gend  = Gends[0]

    # Lists to contains results of the profiling iterations
    tmp_dur = {}
    tmp_th = {}

    for _ in range(N_iter):

        # Retrieves profiling datas
        profiled_layers = profile_onnx_model(onnx_model, fixed_inputs, fixed_sizes)

        # Stores duration and number of threads taken by layers
        dur_profiled = {}
        nr_threads = {}
        outputs_size = {}
        for layer in profiled_layers:
            i = layer['args']['graph_index']
            d = layer['dur']
            out_sizes = layer['args']['output_size']
            nt = len(layer['args']['thread_scheduling_stats']['sub_threads'])
            dur_profiled[i] = d
            nr_threads[i] = nt
            outputs_size[i] = [out_sizes]

        # Adds results of this iteration to global lists
        for layer in dur_profiled.keys():
            if layer not in tmp_dur.keys():
                tmp_dur[layer] = [dur_profiled[layer]]
            else:
                tmp_dur[layer].append(dur_profiled[layer])
            if layer not in tmp_th.keys():
                tmp_th[layer] = [nr_threads[layer]]
            else:
                tmp_th[layer].append(nr_threads[layer])

    # Computes per-layer means for durations and threads
    res_dur = {}
    for layer in tmp_dur.keys():
        res_dur[layer] = statistics.mean(tmp_dur[layer])

    res_th = {}
    for layer in tmp_th.keys():
        res_th[layer] = statistics.mean(tmp_th[layer])

    # Retrieves cpu frequencies
    cpufreq = cpuinfo.get_cpu_info()['hz_actual'][0]

    # Adds profiling informations to layers list
    for i in layers:
        i['dur_profiled'] = dur_profiled[i['layer']]
        i['nr_threads'] = nr_threads[i['layer']]
        i['consumption'] = i['dur_profiled']*i['nr_threads']*cpufreq/(10**6)
        if len(i['outputs']) != len(i['output_size']):
            if len(i['outputs']) == len(outputs_size[i['layer']]):
                out_dtypes_sizes = [SIZES[sat[j]['dtype']] for j in i['outputs']]
                i['output_size'] = [out_dtypes_sizes[j]*int(outputs_size[i['layer']][j]) for j in range(len(i['outputs']))]
            else:
                logging.warning("Can't determine the size of the output of layer %s, arbitrary fixed to np.inf" % i['name'])
                i['output_size'] = [np.inf]*len(i['outputs'])
    
    size_of_edge = {}
    for i in range(len(layers)):
        l = layers[i]
        for o in range(len(l['outputs'])):
            size_of_edge[l['outputs'][o]] = l['output_size'][o]
    
    for l in layers:
        if len(l['inputs']) != len(l['inp_size']):
            l['inp_size'] = [np.inf] * len(l['inputs'])
            for i in range(len(l['inputs'])):
                if l['inputs'][i] in size_of_edge.keys():
                    l['inp_size'][i] = size_of_edge[l['inputs'][i]]


    if CHAIN_GRAPH:
        # Find all crossing points
        # (i.e. nodes where there are no parallel computation)
        crossing_points = find_articulation_points(G, Gstart, Gend)

        # Find all unconsecutive crossing points
        # (i.e. places where we can group things)
        non_cons = find_non_consecutive_nodes(G, crossing_points)

        # Find the associated node lists
        final_nodes = [find_node_between(G, i[0], i[1]) for i in non_cons]

        # Regroup nodes between unconsecutive crossing points
        for ic in range(len(final_nodes)):
            c = final_nodes[ic]
            st = [non_cons[ic][0]]
            en = [non_cons[ic][1]]
            layers = regroup_consecutive_layers(layers, c, G, st, en)
    
    else:
        sorted_groupable = [-1]
        while len(sorted_groupable) > 0 and len(layers) > finalNodes:
            groupable = []
            layerFromOutput = {}
            layerFromInput = {}
            for i_l in range(len(layers)):
                for i in layers[i_l]['outputs']:
                    layerFromOutput[i] = (i_l, layers[i_l]['layer'])
                for i in layers[i_l]['inputs']:
                    if i in layerFromInput.keys():
                        layerFromInput[i].append((i_l, layers[i_l]['layer']))
                    else:
                        layerFromInput[i] = [(i_l, layers[i_l]['layer'])]

            for i_l in range(len(layers)):
                predNodes = [(-1,'-1') for _ in range(len(layers[i_l]['inputs']))]
                succNodes = [(-1,'-1') for _ in range(len(layers[i_l]['outputs']))]
                for i in range(len(layers[i_l]['inputs'])):
                    inName = layers[i_l]['inputs'][i]
                    if inName in layerFromOutput.keys():
                        predNodes[i] = (layerFromOutput[inName][0], layerFromOutput[inName][1])
                for i in range(len(layers[i_l]['outputs'])):
                    outName = layers[i_l]['outputs'][i]
                    if outName in layerFromInput.keys():
                        if len(layerFromInput[outName]) == 1:
                            succNodes[i] = (layerFromInput[outName][0][0], layerFromInput[outName][0][1])
                if len(set(predNodes)) == 1 and predNodes[0][1] != '-1':
                    r = (layers[i_l]['consumption'] + layers[predNodes[0][0]]['consumption']) / max(layers[i_l]['consumption'], layers[predNodes[0][0]]['consumption']) - 1
                    groupable.append((predNodes[0][1], layers[i_l]['layer'], r, False))
                if len(set(succNodes)) == 1 and succNodes[0][1] != '-1':
                    r = (layers[i_l]['consumption'] + layers[succNodes[0][0]]['consumption']) / max(layers[i_l]['consumption'], layers[succNodes[0][0]]['consumption']) - 1
                    groupable.append((layers[i_l]['layer'], succNodes[0][1], r, True))

            sorted_groupable = sorted(groupable, key=lambda t: t[2])

            already_grouped = []
            i = 0
            while len(sorted_groupable) > 0 and (sorted_groupable[i][2] < 0.2 or i < 1) and len(layers) > finalNodes:
                if not(sorted_groupable[i][0] in already_grouped or sorted_groupable[i][1] in already_grouped):
                    i1 = [j for j in range(len(layers)) if layers[j]['layer'] == sorted_groupable[i][0]][0]
                    i2 = [j for j in range(len(layers)) if layers[j]['layer'] == sorted_groupable[i][1]][0]

                    indexes = [i1, i2]
                    layers = regroup_layers_by_index(layers, indexes, inputs_name, output_name)

                    already_grouped.append(sorted_groupable[i][0])
                    already_grouped.append(sorted_groupable[i][1])
                i += 1

    # Create dictionnary for B&B algorithm
    i = 0
    dnnconf = {"dnn": []}
    for layer in layers:
        dnnconf["dnn"].append(
            {
                "id": i,
                "label": layer['name'],
                'consumption': layer['consumption'],
                'output_size': [int(ops) for ops in layer['output_size']],
                'outputs': layer['outputs'],
                'inputs': layer['inputs'],
                'input_initializers': layer['input_initializers']
            })
        i += 1

    # Check for processeability of the output and reorder if needed
    # processeable = check_processability(dnnconf['dnn'])
    # if not processeable:
    #     new_dnn, reordered = reorder(dnnconf['dnn'])
    #     if reordered:
    #         dnnconf['dnn'] = new_dnn
    #     else:
    #         raise ValueError("Produced dnnconf is invalid for processing. Impossible to make it processable")

    return dnnconf

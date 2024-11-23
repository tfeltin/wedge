import os
import numpy as np
import networkx as nx
from onnx import numpy_helper
import onnxruntime as rt

providers = os.environ.get('PROVIDERS')
if providers is None:
    providers = ['CPUExecutionProvider']
else:
    providers = eval(providers)


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
    rt.InferenceSession(model_path, opt, providers=providers)


def create_random_data(shape, type, minvalue, maxvalue, seed):

    """
    Create a numpy array of given shape and values type and
    fill it with random values between minvalue and maxvalue
    """
    nptype = np.dtype(type)
    np.random.seed(seed)
    ret = ((maxvalue-minvalue)*np.random.sample(shape)+minvalue).astype(nptype)
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
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

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

    sizes = [-1, 32, 8, 8, 16, 16, 32, 64, -1, 8, 16, 64, 32, 64, 64, 128, 16]

    layers = []

    graph_index = 0
    for i in range(len(onnx_model.graph.node)):

        typ = onnx_model.graph.node[i].op_type

        if typ not in ['Constant']:

            outp = [i for i in onnx_model.graph.node[i].output
                    if i in sat.keys()]

            inp = [i for i in onnx_model.graph.node[i].input
                   if i in sat.keys()]

            array_out = [sat[i] for i in outp]
            array_in = [sat[i] for i in inp]

            name = onnx_model.graph.node[i].name
            t_attr = onnx_model.graph.node[i].attribute
            attr = onnx_node_attributes_to_dict(t_attr)

            outp_size = 0
            if len(array_out) != 0 and len(array_out[0]['shape']) != 0:
                arr_size = np.prod(array_out[0]['shape'])
                type_size = sizes[array_out[0]['dtype']]
                outp_size = arr_size*type_size

            inp_size = 0
            if len(array_in) != 0 and len(array_in[0]['shape']) != 0:
                arr_size = np.prod(array_in[0]['shape'])
                type_size = sizes[array_in[0]['dtype']]
                inp_size = arr_size*type_size

            layers.append({
                "layer": str(graph_index),
                "name": name,
                "type": typ,
                "inputs": inp,
                "outputs": outp,
                "output_size": outp_size,
                "inp_size": inp_size,
                "odims": array_out,
                "idims": array_in,
                "attr": attr})
            graph_index += 1

    return layers


def regroup_consecutive_layers(layers, gr_lay_outs, G, start, ends):

    """
    Takes a list of layers to group, create
    a grouped layer and returns a layers list
    with the grouped layer
    """

    # Create lists of layers to group ant layers not to group
    ltg = []
    lntg = []
    for la in layers:
        o_no_st = [i for i in la['outputs'] if i in gr_lay_outs and i != start]
        i_no_end = [i for i in la['inputs'] if i in gr_lay_outs and i != ends]
        if len(o_no_st) != 0 or len(i_no_end) != 0:
            ltg.append(la)
        else:
            lntg.append(la)

    # Create combined layer (adding all the layers to group)
    i_s = [la["inp_size"] for la in ltg if start in la['inputs']]
    o_s = [la["output_size"] for la in ltg if ends in la['outputs']]
    comb_layer = {
        "layer": "",
        "name": "",
        "inputs": [start],
        "outputs": [ends],
        "inp_size": i_s[0],
        "output_size": o_s[0],
        "odims": [la["odims"] for la in ltg if ends in la['outputs']][0],
        "idims": [la["idims"] for la in ltg if start in la['inputs']][0],
        "consumption": 0
        }

    for la in ltg:
        if 'layer' in la.keys():
            comb_layer['layer'] += la['layer'] + "/"
        if 'name' in la.keys():
            comb_layer['name'] += la['name'] + "/"
        if 'consumption' in la.keys():
            comb_layer['consumption'] += la['consumption']

    ins = [i for i in range(len(layers)) if start in layers[i]["inputs"]][0]
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


def find_start_of_graph(G):

    """
    Function to find the start node of a directed graph
    having only one start node
    """

    ret = []
    for n in G.nodes:
        if len([i for i in G.predecessors(n)]) == 0:
            ret.append(n)
    return ret


def find_end_of_graph(G):

    """
    Function to find the end node of a directed graph having only one end node
    """

    ret = []
    for n in G.nodes:
        if len([i for i in G.successors(n)]) == 0:
            ret.append(n)
    return ret

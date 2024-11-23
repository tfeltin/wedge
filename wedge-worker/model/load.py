from os.path import exists
from urllib.request import urlretrieve
import onnx

from model.profiler import onnx_optimize_graph, create_layers_list_from_onnx_model, \
                            create_graph_from_layers, find_start_of_graph, find_end_of_graph, \
                            find_articulation_points, find_non_consecutive_nodes, find_node_between, \
                            regroup_consecutive_layers

def download_model(modelURL):
    raw_model_path = 'model/raw_%s' % modelURL.split("/")[-1]
    model_path = 'model/%s' % modelURL.split("/")[-1]
    if modelURL.split('.')[-1].strip() != 'onnx':
        print("Model -- Extension not supported. Wedge 2.0 only support ONNX models")
        return
    if not exists(raw_model_path):
        print("Model -- Downloading %s" % modelURL)
        urlretrieve(modelURL + "?raw=true",  raw_model_path)
        print("Model -- Model saved : %s" % modelURL.split("/")[-1])
    
    onnx_optimize_graph(raw_model_path, model_path)
    return model_path


def load_model_info(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    model_path_list = model_path.split('/')
    model_path_list[-1] = 'inferred_%s' % model_path_list[-1]
    inferred_model_path = '/'.join(model_path_list)
    onnx.shape_inference.infer_shapes_path(model_path, inferred_model_path)

    inferred_model = onnx.load(inferred_model_path)
    layers = create_layers_list_from_onnx_model(model, inferred_model)
    G = create_graph_from_layers(layers)
    Gstart = find_start_of_graph(G)[0]
    Gend = find_end_of_graph(G)[0]

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
        st = non_cons[ic][0]
        en = non_cons[ic][1]
        layers = regroup_consecutive_layers(layers, c, G, st, en)
    
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][-2:]

    return input_shape, layers


def create_partitioned_model(model_path, input_names, output_names):
    output_path = "%s_%s.onnx" % (model_path[:-5], input_names)
    onnx.utils.extract_model(model_path, output_path, input_names, output_names)
    return output_path
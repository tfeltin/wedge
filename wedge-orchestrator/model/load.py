from urllib.request import urlretrieve
from os.path import exists
from model.profiler import onnx_optimize_graph

def download_model(modelURL):
    raw_model_path = 'model/raw_%s' % modelURL.split("/")[-1]
    model_path = 'model/%s' % modelURL.split("/")[-1]
    if modelURL.split('.')[-1].strip() != 'onnx':
        print("  Model -- Extension not supported. Wedge 2.0 only support ONNX models")
        return
    if not exists(raw_model_path):
        print("  Model -- Downloading %s" % modelURL)
        urlretrieve(modelURL + "?raw=true",  raw_model_path)
        print("  Model -- Model saved : %s" % modelURL.split("/")[-1])
    
    onnx_optimize_graph(raw_model_path, model_path)
    return model_path

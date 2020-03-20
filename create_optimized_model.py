"""
File that creates an optimized graph for execution with
tensorrt.
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow.contrib.tensorrt as trt
from conversation import neuralnet_audio

TF_LOGDIR = "data/"
FROZEN_GRAPH_NAME = "audio_graph.pb"
TENSORRT_MODEL_PATH = "data/TensorRT_model.pb"
INFERENCE_BATCH_SIZE = 1

MODEL = neuralnet_audio.build_model()
MODEL.summary()


def freeze_session(
        session,
        keep_var_names=None,
        output_names=None,
        clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables())
                                .difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
        return frozen_graph


def load_graph(frozen_graph_filename):
    """
    loads the protobuf file from the disk and parse it to retrieve the
    unserialized graph_def
    """
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph, graph_def


if __name__ == '__main__':
    sess = tf.keras.backend.get_session()
    graph_output_names = [out.op.name for out in MODEL.outputs]
    graph_input_names = [out.op.name for out in MODEL.inputs]
    print("Graph inputs: {}".format(graph_input_names))
    print("Graph outputs: {}".format(graph_output_names))
    frozen_sess_graph = freeze_session(
        sess, output_names=[
            out.op.name for out in MODEL.outputs])

    tf.train.write_graph(
        frozen_sess_graph,
        TF_LOGDIR,
        FROZEN_GRAPH_NAME,
        as_text=False)
    # tf.summary.FileWriter('tensorboard_logdir', sess.graph_def)from PIL

    (graph, graph_def) = load_graph(TENSORRT_MODEL_PATH)

    your_outputs = ["dense_3/Sigmoid"]

    # convert (optimize) frozen model to TensorRT model
    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,  # frozen model
        outputs=your_outputs,
        max_batch_size=1,  # specify your max batch size
        max_workspace_size_bytes=1 << 31,  # specify the max workspace
        precision_mode="FP16")

    # write the TensorRT model to be used later for inference
    with gfile.FastGFile("./data/TensorRT_model.pb", 'wb') as f:
        f.write(trt_graph.SerializeToString())

    print("TensorRT model is successfully stored!")

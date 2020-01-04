import os
import argparse
import tensorflow as tf
import numpy as np
from util.function import get_classes,detections_boxes,get_anchors
from yolo import yolov3_wider,yolov3_wider_tiny



def freeze_graph(sess, output_graph):

    output_node_names = [
        "output_boxes",
        "inputs",
    ]
    output_node_names = ",".join(output_node_names)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names.split(",")
    )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("{} ops written to {}.".format(len(output_graph_def.node), output_graph))

def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(
                        tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(
                    tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                    --tiny   convert tiny.
                                    --classes get your classes.
                                    --ckpt   Saved as ckpt.
                                    --pb     Saved as frozen graph.
                                    --input  Set your weights path.
                                    --anchors Input your prior anchors.''')
    parser.add_argument('--tiny',default=False,help='tiny model mode.')
    parser.add_argument('--classes',default='config/wider.names',help='Your model preict classes')
    parser.add_argument('--ckpt',default='',help='Turn on the ckpt save mode.')
    parser.add_argument('--pb'  ,default='',help='Turn on the pb save mode.')
    parser.add_argument('--input',default='yolov3_wider.weights',help='outputpath')
    parser.add_argument('--anchors',default='config/wider_face_anchors.txt',help='The path of your prior anchors txt.')

    args = parser.parse_args()

    if(args.tiny):
        model = yolov3_wider_tiny.yolov3_wider_tiny
    else:
        model = yolov3_wider.yolov3_wider

    classes = get_classes(args.classes)
    anchors = get_anchors(args.anchors)

    if(args.ckpt != ''):
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        with tf.variable_scope('detector'):
            detections = model(inputs, len(classes), anchors)
            print(tf.global_variables(scope='detector'))
            load_ops = load_weights(tf.global_variables(
                scope='detector'), args.input)

        saver = tf.train.Saver(tf.global_variables(scope='detector'))
        with tf.Session() as sess:
            sess.run(load_ops)

            save_path = saver.save(sess, save_path=args.ckpt)
            print('Model saved in path: {}'.format(save_path))

    elif(args.pb != ''):
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3],name='inputs')
        with tf.variable_scope('detector'):
            detections = model(inputs, len(classes), anchors)
            load_ops = load_weights(tf.global_variables(scope='detector'),args.input)

        # Sets the output nodes in the current session
        boxes = detections_boxes(detections)

        with tf.Session() as sess:
            sess.run(load_ops)
            freeze_graph(sess, args.pb)
        

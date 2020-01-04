import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as layers
from .yolov3_wider import DBL,pad,detection_layer,upsampling

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def MDBL(input, filters, kernel_size, strides=2):
    padding = 'same'
    input = layers.max_pooling2d(input, 2, strides = strides, padding=padding)
    input = layers.conv2d(input, filters, kernel_size, padding=padding ,use_bias=None)
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    input = tf.nn.leaky_relu(input,alpha=LEAKY_RELU)
    return input

def MDBLs(len, input, filter, kernel_size, strides=2):
    for i in range(len):
        input = MDBL(input, filter*2**i, kernel_size, strides=strides)
    return input

def backbone(input):
    out = DBL(input, 16, 3)

    out = MDBLs(4, out, 32, 3)

    route_1 = out

    out = MDBLs(1, out, 512, 3)
    out = MDBLs(1, out, 1024, 3, 1)

    out = DBL(out, 256, 1)

    route_2 = out
    
    return route_1,route_2

def yolov3_wider_tiny(input, num_classes, anchors):
    """
    YOLO V3 tiny version trained on WIDER FACE dataset.
    """
    img_size = input.get_shape().as_list()[1:3]

    input = tf.cast(input,dtype=tf.float32)
    # normalization
    input = input / 255.
    with tf.variable_scope('backbone_tiny'):
        route_1,route_2 = backbone(input)
    with tf.variable_scope('yolov3_tiny'):
        out_2 = DBL(route_2, 512, 3)
        predict_1 = detection_layer(out_2, num_classes, img_size, anchors[3:6])

        route_2 = DBL(route_2, 128, 1)
        route_2_shape = route_2.get_shape().as_list()[1:3]
        route_1 = upsampling(route_1, route_2_shape)
        route_1 = tf.concat([route_2, route_1],axis=-1)
        out_1 = DBL(route_1, 256, 3)
        predict_2 = detection_layer(out_1, num_classes, img_size, anchors[0:3])

        result = tf.concat([predict_1, predict_2],axis=1)
    return result

if __name__ == "__main__":
    input = tf.placeholder(tf.float32,[None,416,416,3])
    import numpy as np
    with tf.Session() as sess:
        anchors = np.random.rand(6,2)
        a = yolov3_wider_tiny(input, 1, anchors)
        print(a)
        [print(x) for x in tf.global_variables()]
        print(len(tf.global_variables()))
    



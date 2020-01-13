import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as layers
from .yolov3_wider import DBL,pad,detection_layer,upsampling

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def MDBL(input, filters, kernel_size=3, padding="VALID", stride=2):
    input = DBL(input, filters, kernel_size)
    input = layers.max_pooling2d(input, 2, strides = stride, padding=padding)
    return input

def backbone(input):
    input = DBL(input, 16, 3)
    input = layers.max_pooling2d(input, 2, strides=2, padding="VALID")

    input = DBL(input, 32, 3)
    input = layers.max_pooling2d(input, 2, strides=2, padding="VALID")

    input = DBL(input, 64, 3)
    input = layers.max_pooling2d(input, 2, strides=2, padding="VALID")

    input = DBL(input, 128, 3)
    input = layers.max_pooling2d(input, 2, strides=2, padding="VALID")

    input = DBL(input, 256, 3)
    route_1 = input
    input = layers.max_pooling2d(input, 2, strides=2, padding="VALID")

    input = DBL(input, 512, 3)
    input = layers.max_pooling2d(input, 2, strides=1, padding="SAME")

    input = DBL(input, 1024, 3)
    input = DBL(input, 256, 1)
    route_2 = input

    input = DBL(input, 512, 3)
    
    return route_1,route_2,input

def yolov3_wider_tiny(input, num_classes, anchors):
    """
    YOLO V3 tiny version trained on WIDER FACE dataset.
    """
    img_size = input.get_shape().as_list()[1:3]

    input = tf.cast(input,dtype=tf.float32)
    # normalization
    input = input / 255.

    with tf.name_scope('yolov3_tiny'):
        route_1,route_2,input = backbone(input)
        predict_1 = detection_layer(input, num_classes, img_size, anchors[3:6])

        route_2 = DBL(route_2, 128, 1)
        route_1_shape = route_1.get_shape().as_list()[1:3]
        route_2 = upsampling(route_2, route_1_shape)
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
    



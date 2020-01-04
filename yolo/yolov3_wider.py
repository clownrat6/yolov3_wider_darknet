# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.layers as layers

from inspect import isfunction

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def pad(input, kernel_size, mode='CONSTANT'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(input, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

def DBL(input, filters, kernel_size, strides=1):
    padding = 'same'
    if strides > 1:
        padding = 'valid'
        input = pad(input,kernel_size)
    input = layers.conv2d(input, filters, kernel_size, strides=strides, padding=padding ,use_bias=None)
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    input = tf.nn.leaky_relu(input,alpha=LEAKY_RELU)
    return input

def backbone(input):
    """
    To build your backbone for the yolov3,I choose to use darknet53.
    You can also use another.
    Input maybe shape as [None,None,None,3].
    """
    out = DBL(input, 32, 3)
    
    out = DBL(out, 64, 3, strides=2)
    
    out = darkblocks(1,out,32)

    out = DBL(out, 128, 3, strides=2)

    out = darkblocks(2,out,64)

    out = DBL(out, 256, 3, strides=2)

    out = darkblocks(8,out,128)

    route_1 = out

    out = DBL(out, 512, 3, strides=2)

    out = darkblocks(8,out,256)

    route_2 = out

    out = DBL(out, 1024, 3, strides=2)

    out = darkblocks(4,out,512)

    route_3 = out

    return route_1,route_2,route_3

def darkblocks(len,input,filter):
    for i in range(len):
        input = darkblock(input,filter)
    return input

def darkblock(input,filters):
    """
    This kind of block is aspired by the resnetv2,which called bottleneck.
    """
    shortcut = input
    input = DBL(input, filters, 1)
    input = DBL(input, filters*2, 3)
    input = input + shortcut
    return input

def yolo_outblock(input,filters):
    """
    DBLx5
    """
    input = DBL(input, filters, 1)
    input = DBL(input, filters * 2, 3)
    input = DBL(input, filters, 1)
    input = DBL(input, filters*2, 3)
    input = DBL(input, filters, 1)
    route = input
    input = DBL(input, filters*2, 3)
    return input,route

def detection_layer(input, num_classes, img_size, anchors):
    """
    The last convontinal layer and decode layer
    """
    img_size = (416,416)
    num_anchors = len(anchors)
    predict = layers.conv2d(input, num_anchors * (5 + num_classes), 1, strides=1)
    shape = predict.get_shape().as_list()
    grid_size = shape[1:3]
    grids_num = grid_size[0] * grid_size[1]
    bboxes = 5 + num_classes
    predict = tf.reshape(predict, [-1, grids_num, num_anchors ,bboxes])
    
    box_centers, box_sizes, confidence, classes = tf.split(
        predict, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    batch_size = shape[0]
    a = tf.range(grid_size[0], dtype=tf.float32)
    b = tf.range(grid_size[1], dtype=tf.float32)
    x_offset = tf.reshape(a, (-1, 1))
    x_offset = tf.tile(x_offset,[grid_size[1],1])
    y_offset = tf.reshape(b, (1, -1))
    y_offset = tf.reshape(tf.transpose(tf.tile(y_offset,[grid_size[0],1]),[1,0]),[grids_num,1])
    x_y_offset = tf.concat([x_offset,y_offset],axis=-1)
    x_y_offset = tf.tile(tf.reshape(x_y_offset,[1,-1,1,2]),[1,1,num_anchors,1])
    
    box_centers = (box_centers + x_y_offset)*(img_size)/grid_size

    anchors = tf.tile(tf.reshape(anchors,[1,-1,2]),[grids_num,1,1])
    anchors = tf.cast(anchors,dtype=tf.float32)

    box_sizes = tf.exp(box_sizes) * anchors

    classes = tf.nn.sigmoid(classes)
    
    result_detect_result = tf.concat([box_centers,box_sizes, confidence, classes],axis=-1)
    result_detect_result = tf.reshape(result_detect_result,[-1,grids_num*num_anchors,result_detect_result.get_shape().as_list()[-1]])

    return result_detect_result

def upsampling(input, out_shape):
    input = tf.image.resize_nearest_neighbor(input, (out_shape[0],out_shape[1]))
    return input

def yolov3_wider(input, num_classes, anchors):
    """
    YOLO v3 model trained on WIDER FACE dataset.
    """
    img_size = input.get_shape().as_list()[1:3]

    input = tf.cast(input,dtype=tf.float32)
    # normalization
    input = input / 255.

    with tf.variable_scope('darknet53'):
        route_1,route_2,route_3 = backbone(input)
    
    with tf.variable_scope('yolov3'):
        out3,route3 = yolo_outblock(route_3, 512)
        predict_3 = detection_layer(out3, num_classes, img_size, anchors[6:9])

        out2 = DBL(route3, 256, 1)
        out2_shape = route_2.get_shape().as_list()[1:3]
        out2 = upsampling(out2, out2_shape)
        route_2 = tf.concat([out2, route_2],axis=-1)
        out2,route2 = yolo_outblock(route_2, 256)
        predict_2 = detection_layer(out2, num_classes, img_size, anchors[3:6])

        out1 = DBL(route2, 128, 1)
        out1_shape = route_1.get_shape().as_list()[1:3]
        out1 = upsampling(out1, out1_shape)
        route_1 = tf.concat([out1, route_1],axis=-1)
        out1,_ = yolo_outblock(route_1, 128)
        predict_1 = detection_layer(out1, num_classes, img_size, anchors[0:3])

        result = tf.concat([predict_1,predict_2,predict_3],axis=1)
    return result

if __name__ == "__main__":
    input = tf.placeholder(tf.float32,[None,416,416,3])
    import numpy as np
    anchors = np.random.randint(0,20,(9,2))
    temp = yolov3_wider(input, 1, anchors)
    with tf.Session() as sess:
        all = tf.global_variables()
        ok = open('ops_wider.txt','w')
        [ok.write("{} name:{} shape:{}\n".format(i+1, all[i].name,tuple(all[i].get_shape().as_list()))) for i in range(len(all))]
        print(len(all))
        ok.close()
import os
import cv2
import random
import numpy as np
import tensorflow as tf

out_dict={0:'left',1:'middle',2:'right'}

def get_anchors(filepath):
    anchors = open(filepath,'r').readline().strip('\n').split()
    anchors = [(int(x.split(',')[0]),int(x.split(',')[1])) for x in anchors]
    return anchors

def get_classes(filepath):
    f = open(filepath,'r')
    lines = [x.strip('\n') for x in f.readlines()]
    return lines

def detections_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    center_x, center_y, width, height, attrs = tf.split(
        detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1, name="output_boxes")
    return detections

def iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes
    box:[leftup_x,leftdoup_y,rightdown_x,rightdown_y]
    IOU = box1∩box2/box1∪box2
    """
    b1_xl, b1_yl, b1_xr, b1_yr = box1
    b2_xl, b2_yl, b2_xr, b2_yr = box2

    b1_area = max(b1_xr - b1_xl, 0) * max(b1_yr - b1_yl, 0)
    b2_area = max(b2_xr - b2_xl, 0) * max(b2_yr - b2_yl, 0)

    cross_xl = max(b1_xl, b2_xl)
    cross_yl = max(b1_yl, b2_yl)
    cross_xr = min(b1_xr, b2_xr)
    cross_yr = min(b1_yr, b2_yr)

    cross_width = max(cross_xr - cross_xl, 0)
    cross_height = max(cross_yr - cross_yl, 0)
    cross_area =  cross_width*cross_height 

    # prevent the zero division error.
    iou = cross_area + 1e-05 / (b1_area + b2_area - cross_area + 1e-05)
    return iou

def get_tensor_input_bboxes(model, num_classes, input_shape, anchors):

    input = tf.placeholder(tf.float32, [1, *input_shape , 3])

    with tf.variable_scope('detector'):
        detections = model(input, num_classes, anchors)

    boxes = detections_boxes(detections)

    return boxes, input

def get_tensor_input_bboxes_pb(frozen_graph):

    with frozen_graph.as_default():
        boxes = tf.get_default_graph().get_tensor_by_name("output_boxes:0")
        inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")

    return boxes, inputs

def judgebox(filtered_boxes, img, detection_size):
    """
    When there is only one face appearing in the pircture,this will matter.
    """
    if(len(filtered_boxes) == 1):
        """
        if there are existing faces,then do.
        """
        if(len(filtered_boxes[0]) == 1):
            """
            if there is only one face,then do.
            """
            the_box = convert_to_original_size(filtered_boxes[0][0][0], detection_size, img.shape[0:2])
            center_x = (the_box[0] + the_box[2])/2
            center_y = (the_box[1] + the_box[3])/2
            w,h,_ = img.shape
            color = tuple([random.randint(0,255) for x in range(3)])
            out_text = out_dict[int(center_x/h*3)]
            img = cv2.line(img,(int(h/3),0),(int(h/3),h),color)
            img = cv2.line(img,(int(h*2/3),0),(int(h*2/3),h),color)
            img = cv2.putText(img, out_text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8 ,(255,255,255),2)
    return img

def convert_to_original_size(box, detection_size, input_shape):
    w,h = input_shape
    length_norm = max(w,h)
    norm_begin_x = int(length_norm/2-w/2)
    norm_begin_y = int(length_norm/2-h/2)
    offset = [norm_begin_y,norm_begin_x]
    pad_shape = [length_norm,length_norm]
    offset = np.tile(offset,[2])
    detection_size = np.tile(detection_size,[2])
    pad_shape = np.tile(pad_shape,[2])
    box = box*pad_shape/detection_size-offset
    box = box.astype(np.int32)
    return box

def draw_boxes(boxes, img, class_names, detection_size):
    """
    boxes:{0:[(box1),(box2),...],...}
    boxn:(array[leftup_x,leftup_y,rightdown_x,rightdown_y],confidence_score)
    """
    class_num = len(class_names)
    for i in range(len(boxes)):
        one_class_boxes = boxes[i]
        for box,score in one_class_boxes:
            color = [int(score*255/(x+1)) for x in range(3)]
            box = convert_to_original_size(box, detection_size, img.shape[0:2])
            img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
            img = cv2.putText(img, '%s:%.3f'%(class_names[i],score), tuple(box[0:2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return img

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def pad_image(image, out_size, fill_value=128):
    """
    To ensure that the width is same as the height.
    """
    w,h,_ = image.shape
    length_norm = max(w,h)
    norm_begin_x = int(length_norm/2-w/2)
    norm_begin_y = int(length_norm/2-h/2)
    zeros = np.full((length_norm,length_norm,3), fill_value,dtype=np.uint8)
    zeros[norm_begin_x:w+norm_begin_x,norm_begin_y:h+norm_begin_y,:] = image
    out_resize = cv2.resize(zeros,out_size)
    return out_resize

if __name__ == "__main__":
    temp = cv2.imread('../1.jpg')
    temp = pad_image(temp, (416,416))
    cv2.imshow('sdaf',temp)
    cv2.waitKey(0)
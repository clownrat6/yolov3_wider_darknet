import cv2
import time
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from yolo import yolov3_wider,yolov3_wider_tiny
from util.function import pad_image,get_tensor_input_bboxes,get_tensor_input_bboxes_pb,get_classes,draw_boxes,\
                          non_max_suppression,load_graph,judgebox,get_anchors

def main(args=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    # config = tf.ConfigProto(
    #     gpu_options=gpu_options,
    #     log_device_placement=False,
    # )

    classes = get_classes(args.classes)
    if(args.pb != ''):

        t0 = time.time()

        frozenGraph = load_graph(args.pb)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))
        boxes, inputs = get_tensor_input_bboxes_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            if(int(args.input_video) < 10):
                cap = cv2.VideoCapture(int(args.input_video))
            else:
                cap = cv2.VideoCapture(args.input_video)
            while(True):
                start = time.time()
                flag,img = cap.read()
                if not flag:
                    break
                img_resized = pad_image(img, (416,416))
                img_resized = img_resized.astype(np.float32)
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                                     confidence_threshold=0.4,
                                                     iou_threshold=0.5)
                img = draw_boxes(filtered_boxes, img, classes, (416,416))
                img = judgebox(filtered_boxes, img, (416,416))
                end = time.time()
                fps = int(1/(end-start))
                img = cv2.putText(img, 'FPS:{}'.format(fps), (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (217,255,128), 2)
                cv2.imshow('冲',img)
                cv2.waitKey(1)

    elif(args.ckpt != ''):
        if args.tiny:
            model = yolov3_wider_tiny.yolov3_wider_tiny
        else:
            model = yolov3_wider.yolov3_wider
            
        anchors = get_anchors(args.anchors)
        boxes, inputs = get_tensor_input_bboxes(model, len(classes), (416,416), anchors)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer()) 
            t0 = time.time()
            saver.restore(sess, args.ckpt)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            if(int(args.input_video) < 10):
                cap = cv2.VideoCapture(int(args.input_video))
            else:
                cap = cv2.VideoCapture(args.input_video)
            
            while(True):
                start = time.time()
                flag,img = cap.read()
                if not flag:
                    break
                img_resized = pad_image(img, (416,416))
                img_resized = img_resized.astype(np.float32)
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                                     confidence_threshold=0.4,
                                                     iou_threshold=0.5)
                img = draw_boxes(filtered_boxes, img, classes, (416,416))
                img = judgebox(filtered_boxes, img, (416,416))
                end = time.time()
                fps = int(1/(end-start))
                img = cv2.putText(img, 'FPS:{}'.format(fps), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,128), 1)
                cv2.imshow('冲',img)
                cv2.waitKey(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                    --tiny   convert tiny.
                                    --classes get your classes.
                                    --ckpt   path for import ckpt.
                                    --pb     path for import frozen graph.
                                    --input_video path for input video.
                                    --anchors path for your prior anchors.''')

    parser.add_argument('--tiny',default=False,help='tiny model mode.')
    parser.add_argument('--classes',default='config/wider.names',help='Your model preict classes')
    parser.add_argument('--ckpt',default='',help='Turn on the ckpt save mode.')
    parser.add_argument('--pb'  ,default='',help='Turn on the pb save mode.')
    parser.add_argument('--input_video',default=0,help='Input 0 or video file path.')
    parser.add_argument('--anchors',default='config/wider_face_anchors.txt',help='Input the prior anchors.')

    args = parser.parse_args()
    main(args)
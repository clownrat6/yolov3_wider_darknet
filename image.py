import cv2
import time
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from yolo import yolov3_wider,yolov3_wider_tiny
from util.function import pad_image,get_tensor_input_bboxes,get_tensor_input_bboxes_pb,get_classes,draw_boxes,\
                          non_max_suppression,load_graph,judgebox,get_anchors
import matplotlib.pyplot as plt

def main(args=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

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
            img = plt.imread(args.input_img)
            # w,h = img.shape[0:2]
            # length_norm = max(w,h)
            # offset = [int(length_norm/2-w/2),int(length_norm/2-h/2)]
            img_resized = pad_image(img, (416,416))
            # img_recover = cv2.resize(img_resized, (length_norm,length_norm))
            # img_resized = cv2.circle(img_resized, (150,320), 5, (0,0,0), 5)
            # img_recover = cv2.circle(img_recover, (int(150*length_norm/416),int(320*length_norm/416)), 5, (0,0,0), 5)
            # print(offset)
            # point = (int(150*length_norm/416-offset[1]),int(320*length_norm/416-offset[0]))
            # print(point)
            # img = cv2.circle(img, point, 5, (255,255,255), 5)
            # plt.subplot(131)
            # plt.imshow(img_resized)
            # plt.subplot(132)
            # plt.imshow(img_recover)
            # plt.subplot(133)
            # plt.imshow(img)
            # plt.show()
            img_resized = img_resized.astype(np.float32)
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

            temp = open('box1.txt','w')
            for i in range(detected_boxes.shape[1]):
                temp.write("{} {}\n".format(*detected_boxes[0][i]))
            temp.close()
            filtered_boxes = non_max_suppression(detected_boxes,
                                                    confidence_threshold=0.5,
                                                    iou_threshold=0.4)
            img = draw_boxes(filtered_boxes, img, classes, (416,416))
            img = judgebox(filtered_boxes, img, (416,416))

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
            
            img = cv2.imread(args.input_img)
            img_resized = pad_image(img, (416,416))
            img_resized = img_resized.astype(np.float32)
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

            filtered_boxes = non_max_suppression(detected_boxes,
                                                    confidence_threshold=0.4,
                                                    iou_threshold=0.5)

            img = draw_boxes(filtered_boxes, img, classes, (416,416))
            img = judgebox(filtered_boxes, img, (416,416))

    plt.imsave(args.output_img,img)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                    --tiny   convert tiny.
                                    --classes get your classes.
                                    --ckpt   path for import ckpt.
                                    --pb     path for import frozen graph.
                                    --input_img path for input image.
                                    --output_img path for output image
                                    --anchors path for your prior anchors.''')
    parser.add_argument('--tiny',default=False,help='tiny model mode.')
    parser.add_argument('--classes',default='wider.names',help='Your model preict classes')
    parser.add_argument('--ckpt',default='',help='Turn on the ckpt save mode.')
    parser.add_argument('--pb'  ,default='',help='Turn on the pb save mode.')
    parser.add_argument('--input_img',default='a.jpg',help='inputimage_path')
    parser.add_argument('--output_img',default='x.jpg',help='outputimage_path')
    parser.add_argument('--anchors',default='config/wider_face_anchors.txt',help='Input the prior anchors.')    

    args = parser.parse_args()
    main(args)
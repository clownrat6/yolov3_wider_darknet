import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

# At first,put your rootpath of the WIDER_FACE dataset.
# In order to train conveniently,I move all the train and valid picutres in the data folder.

txt_path = 'wider_face_split/'

def move(target,dataset_path):
    index = 0
    label_txt_name = 'wider_face_{}_bbx_gt.txt'.format(target)
    temp = open(os.path.join(dataset_path,txt_path,label_txt_name),'r')
    all_txt = open('data/'+target+'.txt','w')
    while(True):
        filename = temp.readline().strip('\n')
        print(index,filename)
        if(filename==""):
            break
        all_txt.write('data/{}'.format(target) + str(index) + '.jpg\n')
        n_box = int(temp.readline())
        n_box = n_box if(n_box != 0) else 1
        images = 'WIDER_{}/images'.format(target)
        image = cv2.imread(os.path.join(dataset_path,images,filename))
        w,h,_ = image.shape
        one_ = open('data/{}/{}.txt'.format(target,index),'w')
        for i in range(n_box):
            box = temp.readline().strip().split(' ')
            box = [int(x) for x in box]
            leftup = (box[0],box[1])
            rightdown = (box[0]+box[2],box[1]+box[3])
            center_x = box[0]+int(box[2]/2.)
            center_y = box[1]+int(box[3]/2.)
            center_x_ = center_x/h
            center_y_ = center_y/w
            width = box[2]/h
            height = box[3]/w
            one_.write('0 {} {} {} {}\n'.format(center_x_,center_y_,width,height))
        cv2.imwrite('data/{}/{}.jpg'.format(target,index),image)
        index += 1
    
def main(args):
    if(args.dataset != ''):
        for name in ['train','valid']:
            move(name,args.dataset)
    else:
        print('please input dataset root path')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                    --dataset Input your WIDER FACE dataset path.''')

    parser.add_argument('--dataset',default='',help='Path for WIDER FACE dataset.')
    
    args = parser.parse_args()
    main(args)



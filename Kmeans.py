import os
import numpy as np

from PIL import Image

class cluster(object):
    def __init__(self,k,outputpath):
        self.center_num = k
        self.outpath = outputpath

    def iou(self,boxes,clusters):
        '''
        Next thing is wrong !!!!!!
        Box will be shaped as [xu,yu,xd,xd]
        |---------------|
        |               |
        |      A        |
        |       |-------|-------|
        |       |       |       |
        |       |  C    |       |
        |-------|-------|       |
                |          B    |
                |---------------|
        cu = (max(box1[0],box2[0]),max(box1[1],box2[1]))
        cd = (min(box1[2],box2[2]),min(box1[3],box2[3])) 
        '''
        k = self.center_num
        n = boxes.shape[0]

        boxes_area      = boxes[...,0] * boxes[...,1]
        boxes_area      = np.tile(np.reshape(boxes_area,[n,1]),[1,k])
        clusters_area   = clusters[...,0] * clusters[...,1]
        
        clusters_area   = np.tile(np.reshape(clusters_area,(1,k)),[n,1])

        boxes_w         = np.tile(np.reshape(boxes[...,0],[n,1]),[1,k]) # (n,1)
        clusters_w      = np.tile(np.reshape(clusters[...,0], (1,k)),[n,1])
        boxes_w         = np.minimum(boxes_w,clusters_w)

        boxes_h         = np.tile(np.reshape(boxes[...,1],[n,1]),[1,k]) # (n,1)
        clusters_h      = np.tile(np.reshape(clusters[...,1], (1,k)),[n,1])
        boxes_h         = np.minimum(boxes_h,clusters_h)


        cross_area      = boxes_w*boxes_h

        return cross_area/(boxes_area+clusters_area-cross_area)

    def kmeans(self,boxes):
        k = self.center_num
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            # print(current_nearest.shape)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change

            a = [0 for x in range(k)]
            for i in range(box_number):
                a[current_nearest[i]] += 1
                clusters[current_nearest[i],0] += boxes[i,0]
                clusters[current_nearest[i],1] += boxes[i,1]
            clusters = np.array([clusters[abc]/a[abc] for abc in range(k)])    
            print(clusters)
            last_nearest = current_nearest

        return clusters

    def txt2boxes(self):
        all_train_txts = [os.path.splitext(x)[0] for x in os.listdir('data/train') if(os.path.splitext(x)[1] == '.txt')]
        box_collection = []
        for one_record in all_train_txts:
            pic = Image.open('data/train/'+one_record+'.jpg')
            w,h = pic.size
            f = open('data/train/' + one_record+'.txt','r')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split()
                width = int(float(line[3])*w)
                height = int(float(line[4])*h)
                box_collection.append([width, height])
            f.close()
        result = np.array(box_collection)
        return result



    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def result2txt(self, data):
        '''
        remeber to change the result path
        '''
        f = open(self.outpath, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes)
        result = result[np.lexsort(result.T[0, None])]

        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
                self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='''
                                    --output Set your output path.
                                    --num    Set your cluster number.''')
    parser.add_argument('--num',default=9,help='Input the number that How many you want to cluster.')
    parser.add_argument('--output',default='config/anchors.txt',help='Input the path where you want to put your result in')
    args = parser.parse_args()

    obj = cluster(args.num,args.output)
    result = obj.txt2clusters()
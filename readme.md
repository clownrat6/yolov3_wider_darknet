# yolov3_wider

使用了yolov3网络结构而后使用WIDER FACE数据集对网络进行训练，正常结构迭代4000个batch，tiny版迭代了13000个batch.

# 训练过程

首先使用了darknet开源代码，使用的是[AlexeyAB重构版](https://github.com/AlexeyAB/darknet)，而后编译生成darknet可执行文件，首先而后通过generatorfordark.py解析生成darknet能够处理的数据集格式，而后通过Kmeans.py对WIDER FACE数据集进行K聚类，聚类得到9(origin)、6(tiny)个先验的anchor框，其将自动去往Kmeans.py所在文件夹下的data文件夹下去取WIDER_FACE中的box长宽，使用darknet可执行文件训练得到weights文件，而后通过convert.py将weights文件转化为ckpt或者pb文件，而后使用video.py或者image.py输出结果。

## convert.py

使用示例:  
``` python
python convert.py --classes config/wider.names --ckpt temp/temp --anchors config/wider_face_anchors.txt --input yolov3_wider_4000.weights
or
python convert.py --classes config/wider.names --pb temp_tiny/temp.pb --anchors config/wider_face_anchors_tiny.txt --input yolov3_wider_tiny_13000.weights --tiny True
```
1. --classes 输入有多少个类，只用对脸进行分类，所以wider.names中只有一行.
2. --ckpt/--pb 二选一选择使用ckpt方式，以及ckpt保存的路径，或者选择使用pb方式，以及pb文件保存的路径.
3. --anchors 输入先验框路径.
4. --tiny 选择是否使用tiny，注意ckpt、pb、anchors都得是tiny版本.
5. --input 输入weights文件路径.

## video.py

使用示例：
``` python
python video.py --classes config/wider.names --ckpt temp/temp --anchors config/wider_face_anchors.txt --input_video 0
or
python video.py --classes config/wider.names --pb temp_tiny/temp.pb --anchors config/wider_face_anchors_tiny.txt --input_video 0 --tiny True
```
1. --classes 输入有多少个类，只用对脸进行分类，所以wider.names中只有一行.
2. --ckpt/--pb 二选一选择使用ckpt方式，以及ckpt保存的路径，或者选择使用pb方式，以及pb文件保存的路径.
3. --anchors 输入先验框路径.
4. --tiny 选择是否使用tiny，注意ckpt、pb、anchors都得是tiny版本.
5. --input_video 输入视频文件路径，或者输入带驱动相机的句柄号。

## image.py

使用示例
``` python
python image.py --classes config/wider.names --ckpt model/tiny/yolov3_wider_tiny --anchors config/wider_face_anchors_tiny.txt --input_img beautiful.jpg --output_img abc.jpg --tiny True
or
python image.py --classes config/wider.names --pb model/origin/yolov3_wider.pb --anchors config/wider_face_anchors.txt --input_img beautiful.jpg --output_img abc.jpg
```
1. --classes 输入有多少个类，只用对脸进行分类，所以wider.names中只有一行.
2. --ckpt/--pb 二选一选择使用ckpt方式，以及ckpt保存的路径，或者选择使用pb方式，以及pb文件保存的路径.
3. --anchors 输入先验框路径.
4. --tiny 选择是否使用tiny，注意ckpt、pb、anchors都得是tiny版本.
5. --input_image 输入图片路径.
6. --output_image 输出图片路径.

## 训练过程

首先使用generatorfordark.py生成数据集文件夹，使用示例：
``` python
python generatorfordark.py --dataset 'E:\\dataset\\datasets\\WIDER_FACE'
```
而后按照[AlexeyAB重构版](https://github.com/AlexeyAB/darknet)：How to train(to detect your custom objects)的指示,修改配置网络结构的yolov3_wider.config或yolov3_wider_tiny.config以及类别wider.names以及数据配置wider.data,或者使用config文件夹下的文件进行操作，而后开始训练得到.weights文件，转换成ckpt或者pb文件即可使用.

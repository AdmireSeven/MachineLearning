"""
@author:Admire

Function：Project3_2
        Clustering of pixels based on their color value using k-means.

Email:admireseven@163.com
"""
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
#添加中文字体支持
from matplotlib.font_manager import FontProperties

def clusterpixels(infile,k,steps):
    im = array(Image.open(infile))
    dx = im.shape[0] / steps
    dy = im.shape[1] / steps
    #shape[0]返回行数 shape[1]返回列数
    # compute color features for each region
    features = []
    #RGB三色通道
    for x in range(steps):
        for y in range(steps):
            #切片操作
            #slice indices must be integers or None or have an __index__ method
            R = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 0])
            G = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 1])
            B = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 2])
            features.append([R,G,B])
    features = array(features,'f')
    #聚类
    centroids,variance = kmeans(features,k)
    code,distance = vq(features,centroids)
    #create image with cluster labels
    codeim = code.reshape(steps,steps)
    #codeim = imresize(codeim,im.shape[:2],'nearest') 已淘汰
    codeim = np.array(Image.fromarray(codeim).resize((im.shape[1],im.shape[0])))
    return codeim

infile_stones = 'stones.jpg'
im_stones = array(Image.open(infile_stones))
steps = (50,100)
# image is divided in steps*steps region

#显示原图
fig = plt.Figure(figsize = (8,8))
plt.subplot(231)
plt.title('original')
plt.axis('off')
plt.imshow(im_stones)

#对stone的像素进行聚类 100*100
codeim = clusterpixels(infile_stones,2,steps[-1])
plt.subplot(232)
plt.title('k = 2')
plt.imshow(codeim)

codeim = clusterpixels(infile_stones,3,steps[-1])
plt.subplot(233)
plt.title('k = 3')
plt.imshow(codeim)

codeim = clusterpixels(infile_stones,4,steps[-1])
plt.subplot(234)
plt.title('k = 4')
plt.imshow(codeim)

codeim = clusterpixels(infile_stones,5,steps[-1])
plt.subplot(235)
plt.title('k = 5')
plt.imshow(codeim)

codeim = clusterpixels(infile_stones,6,steps[-1])
plt.subplot(236)
plt.title('k = 6')
plt.imshow(codeim)

plt.show()



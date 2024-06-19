# 图像拼接

import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# os.chdir('pwd\\')

# ----------- 1. 基于Stitcher类 -----------
# imgPath为图片所在的文件夹相对路径
imgPath = 'sift\\'
imgList = os.listdir(imgPath)
imgs = []

for imgName in imgList:
    pathImg = os.path.join(imgPath, imgName)
    img = cv.imread(pathImg)
    if img is None:
        print("图片不能读取：" + imgName)
        sys.exit(-1)
    imgs.append(img)

#######################################################
# 此处补全
# 查询 cv.Stitcher() 类的使用方法,构造全景拼接对象
# 利用stitch方法拼接图像，
# 利用imwrite类别输出拼接结果
#######################################################
stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
_result, pano = stitcher.stitch(imgs)
cv.imwrite('result\\p12.png', pano)


# -------------- 2. 高斯差分 -----------------
im1 = cv.imread('sift\\p1.png')
im1 = (im1[:,:,0]).astype(np.double)
sz = 7; sig = 3


#######################################################
# 此处补全：对im1执行高斯模糊。
# 提示：可用cv.GaussianBlur
im_gs = cv.GaussianBlur(im1, (sz, sz), sig)
#######################################################
im3 = im1 - im_gs
# uncomment this line for debuging
# cv.imshow("2", im3);   cv.waitKey(0)
cv.imwrite('result\\im1-im_gs.png', im3)
#######################################################
# 此处补全：将im3的灰度值归一化至[0,255]
im3 = cv.normalize(im3, None, 0, 255, cv.NORM_MINMAX)
#######################################################

cv.imwrite('result\\s1-g.png', im1)
cv.imwrite('result\\s1-gs.png', im_gs)
cv.imwrite('result\\s1-cf.png', im3)


# ----------- 3. sift特征点 -----------
sift = cv.SIFT_create()
im1 = cv.imread('sift\\p1.png')
im2 = cv.imread('sift\\p2.png')

# 获取各个图像的特征点及sift特征向量
# 返回值kp包含sift特征的方向、位置、大小等信息
# des的shape为 (sift_num, 128)， sift_num表示图像检测到的sift特征数量

(kp1, des1) = sift.detectAndCompute(im1, None)
(kp2, des2) = sift.detectAndCompute(im2, None)

# 绘制特征点，并显示为红色圆圈
sift_1 = cv.drawKeypoints(im1, kp1, im1, color=(255, 0, 255))
sift_2 = cv.drawKeypoints(im2, kp2, im2, color=(255, 0, 255))

# cv.imshow("2", sift_1);   cv.waitKey(0)
cv.imwrite('result\\sift_1.png', sift_1)
cv.imwrite('result\\sift_2.png', sift_2)



# -------------- 4. 特征点匹配 -----------------
# 特征点匹配
# K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类

#####################################################################
# 查询 cv2.BFMatcher 类的使用，并通过指定不同的match方式 进行特征匹配
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches1 = bf.knnMatch(des1, des2, k=2)
#####################################################################


############################# 调整ratio, 查看拼接结果 #############################
# ratio=0.4：对于准确度要求高的匹配；
# ratio=0.6：对于匹配点数目要求比较多的匹配；
# ratio=0.5：一般情况下。
ratio1 = 0.2
good1 = []
good_distance = []

for m1, n1 in matches1:
    # 如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio1 * n1.distance:
        good1.append([m1])
        good_distance.append(m1)

distances = [match.distance for match in good_distance]
average_distance = np.mean(distances)
print("总距离="+str(distances))
print("平均距离="+str(average_distance))

#############################################################################
# 查询cv2.drawMathcesKnn()方法,可视化输出匹配结果
match_result1 = cv.drawMatchesKnn(im1, kp1, im2, kp2, good1, None, flags=2)

cv.imwrite("result\\sift_1-2.png", match_result1)




























#
























#

import cv2
import numpy as np
import matplotlib.pyplot as plt

##图片组1
#读取原始图像
sift = cv2.SIFT_create()
r1 = cv2.imread('sift/p1.png')


#图像向下取样
r2 = cv2.pyrDown(r1)
r3 = cv2.pyrDown(r2)
r4 = cv2.pyrDown(r3)
r5 = cv2.pyrDown(r4)
# r1 = (r1[:,:,0]).astype(np.double)
# r2 = (r2[:,:,0]).astype(np.double)
# r3 = (r3[:,:,0]).astype(np.double)
# r4 = (r4[:,:,0]).astype(np.double)
# r5 = (r5[:,:,0]).astype(np.double)
sz = 7; sig = 3
im_gs1 = cv2.GaussianBlur(r1, (sz, sz), sig)
im_gs2 = cv2.GaussianBlur(r2, (sz, sz), sig)
im_gs3 = cv2.GaussianBlur(r3, (sz, sz), sig)
im_gs4 = cv2.GaussianBlur(r4, (sz, sz), sig)
im_gs5 = cv2.GaussianBlur(r5, (sz, sz), sig)

im_dog1 = im_gs1 - r1
im_dog2 = im_gs2 - r2
im_dog3 = im_gs3 - r3
im_dog4 = im_gs4 - r4
im_dog5 = im_gs5 - r5

(kp1, des1) = sift.detectAndCompute(r1, None)
(kp2, des2) = sift.detectAndCompute(r2, None)
(kp3, des3) = sift.detectAndCompute(r3, None)
(kp4, des4) = sift.detectAndCompute(r4, None)
(kp5, des5) = sift.detectAndCompute(r5, None)

sift_1 = cv2.drawKeypoints(r1, kp1, r1, color=(255, 0, 0))
sift_2 = cv2.drawKeypoints(r2, kp2, r2, color=(0, 255, 0))
sift_3 = cv2.drawKeypoints(r3, kp3, r3, color=(0, 0, 255))
sift_4 = cv2.drawKeypoints(r4, kp4, r4, color=(255, 0, 255))
sift_5 = cv2.drawKeypoints(r5, kp5, r5, color=(255, 255, 0))


cv2.imwrite("r1.png",r1)
cv2.imwrite("r2.png",r2)
cv2.imwrite("r3.png",r3)
cv2.imwrite("r4.png",r4)
cv2.imwrite("r5.png",r5)


cv2.imwrite("siftres1.png",sift_1)
cv2.imwrite("siftres2.png",sift_2)
cv2.imwrite("siftres3.png",sift_3)
cv2.imwrite("siftres4.png",sift_4)
cv2.imwrite("siftres5.png",sift_5)

cv2.imwrite("gs1.jpg",im_gs1)
cv2.imwrite("gs2.jpg",im_gs2)
cv2.imwrite("gs3.jpg",im_gs3)
cv2.imwrite("gs4.jpg",im_gs4)
cv2.imwrite("gs5.jpg",im_gs5)

cv2.imwrite("dog1.jpg",im_dog1)
cv2.imwrite("dog2.jpg",im_dog2)
cv2.imwrite("dog3.jpg",im_dog3)
cv2.imwrite("dog4.jpg",im_dog4)
cv2.imwrite("dog5.jpg",im_dog5)

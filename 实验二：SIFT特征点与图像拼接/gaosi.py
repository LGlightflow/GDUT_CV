
import cv2
import numpy as np
import matplotlib.pyplot as plt

####这里执行完后注释掉,执行下一个区域
sift = cv2.SIFT_create()
r1 = cv2.imread('sift/p1.png')
r2 = cv2.pyrDown(r1)
r3 = cv2.pyrDown(r2)
r4 = cv2.pyrDown(r3)
r5 = cv2.pyrDown(r4)
r1 = (r1[:,:,0]).astype(np.double)
r2 = (r2[:,:,0]).astype(np.double)
r3 = (r3[:,:,0]).astype(np.double)
r4 = (r4[:,:,0]).astype(np.double)
r5 = (r5[:,:,0]).astype(np.double)
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
cv2.imwrite("r1.png",r1)
cv2.imwrite("r2.png",r2)
cv2.imwrite("r3.png",r3)
cv2.imwrite("r4.png",r4)
cv2.imwrite("r5.png",r5)

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
####这里执行完后注释掉,执行下一个区域




import cv2
import numpy as np
LENGTH_REPEAT = 20
HEIGHT_REPEAT = 20
img_src = "c:/users/ssk/desktop/a.png"


img = cv2.imread(img_src)

img_return = np.zeros(
    (img.shape[0]*LENGTH_REPEAT, img.shape[1]*HEIGHT_REPEAT, img.shape[2]))

for i in range(LENGTH_REPEAT):
    for j in range(HEIGHT_REPEAT):
        img_return[i*img.shape[0]:(i+1)*img.shape[0],
                   j*img.shape[1]:(j+1)*img.shape[1], :] = img

cv2.imwrite("c:/users/ssk/desktop/a.jpg", img_return)

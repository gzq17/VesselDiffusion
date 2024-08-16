import numpy as np
import os
import cv2
from skimage.morphology import skeletonize


def one_resize(lbl, sz=336):
    if ((lbl != 255) & (lbl != 0)).sum() != 0:
        print('error ' * 5)
    lbl_thin = np.zeros(lbl.shape)
    lbl_thin[lbl == 255] = 1
    lbl_thin = skeletonize(lbl_thin)
    index = np.where(lbl_thin == 1)
    index_arr = np.concatenate([np.array(index[0])[:, np.newaxis],
                                np.array(index[1])[:, np.newaxis]], axis=1)

    lbl_new = cv2.resize(lbl, (sz, sz), interpolation=0)
    for i in range(0, index_arr.shape[0]):
        xx, yy = index_arr[i][0], index_arr[i][1]
        xx, yy = xx * sz / lbl.shape[0], yy * sz / lbl.shape[1]
        xx, yy = round(xx), round(yy)
        if xx == sz:
            xx = sz - 1
        if yy == sz:
            yy = sz - 1
        lbl_new[xx, yy] = 255
    return lbl_new

def check_img_size():
    img_path = './data/pretrain_data/img/'
    lbl_path = './data/pretrain_data/lbl/'
    img_list = sorted(os.listdir(img_path))
    sz = 336
    for name in img_list:
        img_name = img_path + name
        lbl_name = lbl_path + name.replace('.png', '_label.png')
        img = cv2.imread(img_name, 0)
        lbl = cv2.imread(lbl_name, 0)
        if ((lbl != 255) & (lbl != 0)).sum() != 0:
            print('error ' * 5)
        if img.shape[0] != sz or img.shape[1] != sz:
            print(name)
            img_new = cv2.resize(img, (sz, sz))
            lbl_new = one_resize(lbl, sz=sz)
        else:
            continue
        cv2.imwrite(img_name, img_new)
        cv2.imwrite(lbl_name, lbl_new)


if __name__ == '__main__':
    check_img_size()

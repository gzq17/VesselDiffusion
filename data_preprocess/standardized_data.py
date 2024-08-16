import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage import interpolation
import copy
import shutil


def re_sample(image, spacing, new_spacing=np.array([0.4, 0.4, 0.48]), order=3):
    resize_factor = spacing / new_spacing
    temp = resize_factor[2]
    resize_factor[2] = resize_factor[0]
    resize_factor[0] = temp
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    image = interpolation.zoom(image, real_resize_factor, mode='nearest', order=order)
    temp = real_resize_factor[2]
    real_resize_factor[2] = real_resize_factor[0]
    real_resize_factor[0] = temp
    new_spacing = spacing / real_resize_factor
    return image, new_spacing


def coronary_size_re():
    name_path = './data/original_data/'
    out_path = './data/same_spacing/'
    os.makedirs(out_path, exist_ok=True)
    name_list = sorted(os.listdir(name_path), reverse=True)
    for name in name_list:
        print(name)
        out_name = out_path + name
        if '.nrrd' in name:
            out_name = out_path + name.replace('.nrrd', '.nii.gz')
        img_name = name_path + name
        img_ = sitk.ReadImage(img_name)
        img = sitk.GetArrayFromImage(img_)
        new_img, new_spacing = re_sample(img, np.array(img_.GetSpacing()), order=0)
        new_img[new_img > 0.5] = 1
        new_img[new_img <= 0.5] = 0
        print(new_img.shape, new_spacing)
        new_img = sitk.GetImageFromArray(new_img)
        new_img.SetSpacing(new_spacing)
        sitk.WriteImage(new_img, out_name)


def same_sz():
    result_path = './data/same_spacing/'
    out_path = './data/same_sz_data/'
    os.makedirs(out_path, exist_ok=True)
    img_list = sorted(os.listdir(result_path))
    sz = [256, 336, 336]
    num_list = []
    for name in img_list:
        print(name)
        img_name = result_path + name
        out_name = out_path + name
        img_ = sitk.ReadImage(img_name)
        img = sitk.GetArrayFromImage(img_)
        index = np.where(img == 1)
        new_img = copy.deepcopy(img)
        add_cut = []
        for i in range(0, 3):
            if img.shape[i] < sz[i]:
                ii = (sz[i] - img.shape[i]) // 2
                jj = sz[i] - img.shape[i] - ii
                add_cut.append([True, ii, jj])
            else:
                center = (index[i].max() + index[i].min()) // 2
                y_b, y_e = center - sz[i] // 2, center + sz[i] // 2
                if y_b < 0:
                    y_b = 0
                    y_e = sz[i]
                if y_e > img.shape[i]:
                    y_e = img.shape[i]
                    y_b = y_e - sz[i]
                add_cut.append([False, y_b, y_e])
        if add_cut[0][0]:
            if add_cut[0][1] != 0:
                one_slicer = np.zeros((add_cut[0][1], new_img.shape[1], new_img.shape[2]))
                new_img = np.concatenate([one_slicer, new_img], axis=0)
            if add_cut[0][2] != 0:
                one_slicer = np.zeros((add_cut[0][2], new_img.shape[1], new_img.shape[2]))
                new_img = np.concatenate([new_img, one_slicer], axis=0)
        else:
            new_img = new_img[add_cut[0][1]: add_cut[0][2], :, :]
        if add_cut[1][0]:
            if add_cut[1][1] != 0:
                one_slicer = np.zeros((new_img.shape[0], add_cut[1][1], new_img.shape[2]))
                new_img = np.concatenate([one_slicer, new_img], axis=1)
            if add_cut[1][2] != 0:
                one_slicer = np.zeros((new_img.shape[0], add_cut[1][2], new_img.shape[2]))
                new_img = np.concatenate([new_img, one_slicer], axis=1)
        else:
            new_img = new_img[:, add_cut[1][1]: add_cut[1][2], :]
        if add_cut[2][0]:
            if add_cut[2][1] != 0:
                one_slicer = np.zeros((new_img.shape[0], new_img.shape[1], add_cut[2][1]))
                new_img = np.concatenate([one_slicer, new_img], axis=2)
            if add_cut[2][2] != 0:
                one_slicer = np.zeros((new_img.shape[0], new_img.shape[1], add_cut[2][2]))
                new_img = np.concatenate([new_img, one_slicer], axis=2)
        else:
            new_img = new_img[:, :, add_cut[2][1]: add_cut[2][2]]
        print(new_img.shape, new_img.sum())
        num_list.append(new_img.sum())
        new_img_ = sitk.GetImageFromArray(new_img)
        new_img_.SetSpacing(img_.GetSpacing())
        sitk.WriteImage(new_img_, out_name)
    
    if os.path.exists(result_path):
        shutil.rmtree(result_path)

if __name__ == '__main__':
    coronary_size_re()
    same_sz()
    

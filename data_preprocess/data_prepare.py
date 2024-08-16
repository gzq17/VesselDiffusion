import copy
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
import numpy as np
import open3d as o3d
import os
import cv2


def farthest_point_sampling(arr, M):
    N = arr.shape[0]
    distances = np.full(N, np.inf)
    farthest_pts = np.zeros((M, 3))
    farthest_pts[0] = arr[np.random.randint(len(arr))]
    for i in range(1, M):
        new_dist = np.sum((arr - farthest_pts[i-1])**2, axis=1)
        distances = np.minimum(distances, new_dist)
        farthest_pts[i] = arr[np.argmax(distances)]
    return farthest_pts


def bin2point():
    result_path = './data/same_sz_data/'
    out_path = './data/coronary3d_point/'
    img_list = sorted(os.listdir(result_path))
    N = len(img_list)
    num_list = [2048, 4096]
    np.random.seed(42)
    sz = [256.0, 336.0, 336.0]
    npy_arr = [np.zeros((N, 2048, 3)), np.zeros((N, 4096, 3))]
    kk = 0
    os.makedirs(out_path, exist_ok=True)
    for name in img_list:
        # if name != '013.nii.gz':
        #     continue
        if '.nii.gz' not in name:
            continue
        print(name)
        img_name = result_path + name
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        img_dist = distance_transform_edt(img)
        binary_img = (img_dist < 1.2) & (img_dist > 0.5)
        binary_img[binary_img] = 1.0
        # mip_img = np.max(img, 0)
        # print(mip_img[82, 57])
        # print(binary_img[:, 82, 57].sum())
        bin_index = np.where(binary_img == 1)
        print(((binary_img == 1.0) & (img == 0)).sum())
        point_arr = np.concatenate([np.array(bin_index[0])[:, np.newaxis],
                                        np.array(bin_index[1])[:, np.newaxis],
                                        np.array(bin_index[2])[:, np.newaxis]], axis=1)
        point_arr = point_arr.astype('float')
        print(point_arr.shape)
        for ii in range(point_arr.shape[1]):
            point_arr[:, ii] = (point_arr[:, ii] - sz[ii] / 2) / (sz[ii] / 2)
        print(point_arr.shape, point_arr.max(), point_arr.min())
        for ii in range(len(num_list)):
            out_name = out_path + name[:-7] + '_' + str(num_list[ii]) + '.pcd'
            arr_res = farthest_point_sampling(point_arr, num_list[ii])
            print(arr_res.shape, arr_res.max(), arr_res.min())
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(arr_res)
            o3d.io.write_point_cloud(out_name, pcd)
            npy_arr[ii][kk] = arr_res
        kk += 1
    np.save(out_path + '0000_2048.npy', npy_arr[0])
    np.save(out_path + '0000_4096.npy', npy_arr[1])


def coronary3d2mip():
    result_path = './data/same_sz_data/'
    mip_img_path = './data/same_mip_all/'
    os.makedirs(mip_img_path, exist_ok=True)
    name_list = sorted(os.listdir(result_path))
    for name in name_list:
        img_name = result_path + name
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        mip_img = np.max(img, 0)
        print(name, mip_img.shape)
        cv2.imwrite(mip_img_path + name.replace('.nii.gz', '.png'), mip_img * 255.0)


if __name__ == '__main__':
    bin2point()
    coronary3d2mip()

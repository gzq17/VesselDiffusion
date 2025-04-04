import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import mitsuba as mi
from PIL import Image
import Imath
import os
import OpenEXR
import time
import argparse
import copy
import cv2
import json
root_path = ''


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
    
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
    
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    
    """

#0.025 0.015
xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="{}"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
    
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def raoate_pcl(pcl, angle_x, angle_y, angle_z):
    rotation = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
    rotated_points = rotation.apply(pcl)
    return rotated_points


def gene_render(name, angle_x=0, angle_y=0, angle_z=0, xml_path=None, input_pcd=None, ball_r=0.015):
    if name[-4:] == '.npy':
        pcl_r = np.load(root_path + name)
        if pcl_r.shape[1] == 6:
            pcl_r = pcl_r[:, :3]
    elif name[-4:] == '.pcd' or name[-4:] == '.ply':
        point_cloud = o3d.io.read_point_cloud(root_path + name)
        pcl_r = np.asarray(point_cloud.points)
    elif name == 'kkkk':
        pcl_r = copy.deepcopy(input_pcd)
    else:
        return
    print(pcl_r.shape)
    pcl_r[:, 1] = -1 * pcl_r[:, 1]
    xml_segments = [xml_head]
    if angle_x != 0 or angle_y != 0 or angle_z != 0:
        pcl_r = raoate_pcl(pcl_r, angle_x, angle_y, angle_z)
    # pcl = standardize_bbox(pcl_r, pcl_r.shape[0])
    pcl = standardize_bbox(pcl_r, 2048)
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        xml_segments.append(xml_ball_segment.format(ball_r, pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)
    name_bb = name[:-4] + f"_{str(angle_x).zfill(3)}_{str(angle_y).zfill(3)}_{str(angle_z).zfill(3)}"
    if xml_path is None:
        xml_path = root_path + name_bb + ".xml"
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcl_r)
    # o3d.io.write_point_cloud(root_path + name_bb + '.ply', pcd)

def mi_show(xml_path=None, out_path=None):
    if xml_path is None:
        xml_path = root_path + 'img_0007_045_045_000.xml'
    if out_path is None:
        out_path = root_path + 'img_0007_045_045_000.exr'
    mi.set_variant('scalar_rgb')#scalar_rgb cuda_ad_rgb
    scene = mi.load_dict(mi.cornell_box())
    scene = mi.load_file(xml_path)
    # Render the scene
    img = mi.render(scene)
    # Write the rendered image to an EXR file
    mi.Bitmap(img).write(out_path)

def exr2png():
    exr_path = root_path + 'img_0007_045_045_000.exr'
    mi.set_variant('scalar_rgb')
    bitmap = mi.Bitmap(exr_path)

    # Convert Bitmap to numpy array
    data = np.array(bitmap)
    print(data.min(), data.max())
    data = (data - data.min()) / (data.max() - data.min())

    # Convert to uint8
    data = (data * 255).astype(np.uint8)

    # Create a PIL Image from numpy array
    image = Image.fromarray(data)

    # Save as PNG file
    png_path = os.path.join(root_path, 'img_0007_045_045_000.png')
    image.save(png_path)

    print(f"Image saved as {png_path}")

def exr_to_png(exr_path=None, png_path=None):
    # Open the .exr file
    if exr_path is None:
        exr_path = root_path + 'img_0007_045_045_000.exr'
    if png_path is None:
        png_path = root_path + 'img_0007_045_045_000.png'
    exr_file = OpenEXR.InputFile(exr_path)

    # Get the header information from the .exr file
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Define the pixel type and channel type
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Read the RGB channels
    rgb_channels = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in 'RGB']
    
    # Reshape and normalize the RGB channels
    rgb = [np.reshape(channel, (size[1], size[0])) for channel in rgb_channels]
    rgb = np.dstack(rgb)
    rgb = np.clip(rgb, 0, 1)  # Clip values to the range [0, 1]
    rgb = (rgb * 255).astype(np.uint8)  # Convert to 8-bit per channel

    # Create and save the image
    image = Image.fromarray(rgb)
    image.save(png_path)

def visualization():
    json_name = './data_angle_record.json'
    root_path = ''
    ply_path = root_path + '3D_model/'
    choose_angle_path = root_path + 'choose_final/'
    os.makedirs(choose_angle_path, exist_ok=True)
    with open(json_name) as f:
        description_result = json.load(f)
    angle_dir = {}
    for cls_name in description_result.keys():
        for obj_name in description_result[cls_name].keys():
            angle_dir[obj_name] = [description_result[cls_name][obj_name], cls_name]
    for ply_name in angle_dir.keys():
        print(ply_name)
        ply_cls_name = angle_dir[ply_name][1]
        pcd_name = ply_path + ply_cls_name + '/' + ply_name
        if len(angle_dir[ply_name][0]) == 3:
            print('have done')
            angle_x, angle_y, angle_z = angle_dir[ply_name][0][0], angle_dir[ply_name][0][1], angle_dir[ply_name][0][2]
        else:
            angle_x, angle_y, angle_z = 0, 0, 0
            continue
        print(angle_x, angle_y, angle_z)
        out_path = choose_angle_path + ply_cls_name + '/'
        os.makedirs(out_path, exist_ok=True)
    
        pcd = o3d.io.read_point_cloud(pcd_name)
        pcd_shape = np.asarray(pcd.points)
        pcd_color = np.asarray(pcd.colors)
        name = ply_name
    
        ball_r = 0.012
        name_bb = name[:-4] + f"_{str(angle_x).zfill(4)}_{str(angle_y).zfill(4)}_{str(angle_z).zfill(4)}"
        xml_path = out_path + name_bb + ".xml"
        exr_path = out_path + name_bb + '.exr'
        png_path = out_path + name_bb + '.png'
        if os.path.exists(png_path):
            continue
    
        pcl_r = pcd_shape
        pcl_r[:, 1] = -1 * pcl_r[:, 1]
        xml_segments = [xml_head]
        if angle_x != 0 or angle_y != 0 or angle_z != 0:
            pcl_r = raoate_pcl(pcl_r, angle_x, angle_y, angle_z)
        # pcl = standardize_bbox(pcl_r, pcl_r.shape[0])
        mins = np.amin(pcl_r, axis=0)
        maxs = np.amax(pcl_r, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs - mins)
        print("Center: {}, Scale: {}".format(center, scale))
        pcl = ((pcl_r - center) / scale).astype(np.float32)  # [-0.5, 0.5]
        r, g, b = pcd_color[:, 0].mean(), pcd_color[:, 1].mean(), pcd_color[:, 2].mean()
        
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125
        
        for i in range(pcl.shape[0]):
            if '_gray' in name:
                xml_segments.append(xml_ball_segment.format(ball_r, pcl[i, 0], pcl[i, 1], pcl[i, 2], 0.7, 0.7, 0.7))
            else:
                xml_segments.append(xml_ball_segment.format(ball_r, pcl[i, 0], pcl[i, 1], pcl[i, 2], r, g, b))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)
        name_bb = name[:-4] + f"_{str(angle_x).zfill(3)}_{str(angle_y).zfill(3)}_{str(angle_z).zfill(3)}"
        if xml_path is None:
            xml_path = root_path + name_bb + ".xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        
        time.sleep(1.0)
        mi_show(xml_path, exr_path)
        time.sleep(1.0)
        exr_to_png(exr_path, png_path)
        time.sleep(1.0)
        os.remove(exr_path)
        os.remove(xml_path)  

if __name__ == '__main__':
    visualization()

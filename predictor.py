from flask import Flask, send_file
import flask
import os,sys
import json
import logging
import shutil
import torch
import fire
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from functools import partial
from huggingface_hub import snapshot_download
import subprocess
import boto3
import json
import io 
from PIL import Image
import base64

is_local_run = False

code_dir = snapshot_download("One-2-3-45/code", local_dir="/tmp") if not is_local_run else "../code" # , token=os.environ['TOKEN']
print("CODE_DIR",code_dir)
sys.path.append(code_dir)

elev_est_dir = os.path.abspath(os.path.join(code_dir, "one2345_elev_est"))
sys.path.append(elev_est_dir)

if not is_local_run:
    # export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.0;8.6"
    # export IABN_FORCE_CUDA=1
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    os.environ["IABN_FORCE_CUDA"] = "1"
    os.environ["FORCE_CUDA"] = "1"
    subprocess.run(['pip', 'install', 'inplace_abn'])
    # FORCE_CUDA=1 pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    #subprocess.run(['pip', 'install', '--no-cache-dir', 'git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0'])


import cv2
from PIL import Image
import trimesh
import tempfile
from zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from sam_utils import sam_init, sam_out_nosave
from utils import image_preprocess_nosave, gen_poses
from one2345_elev_est.estimate_wild_imgs import estimate_elev
from rembg import remove


#your model artifacts should be stored in /opt/ml/model/ 

device_idx= 0
ckpt='zero123-xl.ckpt'
device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
models = init_model(device, os.path.join(code_dir, ckpt))

# init sam model
predictor = sam_init(device_idx)

with open('instructions_12345.md', 'r') as f:
    article = f.read()

_GPU_INDEX = 0

# The flask app for serving predictions
app = Flask(__name__)

#if __name__ == "__main" :
# app.run(debug=True)

@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly 
    return flask.Response(response= '\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    
    #Process input
    scale=3
    ddim_steps=75
    stage2_steps=50
    preprocess=True
    input_json = flask.request.get_json()
    resp = input_json['input']
    input_im = resp['img']
    input_im = base64.b64decode(input_im)
    if preprocess:
        input_im = preprocess_api(predictor, input_im)
    model = models['turncam'].half()
    # folder to save the stage 1 images
    exp_dir = tempfile.TemporaryDirectory(dir=os.path.join(os.path.dirname(__file__), 'demo_tmp')).name
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = int(estimate_elev(exp_dir))
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)

    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if polar_angle <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)
    result = reconstruct(exp_dir)

    return send_file(
        result,
        as_attachment=True
    )


def reconstruct(exp_dir, output_format=".obj", device_idx=0):

    main_dir_path = os.path.dirname(__file__)
    torch.cuda.empty_cache()
    os.chdir(os.path.join(code_dir, 'SparseNeuS_demo_v1/'))

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"meshes_val_bg/lod0/mesh_00215000_gradio_lod0.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)

def convert_mesh_format(exp_dir, output_format=".obj"):
    ply_path = os.path.join(exp_dir, f"meshes_val_bg/lod0/mesh_00215000_gradio_lod0.ply")
    mesh_path = os.path.join(exp_dir, f"mesh{output_format}")
    mesh = trimesh.load_mesh(ply_path)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]) @ trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    mesh.apply_transform(rotation_matrix)
    mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    mesh.faces = np.fliplr(mesh.faces)
    if output_format == ".obj":
        # Export the mesh as .obj file with colors
        mesh.export(mesh_path, file_type='obj', include_color=True)
    else:
        mesh.export(mesh_path, file_type='glb')
    return mesh_path

def preprocess_api(predictor, raw_im):
    raw_im =  Image.open(io.BytesIO(raw_im))
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_rem = raw_im.convert('RGBA')
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), x_min, y_min, x_max, y_max)
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=False, rescale=True)
    torch.cuda.empty_cache()
    return input_256

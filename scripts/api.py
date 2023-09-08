import hashlib
import os
import platform
import subprocess
import sys
import threading
from glob import glob
from shutil import copyfile

import gradio as gr
import numpy as np
from fastapi import FastAPI
from modules.api import api
from modules.api.models import *
from modules.generation_parameters_copypaste import \
    create_override_settings_dict
from modules.paths import models_path
from PIL import Image
from scripts.easyphoto_config import validation_prompt
from scripts.easyphoto_infer import *
from scripts.easyphoto_train import *


DEFAULT_CACHE_LOG_FILE = "train_kohya_log.txt"
python_executable_path = sys.executable

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def easyphoto_train_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_train_forward")
    def _easyphoto_train_forward_api(
        imgs: dict,
    ):
        id_task                 = imgs.get("id_task", "")
        webui_id                = imgs.get("webui_id", "")
        user_id                 = imgs.get("user_id", "tmp")
        resolution              = imgs.get("resolution", 512)
        val_and_checkpointing_steps = imgs.get("val_and_checkpointing_steps", 800)
        max_train_steps         = imgs.get("max_train_steps", 0)
        steps_per_photos        = imgs.get("steps_per_photos", 200)
        train_batch_size        = imgs.get("train_batch_size", 1)

        gradient_accumulation_steps = imgs.get("gradient_accumulation_steps", 4)
        dataloader_num_workers  = imgs.get("dataloader_num_workers", 16)
        learning_rate           = imgs.get("learning_rate", 1e-4)
        rank                    = imgs.get("rank", 64)
        network_alpha           = imgs.get("network_alpha", 128)
        instance_images         = imgs.get("instance_images", [])
        args                    = imgs.get("args", []) 

        instance_images         = [api.decode_base64_to_image(init_image) for init_image in instance_images]
        message = easyphoto_train_forward(
            id_task,
            webui_id,
            user_id,
            resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos,
            train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, 
            rank, network_alpha,
            instance_images,
            *args
        )
        return {"message": message}
    
def easyphoto_infer_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_infer_forward")
    def _easyphoto_infer_forward_api(
        imgs: dict,
    ):
        webui_id                    = imgs.get("webui_id", "")
        selected_template_images    = imgs.get("selected_template_images", [])
        init_image                  = imgs.get("init_image", None)
        additional_prompt           = imgs.get("additional_prompt", "")
        before_face_fusion_ratio    = imgs.get("before_face_fusion_ratio", 0.50)
        after_face_fusion_ratio     = imgs.get("after_face_fusion_ratio", 0.50)

        first_diffusion_steps       = imgs.get("first_diffusion_steps", 50)
        first_denoising_strength    = imgs.get("first_denoising_strength", 0.45)

        second_diffusion_steps      = imgs.get("second_diffusion_steps", 20)
        second_denoising_strength   = imgs.get("second_denoising_strength", 0.35)
        seed                        = imgs.get("seed", -1)
        crop_face_preprocess        = imgs.get("crop_face_preprocess", True)
        apply_face_fusion_before    = imgs.get("apply_face_fusion_before", 0.5)
        apply_face_fusion_after     = imgs.get("apply_face_fusion_after", 0.5)
        color_shift_middle          = imgs.get("color_shift_middle", True)
        color_shift_last            = imgs.get("color_shift_last", True)
        tabs                        = imgs.get("tabs", 0)

        user_ids                    = imgs.get("user_ids", [])

        selected_template_images    = [api.decode_base64_to_image(_) for _ in selected_template_images]
        init_image                  = None if init_image is None else api.decode_base64_to_image(init_image)
        
        comment, outputs = easyphoto_infer_forward(
            webui_id, selected_template_images, init_image, additional_prompt, \
            before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, \
            seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, tabs, *user_ids
        )
        outputs = [api.encode_pil_to_base64(output) for output in outputs]
        
        return {"message": comment, "outputs": outputs}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(easyphoto_train_forward_api)
    script_callbacks.on_app_started(easyphoto_infer_forward_api)
except:
    pass


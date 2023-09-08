import os
import platform
import subprocess
import threading
from shutil import copyfile

import time
import requests
import json
import sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

from modules import shared
from modules.paths import models_path
from modules.api import api
from scripts.easyphoto_config import get_backend_paths, validation_prompt
from scripts.preprocess import preprocess_images
from scripts.easyphoto_utils import check_files_exists_and_download

DEFAULT_CACHE_LOG_FILE = "train_kohya_log.txt"
python_executable_path = sys.executable

# Attention! Output of js is str or list, not float or int
def easyphoto_train_forward(
    id_task: str,
    webui_id: str,
    user_id: str,
    resolution: int, val_and_checkpointing_steps: int, max_train_steps: int, steps_per_photos: int,
    train_batch_size: int, gradient_accumulation_steps: int, dataloader_num_workers: int, learning_rate: float, 
    rank: int, network_alpha: int,
    instance_images: list,
    *args
):  

    if shared.cmd_opts.just_ui:
        print(instance_images)
        simple_req = dict(
            uid = id_task,
            webui_id = shared.cmd_opts.uid,
            user_id = user_id, 
            resolution = resolution,
            val_and_checkpointing_steps = val_and_checkpointing_steps,
            max_train_steps = max_train_steps,
            steps_per_photos = steps_per_photos,
            train_batch_size = train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,

            dataloader_num_workers = dataloader_num_workers,
            learning_rate = learning_rate,
            rank = rank,
            network_alpha = network_alpha,
            instance_images = [api.encode_pil_to_base64(Image.open(init_image['name'])) for init_image in instance_images],
            args = args
        )
        url = '/'.join([shared.cmd_opts.server_path, 'easyphoto/easyphoto_train_forward'])
        data = requests.post(url, json=simple_req)
        print(data.text)
        comments = json.loads(data.text)['message']
        return comments
    else:
        check_files_exists_and_download()
        easyphoto_img2img_samples, easyphoto_outpath_samples, user_id_outpath_samples, cache_outpath_samples, id_path = get_backend_paths(webui_id)

        if user_id == "" or user_id is None:
            return "User id cannot be set to empty."
        if user_id == "none" :
            return "User id cannot be set to none."
        
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                ids = f.readlines()
            ids = [_id.strip() for _id in ids]
        else:
            ids = []
        if user_id in ids:
            return "User id 不能重复。"
        
        # 模板的地址
        training_templates_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates")
        # 原始数据备份
        original_backup_path    = os.path.join(user_id_outpath_samples, user_id, "original_backup")
        # 人脸的参考备份
        ref_image_path          = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

        # 训练数据保存
        user_path               = os.path.join(user_id_outpath_samples, user_id, "processed_images")
        images_save_path        = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
        json_save_path          = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

        # 训练权重保存
        weights_save_path       = os.path.join(user_id_outpath_samples, user_id, "user_weights")
        webui_save_path         = os.path.join(models_path, f"Lora/{user_id}.safetensors")
        webui_load_path         = os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors")
        sd15_save_path          = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "stable-diffusion-v1-5")
        
        os.makedirs(original_backup_path, exist_ok=True)
        os.makedirs(user_path, exist_ok=True)
        os.makedirs(images_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

        max_train_steps         = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))

        for index, user_image in enumerate(instance_images):
            try:
                image = Image.open(user_image['name']).convert("RGB")
            except:
                image = user_image.convert("RGB")
            image.save(os.path.join(original_backup_path, str(index) + ".jpg"))
            
        sub_threading = threading.Thread(target=preprocess_images, args=(images_save_path, json_save_path, validation_prompt, original_backup_path, ref_image_path,))
        sub_threading.start()
        sub_threading.join()

        train_images = glob(os.path.join(images_save_path, "*.jpg"))
        if len(train_images) == 0:
            return "Failed to obtain preprocessed images, please check the preprocessing process"
        if not os.path.exists(json_save_path):
            return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

        train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
        print("train_file_path : ", train_kohya_path)
        
        # extensions/sd-webui-EasyPhoto/train_kohya_log.txt, use to cache log and flush to UI
        cache_log_file_path = os.path.join(cache_outpath_samples, DEFAULT_CACHE_LOG_FILE)
        print("cache_log_file_path   : ", cache_log_file_path)
        if platform.system() == 'Windows':
            pwd = os.getcwd()
            dataloader_num_workers = 0 # for solve multi process bug
            command = [
                f'{python_executable_path}', '-m', 'accelerate.commands.launch', '--mixed_precision=fp16', "--main_process_port=3456", f'{train_kohya_path}',
                f'--pretrained_model_name_or_path={os.path.relpath(sd15_save_path, pwd)}',
                f'--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}', 
                f'--train_data_dir={os.path.relpath(user_path, pwd)}',
                '--caption_column=text', 
                f'--resolution={resolution}',
                '--random_flip',
                f'--train_batch_size={train_batch_size}',
                f'--gradient_accumulation_steps={gradient_accumulation_steps}',
                f'--dataloader_num_workers={dataloader_num_workers}', 
                f'--max_train_steps={max_train_steps}',
                f'--checkpointing_steps={val_and_checkpointing_steps}', 
                f'--learning_rate={learning_rate}',
                '--lr_scheduler=constant',
                '--lr_warmup_steps=0', 
                '--train_text_encoder', 
                '--seed=42', 
                f'--rank={rank}',
                f'--network_alpha={network_alpha}', 
                f'--validation_prompt={validation_prompt}', 
                f'--validation_steps={val_and_checkpointing_steps}', 
                f'--output_dir={os.path.relpath(weights_save_path, pwd)}', 
                f'--logging_dir={os.path.relpath(weights_save_path, pwd)}', 
                '--enable_xformers_memory_efficient_attention', 
                '--mixed_precision=fp16', 
                f'--template_dir={os.path.relpath(training_templates_path, pwd)}', 
                '--template_mask', 
                '--merge_best_lora_based_face_id', 
                f'--merge_best_lora_name={user_id}',
                f'--cache_log_file={cache_log_file_path}'
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing the command: {e}")
        else:
            os.system(
                f'''
                accelerate launch --mixed_precision="fp16" --main_process_port=3456 {train_kohya_path} \
                --pretrained_model_name_or_path="{sd15_save_path}" \
                --pretrained_model_ckpt="{webui_load_path}" \
                --train_data_dir="{user_path}" --caption_column="text" \
                --resolution={resolution} --random_flip --train_batch_size={train_batch_size} --gradient_accumulation_steps={gradient_accumulation_steps} --dataloader_num_workers={dataloader_num_workers} \
                --max_train_steps={max_train_steps} --checkpointing_steps={val_and_checkpointing_steps} \
                --learning_rate={learning_rate} --lr_scheduler="constant" --lr_warmup_steps=0 \
                --train_text_encoder \
                --seed=42 \
                --rank={rank} --network_alpha={network_alpha} \
                --validation_prompt="{validation_prompt}" \
                --validation_steps={val_and_checkpointing_steps} \
                --output_dir="{weights_save_path}" \
                --logging_dir="{weights_save_path}" \
                --enable_xformers_memory_efficient_attention \
                --mixed_precision='fp16' \
                --template_dir="{training_templates_path}" \
                --template_mask \
                --merge_best_lora_based_face_id \
                --merge_best_lora_name="{user_id}" \
                --cache_log_file="{cache_log_file_path}"
                '''
            )
        
        best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
        if not os.path.exists(best_weight_path):
            return "Failed to obtain Lora after training, please check the training process."

        copyfile(best_weight_path, webui_save_path)
        with open(id_path, "a") as f:
            f.write(f"{user_id}\n")
        return "The training has been completed."
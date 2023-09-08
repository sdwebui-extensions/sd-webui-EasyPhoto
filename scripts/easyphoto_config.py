import os, glob
from modules.paths import data_path
from modules import script_callbacks, shared

# save_dirs
data_dir                        = shared.cmd_opts.data_dir

def get_backend_paths(uuid):
    if uuid is not None:
        easyphoto_img2img_samples       = os.path.join(data_dir, "data-" + str(uuid), 'outputs/img2img-images')
        easyphoto_outpath_samples       = os.path.join(data_dir, "data-" + str(uuid), 'outputs/easyphoto-outputs')
        user_id_outpath_samples         = os.path.join(data_dir, "data-" + str(uuid), 'outputs/easyphoto-user-id-infos')
        cache_outpath_samples           = os.path.join(data_dir, "data-" + str(uuid), 'outputs/easyphoto-cache')
    else:
        easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
        easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
        user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')
        cache_outpath_samples           = os.path.join(data_dir, 'outputs/easyphoto-cache')

    if not os.path.exists(cache_outpath_samples):
        os.makedirs(cache_outpath_samples, exist_ok=True)
    id_path                         = os.path.join(cache_outpath_samples, "ids.txt")

    return easyphoto_img2img_samples, easyphoto_outpath_samples, user_id_outpath_samples, cache_outpath_samples, id_path
    
def get_ui_paths():
    easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
    easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
    user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')
    cache_outpath_samples           = os.path.join(data_dir, 'outputs/easyphoto-cache')
    if not os.path.exists(cache_outpath_samples):
        os.makedirs(cache_outpath_samples, exist_ok=True)
    id_path                         = os.path.join(cache_outpath_samples, "ids.txt")

    return easyphoto_img2img_samples, easyphoto_outpath_samples, user_id_outpath_samples, cache_outpath_samples, id_path

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE    = '(best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE    = '(glasses:1.5), (worst quality:2), (low quality:2), (normal quality:2), over red lips, hair, teeth, lowres, watermark, badhand, ((naked:2, nude:2, nsfw)), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)), mural,'

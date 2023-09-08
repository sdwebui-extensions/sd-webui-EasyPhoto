import os

import gradio as gr
import glob

from scripts.easyphoto_infer import easyphoto_infer_forward
from scripts.easyphoto_config import get_ui_paths
from scripts.easyphoto_train import easyphoto_train_forward, DEFAULT_CACHE_LOG_FILE
from modules import script_callbacks, shared

gradio_compat = True

try:
    from distutils.version import LooseVersion

    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

easyphoto_img2img_samples, easyphoto_outpath_samples, user_id_outpath_samples, cache_outpath_samples, id_path = get_ui_paths()

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""
    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)

    def get_block_name(self):
        return "button"

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

def refresh_display():
    cache_log_file_path = os.path.join(cache_outpath_samples, DEFAULT_CACHE_LOG_FILE)
    lines_limit = 1
    try:
        with open(cache_log_file_path, "r", newline="") as f:
            lines = []
            for s in f.readlines():
                line = s.replace("\x00", "")
                if line.strip() == "" or line.strip() == "\r":
                    continue
                lines.append(line)

            total_lines = len(lines)
            if total_lines <= lines_limit:
                chatbot = [(None, ''.join(lines))]
            else:
                chatbot = [(None, ''.join(lines[total_lines-lines_limit:]))]
            return chatbot
    except Exception:
        with open(cache_log_file_path, "w") as f:
            pass
        return None

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as easyphoto_tabs:
        with gr.TabItem('Train'):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    uuid = gr.Text(label="User_ID", value="", visible=False)
                    webui_id = gr.Text(label="Webui_ID", value="", visible=False)

                    with gr.Column():
                        gr.Markdown('Training photos')

                        instance_images = gr.Gallery().style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row():
                            upload_button = gr.UploadButton(
                                "Upload Photos", file_types=["image"], file_count="multiple"
                            )
                            clear_button = gr.Button("Clear Photos")
                        clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                        upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                        
                    with gr.Column():
                        gr.Markdown('Params Setting')
                        with gr.Accordion("Advanced Options", open=True):
                            with gr.Row():
                                resolution = gr.Textbox(
                                    label="resolution",
                                    value=512,
                                    interactive=True
                                )
                                val_and_checkpointing_steps = gr.Textbox(
                                    label="validation & save steps",
                                    value=100,
                                    interactive=True
                                )
                                max_train_steps = gr.Textbox(
                                    label="max train steps",
                                    value=800,
                                    interactive=True
                                )
                                steps_per_photos = gr.Textbox(
                                    label="max steps per photos",
                                    value=200,
                                    interactive=True
                                )

                            with gr.Row():
                                train_batch_size = gr.Textbox(
                                    label="train batch size",
                                    value=1,
                                    interactive=True
                                )
                                gradient_accumulation_steps = gr.Textbox(
                                    label="gradient accumulationsteps",
                                    value=4,
                                    interactive=True
                                )
                                dataloader_num_workers =  gr.Textbox(
                                    label="dataloader num workers",
                                    value=16,
                                    interactive=True
                                )
                                learning_rate = gr.Textbox(
                                    label="learning rate",
                                    value=1e-4,
                                    interactive=True
                                )
                            with gr.Row():
                                rank = gr.Textbox(
                                    label="rank",
                                    value=128,
                                    interactive=True
                                )
                                network_alpha = gr.Textbox(
                                    label="network alpha",
                                    value=64,
                                    interactive=True
                                )
                        gr.Markdown(
                            '''
                            1. Upload 5 to 10 daily photos of the training required. 
                            2. Click on the Start Training button below to start the training process, approximately 25 minutes.
                            3. Switch to Inference and generate photos based on the template. 

                            Parameter parsing:
                            - **max steps per photo** represents the maximum number of training steps per photo.
                            - **max train steps** represents the maximum training step.
                            - Final training step = Min(photo_num * max_steps_per_photos, max_train_steps)
                            '''
                        )

                with gr.Row():
                    with gr.Column(width=3):
                        run_button = gr.Button('Start Training')
                    with gr.Column(width=1):
                        refresh_button = gr.Button('Refresh Log')

                gr.Markdown(
                    '''
                    We need to train first to predict, please wait for the training to complete, thank you for your patience.  
                    '''
                )
                output_message  = gr.Markdown()

                with gr.Box():
                    logs_out        = gr.Chatbot(label='Training Logs', height=700)
                    block           = gr.Blocks()
                    with block:
                        block.load(refresh_display, None, logs_out, every=1)

                    refresh_button.click(
                        fn = refresh_display,
                        inputs = [],
                        outputs = [logs_out]
                    )

                run_button.click(fn=easyphoto_train_forward,
                                _js="ask_for_style_name",
                                inputs=[
                                    dummy_component,
                                    webui_id,
                                    uuid,
                                    resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, instance_images,
                                ],
                                outputs=[output_message])
                                
        with gr.TabItem('Inference'):
            dummy_component = gr.Label(visible=False)
            webui_id = gr.Text(label="Webui_ID", value="", visible=False)

            training_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'training_templates/*.jpg'))
            infer_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'infer_templates/*.jpg'))
            preset_template = list(training_templates) + list(infer_templates)

            with gr.Blocks() as demo:
                with gr.Row():
                    with gr.Column():
                        model_selected_tab = gr.State(0)

                        with gr.TabItem("template images") as template_images_tab:
                            template_gallery_list = [(i, i) for i in preset_template]
                            gallery = gr.Gallery(template_gallery_list).style(columns=[4], rows=[2], object_fit="contain", height="auto")
                            
                            def select_function(evt: gr.SelectData):
                                return [preset_template[evt.index]]

                            selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                            gallery.select(select_function, None, selected_template_images)
                            
                        with gr.TabItem("upload image") as upload_image_tab:
                            init_image = gr.Image(label="Image for skybox", elem_id="{id_part}_image", show_label=False, source="upload")
                            
                        model_selected_tabs = [template_images_tab, upload_image_tab]
                        for i, tab in enumerate(model_selected_tabs):
                            tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])

                        with gr.Row():
                            def select_function():
                                if os.path.exists(id_path):
                                    with open(id_path, "r") as f:
                                        ids = f.readlines()
                                    ids = [_id.strip() for _id in ids]
                                else:
                                    ids = []
                                return gr.update(choices=["none"] + ids)

                            if os.path.exists(id_path):
                                with open(id_path, "r") as f:
                                    ids = f.readlines()
                                ids = [_id.strip() for _id in ids]
                            else:
                                ids = []

                            uuids           = []
                            num_of_faceid   = shared.opts.data.get("num_of_faceid", 1)
                            for i in range(int(num_of_faceid)):
                                if int(num_of_faceid) > 1:
                                    uuid = gr.Dropdown(value="none", choices=["none"] + ids, label=f"Used_{i} id", visible=True)
                                else:
                                    uuid = gr.Dropdown(value="none", choices=["none"] + ids, label="Used id (The User id you provide while training)", visible=True)

                                uuids.append(uuid)
                            
                            refresh = ToolButton(value="\U0001f504")
                            for i in range(int(num_of_faceid)):
                                refresh.click(
                                    fn=select_function,
                                    inputs=[],
                                    outputs=[uuids[i]]
                                )

                        with gr.Accordion("Advanced Options", open=False):
                            additional_prompt = gr.Textbox(
                                label="Additional Prompt",
                                lines=3,
                                value='masterpiece, beauty',
                                interactive=True
                            )
                            seed = gr.Textbox(
                                label="Seed", 
                                value=12345,
                            )
                            with gr.Row():
                                before_face_fusion_ratio = gr.Slider(
                                    minimum=0.2, maximum=0.8, value=0.50,
                                    step=0.05, label='Face Fusion Ratio Before'
                                )
                                after_face_fusion_ratio = gr.Slider(
                                    minimum=0.2, maximum=0.8, value=0.50,
                                    step=0.05, label='Face Fusion Ratio After'
                                )

                            with gr.Row():
                                first_diffusion_steps = gr.Slider(
                                    minimum=15, maximum=50, value=50,
                                    step=1, label='First Diffusion steps'
                                )
                                first_denoising_strength = gr.Slider(
                                    minimum=0.30, maximum=0.60, value=0.45,
                                    step=0.05, label='First Diffusion denoising strength'
                                )
                            with gr.Row():
                                second_diffusion_steps = gr.Slider(
                                    minimum=15, maximum=50, value=20,
                                    step=1, label='Second Diffusion steps'
                                )
                                second_denoising_strength = gr.Slider(
                                    minimum=0.20, maximum=0.40, value=0.30,
                                    step=0.05, label='Second Diffusion denoising strength'
                                )
                            with gr.Row():
                                crop_face_preprocess = gr.Checkbox(
                                    label="Crop Face Preprocess",  
                                    value=True
                                )
                                apply_face_fusion_before = gr.Checkbox(
                                    label="Apply Face Fusion Before", 
                                    value=True
                                )
                                apply_face_fusion_after = gr.Checkbox(
                                    label="Apply Face Fusion After",  
                                    value=True
                                )
                            with gr.Row():
                                color_shift_middle = gr.Checkbox(
                                    label="Apply color shift first",  
                                    value=True
                                )
                                color_shift_last = gr.Checkbox(
                                    label="Apply color shift last",  
                                    value=True
                                )

                            with gr.Box():
                                gr.Markdown(
                                    '''
                                    Parameter parsing:
                                    1. **Face Fusion Ratio Before** represents the proportion of the first facial fusion, which is higher and more similar to the training object.  
                                    2. **Face Fusion Ratio After** represents the proportion of the second facial fusion, which is higher and more similar to the training object.  
                                    3. **Crop Face Preprocess** represents whether to crop the image before generation, which can adapt to images with smaller faces.  
                                    4. **Apply Face Fusion Before** represents whether to perform the first facial fusion.  
                                    5. **Apply Face Fusion After** represents whether to perform the second facial fusion.  
                                    '''
                                )
                            
                        display_button = gr.Button('Start Generation')

                    with gr.Column():
                        gr.Markdown('Generated Results')

                        output_images = gr.Gallery(
                            label='Output',
                            show_label=False
                        ).style(columns=[4], rows=[2], object_fit="contain", height="auto")
                        infer_progress = gr.Textbox(
                            label="Generation Progress",
                            value="No task currently",
                            interactive=False
                        )
                    
                display_button.click(
                    fn=easyphoto_infer_forward,
                    inputs=[webui_id, selected_template_images, init_image, additional_prompt, 
                            before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, \
                            seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, model_selected_tab, *uuids],
                    outputs=[infer_progress, output_images]
                )
            
    return [(easyphoto_tabs, "EasyPhoto", f"EasyPhoto_tabs")]

# 注册设置页的配置项
def on_ui_settings():
    section = ('EasyPhoto', "EasyPhoto")
    shared.opts.add_option("num_of_faceid", shared.OptionInfo(
        1, "Num of faceid", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}, section=section))

    shared.opts.add_option("easyphoto_cache_model", shared.OptionInfo(
        True, "Cache preprocess model in Inference", gr.Checkbox, {}, section=section))
        
    shared.opts.add_option("EasyPhoto_outpath_samples", shared.OptionInfo(
        easyphoto_outpath_samples, "EasyPhoto output path for image", section=section))

    shared.opts.add_option("EasyPhoto_user_id_outpath", shared.OptionInfo(
        user_id_outpath_samples, "EasyPhoto user id outpath", section=section)) 

script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)

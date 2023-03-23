import modules
import modules.scripts as scripts
import gradio as gr
import os
import torch
from modules import images,paths,ui_common
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

from scripts import lora_compvis, lora_model_util
from scripts.lora_model_util import lora_models
from PIL import Image,ImageDraw,ImageFont

import math
from modules import sd_samplers, shared, script_callbacks, errors
import shutil
# 1girl, tifalockhart, shuimobysim, beautiful girl

style_matrix_enabled = False
cur_model_scheme = "scheme 1"
spawn_x_count = 1
spawn_y_count = 1
current_index = 0
current_models = []
origin_save_files = ui_common.save_files
output_image_files = []


def getwatermarkimage(watermark,digit):

    ones = digit%10
    tens = digit//10

    width = int(watermark.size[0]/10)
    height = int(watermark.size[1])
    ones_image = watermark.crop((ones*width,0,(ones+1)*width,height))
    tens_image = watermark.crop((tens*width,0,(tens+1)*width,height))
    final_image = Image.new('RGBA', size=(width*2, height), color='#00000000')
    final_image.paste(tens_image,(0,0))
    final_image.paste(ones_image,(width,0))

    return final_image


def get_image_weight(col, row, index, scheme):

    if(scheme == "scheme 1"):
        deltaX = 1.0/max(col-1,1)
        deltaY = 1.0/max(row-1,1)
        index_x = index % col
        index_y = int(index // col)
        return [1, index_x*deltaX, 1, index_y*deltaY]
    
    if(scheme == "scheme 2"):
        deltaX = 1.0/max(col-1,1)
        deltaY = 1.0/max(row-1,1)
        index_x = index % col
        index_y = int(index // col)
        print([1-(index_x*deltaX), index_x*deltaX, 1-(index_y*deltaY), index_y*deltaY])
        return [1-(index_x*deltaX), index_x*deltaX, 1-(index_y*deltaY), index_y*deltaY]

def clamp_number(num,a,b):
    return max(min(num, max(a,b)),min(a,b))

def custom_add(a, b):
    c = list(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return type(a)(c)

def image_grid(imgs, batch_size=1, rows=None):
    global spawn_x_count,spawn_y_count,current_index,current_models,cur_model_scheme

    if (len(imgs) == 1):
        cols = 1
        rows = 1
        params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
        script_callbacks.image_grid_callback(params)
        return imgs[0]

    else:
        cols = max(spawn_x_count,1)
        rows = max(spawn_y_count,1)

        watermark = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"digits.png"))
        i = 0
        for img in imgs:
            temp = getwatermarkimage(watermark,i)
            x = int((img.size[0] - temp.size[0])/2)
            y = int((img.size[1] - temp.size[1])/2)
            img.paste(temp,(x,y),temp)
            i+=1

        new_imgs = []
        for i in range(spawn_y_count):
            for j in range(spawn_x_count):
                new_imgs.append(imgs[((spawn_y_count - i - 1)*spawn_x_count) + j])

        imgs = new_imgs
        
    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)

    w, h = imgs[0].size
    totalWidth = params.cols * w + 300
    totalHeight = params.rows * h + 300
    grid = Image.new('RGB', size=(totalWidth, totalHeight), color='black')

    for i, img in enumerate(params.imgs):
        grid.paste(img, box=(i % params.cols*w + 150, i // params.cols*h + 150))

    # ------------------------------------------------------画下标权重------------------------------------------

    draw = ImageDraw.Draw(grid)
    fillColor = "#00FFFF"
    font = ImageFont.truetype("C:\\Windows\\Fonts\\msyhbd.ttc", 30)

    if(cur_model_scheme == "scheme 1"):
        for i in range(spawn_x_count):
            weight = get_image_weight(spawn_x_count, 1, i, cur_model_scheme)
            if(current_models[4] != "None"):
                fillColor = "#0000FF"
                model_path = lora_models.get(current_models[4], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[0])
                draw.text((w*i + (w/2) + 100 - len(content)*10/2, totalHeight - 90),content,font=font,align="center",fill=fillColor)

            if(current_models[5] != "None"):
                fillColor = "#FFFF00"
                model_path = lora_models.get(current_models[5], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[1])
                draw.text((w*i + (w/2) + 100 - len(content)*10/2, totalHeight - 55),content,font=font,align="center",fill=fillColor)

        for i in range(spawn_y_count):
            weight = get_image_weight(1, spawn_y_count, spawn_y_count - i - 1, cur_model_scheme)
            if(current_models[4] != "None"):
                fillColor = "#0000FF"
                model_path = lora_models.get(current_models[4], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[2])
                draw.text((10, h*i + (h/2) + 100 - len(content)*10/2),content,font=font,align="center",fill=fillColor)

            if(current_models[6] != "None"):
                fillColor = "#00FF00"
                model_path = lora_models.get(current_models[6], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[3])
                draw.text((45, h*i + (h/2) + 135 - len(content)*10/2),content,font=font,align="center",fill=fillColor)

    elif(cur_model_scheme == "scheme 2"):
        for i in range(spawn_x_count):
            weight = get_image_weight(spawn_x_count, 1, i, cur_model_scheme)
            if(current_models[0] != "None"):
                fillColor = "#FF0000"
                model_path = lora_models.get(current_models[0], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[0])
                draw.text((w*i + (w/2) + 100 - len(content)*10/2, totalHeight - 90),content,font=font,align="center",fill=fillColor)

            if(current_models[1] != "None"):
                fillColor = "#FFFF00"
                model_path = lora_models.get(current_models[1], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[1])
                draw.text((w*i + (w/2) + 100 - len(content)*10/2, totalHeight - 55),content,font=font,align="center",fill=fillColor)

        for i in range(spawn_y_count):
            weight = get_image_weight(1, spawn_y_count, spawn_y_count - i - 1, cur_model_scheme)
            if(current_models[2] != "None"):
                fillColor = "#0000FF"
                model_path = lora_models.get(current_models[2], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[2])
                draw.text((10, h*i + (h/2) + 100 - len(content)*10/2),content,font=font,align="center",fill=fillColor)

            if(current_models[3] != "None"):
                fillColor = "#00FF00"
                model_path = lora_models.get(current_models[3], None)
                model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
                content = "{:.2f}".format(weight[3])
                draw.text((45, h*i + (h/2) + 135 - len(content)*10/2),content,font=font,align="center",fill=fillColor)

    #leftTitle = leftTitle.rotate(-90,expand = 1)
    #grid.paste(leftTitle, box=(0, 0))

    # ------------------------------------------------------画坐标轴------------------------------------------
    fillColor = "#FFFFFF"
    width = 5
    start = (totalWidth / 100 , totalHeight - 150)
    end = (int(totalWidth * 0.99) , totalHeight - 150)
    draw.line(start+end, fill=fillColor, width=width)

    start = tuple(custom_add(list(end),[-20, -20]))
    draw.line(end+start, fill=fillColor, width=width)
    start = tuple(custom_add(list(end),[-20, +20]))
    draw.line(end+start, fill=fillColor, width=width)

    start = ( 150 , totalHeight / 100)
    end = ( 150 , int(totalHeight * 0.99))
    draw.line(start+end, fill=fillColor, width=width)
    
    end = tuple(custom_add(list(start),[-20, +20]))
    draw.line(start+end, fill=fillColor, width=width)
    end = tuple(custom_add(list(start),[+20, +20]))
    draw.line(start+end, fill=fillColor, width=width)


    font = ImageFont.truetype("C:\\Windows\\Fonts\\msyhbd.ttc", 30)

    if(cur_model_scheme == "scheme 1"):

        fillColor = "#0000FF"
        content = "X0,Y0 "
        if(current_models[4] != "None"):
            model_path = lora_models.get(current_models[4], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (50 , totalHeight - 155)
        draw.text(start,content,font=font,align="center",fill=fillColor)

        fillColor = "#FFFF00"
        content = "X1 "
        if(current_models[5] != "None"):
            model_path = lora_models.get(current_models[5], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (totalWidth - 95 - len(content)*15, totalHeight - 135)
        draw.text(custom_add(start,(0,10)),content,font=font,align="center",fill=fillColor)

        fillColor = "#00FF00"
        content = "Y1 "
        if(current_models[6] != "None"):
            model_path = lora_models.get(current_models[6], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (50 , 100)
        draw.text(start,content,font=font,align="center",fill=fillColor)
    
    elif(cur_model_scheme == "scheme 2"):
        fillColor = "#FF0000"
        content = "X0 "
        if(current_models[0] != "None"):
            model_path = lora_models.get(current_models[0], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (150 , totalHeight - 135)
        draw.text(custom_add(start,(0,10)),content,font=font,align="center",fill=fillColor)
        
        fillColor = "#FFFF00"
        content = "X1 "
        if(current_models[1] != "None"):
            model_path = lora_models.get(current_models[1], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (totalWidth - 95 - len(content)*15, totalHeight - 135)
        draw.text(custom_add(start,(0,10)),content,font=font,align="center",fill=fillColor)

        fillColor = "#0000FF"
        content = "Y0 " 
        if(current_models[2] != "None"):
            model_path = lora_models.get(current_models[2], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (50 , totalHeight - 155)
        draw.text(start,content,font=font,align="center",fill=fillColor)

        fillColor = "#00FF00"
        content = "Y1 "
        if(current_models[3] != "None"):
            model_path = lora_models.get(current_models[3], None)
            model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
            content += f"({model_name})"
        start = (50 , 100)
        draw.text(start,content,font=font,align="center",fill=fillColor)


    # ------------------------------------------------------------------------------------------------

    return grid

def on_before_image_saved(params):
    global output_image_files
    #print("--------------on_before_image_saved-------------")
    #print(params.p)
    #print(params.filename)
    #print(params.pnginfo)
    #print("--------------on_before_image_saved-------------")
    output_image_files.append(params.filename)


def save_files(js_data, images, do_make_zip, index): 
    global style_matrix_enabled
    if(style_matrix_enabled):
        global origin_save_files,output_image_files
        i = 0
        for image in images:
            if(i>0):
                try:
                    imagepath = os.path.abspath(".\\"+output_image_files[i-1])
                    shutil.copyfile(imagepath, image["name"]) # 把无水印的图片拷贝过去覆盖到下载的图片位置
                except BaseException as e:
                    print(f"Error save_files {e}")
            i+=1
    
    return origin_save_files(js_data, images, do_make_zip, index)

class Script(scripts.Script):  

    def __init__(self) -> None:
        super().__init__()
        self.latest_networks = []
        self.origin_image_grid = images.image_grid
        self.custom_processed_images = []

        self.origin_save_files = ui_common.save_files
        ui_common.save_files = save_files

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Output Style Matrix"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        styles = []
        components = []
        models = list(lora_models.keys())
        

        #gr.HTML('<br />')
        with gr.Accordion('Output Style Matrix', open=False):

            with gr.Column():

                with gr.Row():
                    enabled = gr.Checkbox(label='Enable', value=False)
                x_count = gr.Slider(minimum=1, maximum=8, step=1, label="x_count", value=4)
                y_count = gr.Slider(minimum=1, maximum=8, step=1, label="y_count", value=4)
                #batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="stylematrix_batch_size")
                
                with gr.Row():
                    blend_scheme = gr.Dropdown(label="blend scheme", choices=["scheme 1","scheme 2"], value="scheme 1")
                
                with gr.Column(visible=True) as blend_scheme_1:
                    with gr.Row():
                        style05 = gr.Dropdown(label="X0,Y0", choices=models, value="None")
                    with gr.Row():
                        style06 = gr.Dropdown(label="X1", choices=models, value="None")
                        style07 = gr.Dropdown(label="Y1", choices=models, value="None")

                with gr.Column(visible=False) as blend_scheme_2:
                    with gr.Row():
                        style01 = gr.Dropdown(label="X0", choices=models, value="None")
                        #style01.change(lambda module, model : model_util.lora_models.get(model, "None"), inputs=[style01], outputs=[])
                        style02 = gr.Dropdown(label="X1", choices=models, value="None")
                    with gr.Row():
                        style03 = gr.Dropdown(label="Y0", choices=models, value="None")
                        style04 = gr.Dropdown(label="Y1", choices=models, value="None")

                with gr.Row():
                    refresh_models = gr.Button(value='Refresh models')

        # def changeProcessing(enable):
        #     print("change")
        #     from modules.processing import StableDiffusionProcessingTxt2Img
        #     modules.processing.StableDiffusionProcessingTxt2Img = CustomStableDiffusionProcessing if enable else self.origin_SDProcessing
        #     print(modules.processing.StableDiffusionProcessingTxt2Img)

        def on_enabled_changed(enabled):
            global style_matrix_enabled
            style_matrix_enabled = enabled
        enabled.change(fn=on_enabled_changed, inputs = [enabled],outputs = [])

        def gr_show(visible=True):
            return {"visible": visible, "__type__": "update"}

        def update_model_scheme(scheme):
            global cur_model_scheme
            cur_model_scheme = scheme
            return (gr_show(scheme=="scheme 1"),gr_show(scheme=="scheme 2"))

        blend_scheme.change(fn=update_model_scheme, inputs=[blend_scheme], outputs=[blend_scheme_1,blend_scheme_2])
        
        # x_count.change(fn=changeProcessing, inputs = [enabled],outputs = [])
        # y_count.change(fn=changeProcessing, inputs = [enabled],outputs = [])
                    
        def refresh_all_models(*dropdowns):
            print(f"refresh_all_models!!!!")
            
            lora_model_util.update_models()
            updates = []
            
            for i in range(len(dropdowns)):
                dd = dropdowns[i]
                if(i<7):
                    if dd in lora_models:
                        selected = dd
                    else:
                        selected = "None"
                    update = gr.Dropdown.update(value=selected, choices=list(lora_models.keys()))
                    updates.append(update)
                # else:
                #     updates.append(dd)
            return updates

        styles = [style01,style02,style03,style04,style05,style06,style07]
        components = [enabled,x_count,y_count]#,batch_size]

        refresh_models.click(refresh_all_models, inputs=styles, outputs=styles)

        return styles+components

    def run(self, p, components):
        pass
        print("---------------- run\n")
    
    def process(self, p, *args):

        global spawn_x_count,spawn_y_count,current_index,current_models,output_image_files

        # ui_common.save_files = save_files if args[4] == True else self.origin_save_files # Override 保存图片（需要在事件绑定之前替换，所以写在这里没用）

        images.image_grid = image_grid if args[7] == True else self.origin_image_grid
        current_models = args[:7]
        print(f"current_models = {current_models}")
        
        self.custom_processed_images = []
        output_image_files = []

        if args[7] == False:
            return
        
        self.model_weight = [1,0,1,0]
        spawn_x_count = max(args[8],1)
        spawn_y_count = max(args[9],1)
        p.n_iter = spawn_x_count * spawn_y_count
        #modules.shared.state.job_count = spawn_x_count * spawn_y_count
        current_index = 0

        print("---------------- process 开始处理\n")

        #--------------------------------------------------为了修改生成图的数量 把modules.processing.process_images_inner里process函数之前的代码抄了过来------------------------------------------------------------------
        from modules.processing import get_fixed_seed
        from modules import devices
        from modules.sd_hijack import model_hijack
        
        devices.torch_gc()

        seed = get_fixed_seed(p.seed)
        subseed = get_fixed_seed(p.subseed)

        modules.sd_hijack.model_hijack.apply_circular(p.tiling)
        modules.sd_hijack.model_hijack.clear_comments()

        comments = {}

        if type(p.prompt) == list:
            p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(seed) == list:
            p.all_seeds = seed
        else:
            p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

        if type(subseed) == list:
            p.all_subseeds = subseed
        else:
            p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

        def infotext(iteration=0, position_in_batch=0):
            return modules.processingcreate_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

        if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
            model_hijack.embedding_db.load_textual_inversion_embeddings()
        #--------------------------------------------------------------------------------------------------------------------

    def postprocess(self, p, processed, *args):

        if args[7] == False:
            return

        images.image_grid = self.origin_image_grid

        self.custom_processed_images.insert(0, processed.images[0])
        processed.images = self.custom_processed_images
        print("---------------- postprocess 处理完毕\n")
        #print(processed)


    def postprocess_batch(self, p, *args, **kwargs):
        print("---------------- postprocess_batch 单次处理完毕\n")



    def postprocess_image(self, p, pp, *args):
        
        if args[7] == False:
            return

        global spawn_x_count,spawn_y_count,current_index,cur_model_scheme
        print("---------------- PostprocessImageArgs 单张图片生成完毕\n")
        #print(args)
        #print(pp.image)
        #print(p.outpath_samples)
        
        processed_image = Image.new('RGBA', size=pp.image.size, color='black')
        processed_image.paste(pp.image, box=(0,0))
        draw = ImageDraw.Draw(processed_image)
        #fillColor = "#FFFFFF"
        #draw.line((0,0,500,500), fill=fillColor, width=10)
        
        watermark = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"digits.png"))
        watermark = getwatermarkimage(watermark,current_index)
        x = int((processed_image.size[0] - watermark.size[0])/2)
        y = int((processed_image.size[1] - watermark.size[1])/2)
        processed_image.paste(watermark,(x,y),watermark)

        self.custom_processed_images.append(processed_image)


        current_index += 1
        current_index = min(spawn_x_count*spawn_y_count, current_index)
        self.model_weight = get_image_weight(args[8], args[9], current_index, cur_model_scheme)
        #print("-----------------------------------------------------------------------------------------------------------------------\n")
        #print(self.model_weight)
        #print("-----------------------------------------------------------------------------------------------------------------------\n")

    def restore_networks(self, sd_model):
        unet = sd_model.model.diffusion_model
        text_encoder = sd_model.cond_stage_model

        if len(self.latest_networks) > 0:
            #print("restoring last networks")
            for network, _ in self.latest_networks[::-1]:
                network.restore(text_encoder, unet)
            self.latest_networks.clear()

    def process_batch(self, p, *args, **kwargs):
        
        print("-----------process_batch 开始单次处理 " + str(args) + " \n")
        #print(self.model_weight)
        if args[7] == False:
            self.restore_networks(p.sd_model)
            return
        
        # or args[0] == "None"

        self.restore_networks(p.sd_model)

        for i in range(4):

            if(cur_model_scheme == "scheme 1" and i > 3):
                break
            
            model = args[i+3] if cur_model_scheme == "scheme 1" else args[i]
            if(model == "None" or self.model_weight[i] == 0):
                continue

            model_path = lora_models.get(model, None)
            if model_path is None:
                raise RuntimeError(f"model not found: {model}")
            
            unet = p.sd_model.model.diffusion_model
            text_encoder = p.sd_model.cond_stage_model
            
            if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
                model_path = model_path[1:-1]
            if not os.path.exists(model_path):
                print(f"file not found: {model_path}")
                return

            if os.path.splitext(model_path)[1] == '.safetensors':
                from safetensors.torch import load_file
                du_state_dict = load_file(model_path)
            else:
                du_state_dict = torch.load(model_path, map_location='cpu')

            weight_tenc = self.model_weight[i]
            weight_unet = self.model_weight[i]
            network, info = lora_compvis.create_network_and_apply_compvis(du_state_dict, weight_tenc, weight_unet, text_encoder, unet)
            # in medvram, device is different for u-net and sd_model, so use sd_model's
            network.to(p.sd_model.device, dtype=p.sd_model.dtype)

            print(f"LoRA model {model} loaded: {info}")
            self.latest_networks.append((network, model))


# def on_script_unloaded():
#     print("----------------------on_script_unloaded!\n")

# def on_image_grid(ImageGridLoopParams):
#     print("on_image_grid\n")

# script_callbacks.on_script_unloaded(on_script_unloaded)
# script_callbacks.on_image_grid(on_image_grid)
script_callbacks.on_before_image_saved(on_before_image_saved) #
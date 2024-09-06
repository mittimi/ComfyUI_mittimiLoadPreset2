import torch
import toml
import os
import comfy.sd
import folder_paths
import hashlib
from server import PromptServer
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
from comfy.cli_args import args
from nodes import LoraLoader, CLIPTextEncode
from .utils import extractLoras


routes = PromptServer.instance.routes
@routes.post('/mittimi_path')
async def my_function(request):
    the_data = await request.post()
    LoadSetParamMittimi.handle_my_message(the_data)
    return web.json_response({})


my_directory_path = os.path.dirname((os.path.abspath(__file__)))
presets_directory_path = os.path.join(my_directory_path, "presets")
preset_list = [f for f in os.listdir(presets_directory_path) if os.path.isfile(os.path.join(presets_directory_path, f))]
if len(preset_list) > 1: preset_list.sort()

vae_new_list = folder_paths.get_filename_list("vae")
vae_new_list.insert(0,"Use_merged_vae")


class LoadSetParamMittimi:
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (preset_list,),
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "ClipNum": ("INT", {"default": -1, "min": -10, "max": -1} ),
                "vae": (vae_new_list,),
                "PosPromptA": ("STRING", {"multiline": True}),
                "PosPromptB": ("STRING", {"multiline": True}),
                "PosPromptC": ("STRING", {"multiline": True}),
                "NegPromptA": ("STRING", {"multiline": True}),
                "NegPromptB": ("STRING", {"multiline": True}),
                "NegPromptC": ("STRING", {"multiline": True}),
                "Width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "Height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "BatchSize": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "Steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "CFG": ("FLOAT", ),
                "SamplerName": (comfy.samplers.KSampler.SAMPLERS,),
                "Scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "Seed": ("INT", {"default": 1}),
            },
            "hidden": {"node_id": "UNIQUE_ID" }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LATENT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "PDATA", )
    RETURN_NAMES = ("model", "clip", "vae", "positive_prompt", "negative_prompt", "positive_prompt_text", "negative_prompt_text", "Latent", "Steps", "CFG", "sampler_name", "scheduler", "seed", "parameters_data", )
    FUNCTION = "loadAndSettingParameters03"
    CATEGORY = "mittimiTools"

    def loadAndSettingParameters03(self, checkpoint, ClipNum, vae, PosPromptA, PosPromptB, PosPromptC, NegPromptA, NegPromptB, NegPromptC, Width, Height, BatchSize, Steps, CFG, SamplerName, Scheduler, Seed, node_id, preset=[], ):

        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        out3 = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        re_ckpt = out3[0]
        re_vae = out3[2]
        
        sha256_hash = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        model_hash = sha256_hash.hexdigest()[:10]
        
        re_clip = out3[1].clone()
        re_clip.clip_layer(ClipNum)
        
        if (vae != "Use_merged_vae"):
            vae_path = folder_paths.get_full_path("vae", vae)
            sd = comfy.utils.load_torch_file(vae_path)
            re_vae = comfy.sd.VAE(sd=sd)
            
        Latent = torch.zeros([BatchSize, 4, Height // 8, Width // 8], device=self.device)
        
        poscombtxt = PosPromptA + PosPromptB + PosPromptC
        negcombtxt = NegPromptA + NegPromptB + NegPromptC
        
        poslora = []
        postxt, poslora = extractLoras(poscombtxt)
        
        if len(poslora):
            for lora in poslora:
                re_ckpt, re_clip = LoraLoader().load_lora(re_ckpt, re_clip, lora['lora'], lora['strength'], lora['strength'])

        postokens = re_clip.tokenize(postxt)
        negtokens = re_clip.tokenize(negcombtxt)
        posoutput = re_clip.encode_from_tokens(postokens, return_pooled=True, return_dict=True)
        negoutput = re_clip.encode_from_tokens(negtokens, return_pooled=True, return_dict=True)
        poscond = posoutput.pop("cond")
        negcond = negoutput.pop("cond")
        
        parameters_data = []
        parameters_data.append( {'posp':poscombtxt, 'negp':negcombtxt, 'step':Steps, 'sampler':SamplerName, 'scheduler':Scheduler, 'cfg':CFG, 'seed':Seed, 'width':Width, 'height':Height, 'hash':model_hash, 'checkpoint':checkpoint, 'clip':ClipNum, 'vae':vae} )

        return(re_ckpt, re_clip, re_vae, [[poscond, posoutput]], [[negcond, negoutput]], postxt, negcombtxt, {"samples":Latent}, Steps, CFG, SamplerName, Scheduler, Seed, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r') as f:
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})
        

class CombineParamDataMittimi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "param1": ("PDATA", ),
            },
            "optional": {
                "param2": ("PDATA", ),
            },
        }
    RETURN_TYPES = ("PDATA", )
    RETURN_NAMES = ("parameters_data", )
    FUNCTION = "combineparamdataMittimi"
    CATEGORY = "mittimiTools"

    def combineparamdataMittimi(self, param1, param2=None, ):

        return_param = param1[:]
        if param2:
            return_param += param2[:]
        
        return(return_param, )
    

class SaveImageParamMittimi:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "parameters_data": ("PDATA", ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveimageMittimi"
    CATEGORY = "mittimiTools"
    
    def saveimageMittimi(self, images, filename_prefix, parameters_data=None, prompt=None, extra_pnginfo=None, ):
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        seed_counter = 0
        bseed = 0
        if parameters_data: bseed = parameters_data[0]['seed'] if "seed" in parameters_data[0] else 0

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if parameters_data is not None:
                    
                    parameters_text = ""
                    param_counter = 0
                    for pd in parameters_data:
                        if param_counter > 0: parameters_text += "\n\n"
                        parameters_text += f"{pd['posp']}\nNegative prompt: {pd['negp']}\nSteps: {pd['step']}, Sampler: {pd['sampler']}, Scheduler: {pd['scheduler']}, CFG Scale: {pd['cfg']}, Seed: {pd['seed']+seed_counter}, Size: {pd['width']}x{pd['height']}, Model hash: {pd['hash']}, Model: {pd['checkpoint']}, Clip: {pd['clip']}, VAE: {pd['vae']}"
                        param_counter += 1
                    parameters_text += ", Version: ComfyUI"
                    metadata.add_text("parameters", parameters_text)
                    
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_{seed_counter + bseed}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            seed_counter += 1

        return { "ui": { "images": results } }


NODE_CLASS_MAPPINGS = {
    "LoadSetParamMittimi": LoadSetParamMittimi,
    "SaveImageParamMittimi": SaveImageParamMittimi,
    "CombineParamDataMittimi": CombineParamDataMittimi,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSetParamMittimi": "LoadSetParameters",
    "SaveImageParamMittimi": "SaveImageWithParamText",
    "CombineParamDataMittimi": "CombineParamData",
}
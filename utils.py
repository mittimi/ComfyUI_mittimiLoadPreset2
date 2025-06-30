import os
import folder_paths
import glob
import re
import comfy.lora
import numpy as np
import importlib.util

is_negpip = False
negpipfile_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-ppm/clip_negpip.py")
try:
    spec = importlib.util.spec_from_file_location("clip_negpip", negpipfile_path)
    negpip_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(negpip_module)
    is_negpip = True
except:
    print("No Negpip.")

def runNegpip(self, ckpt, clip):

    if not is_negpip:
        return ckpt, clip
    
    re_ckpt, re_clip = negpip_module.CLIPNegPip.patch(self, ckpt, clip)
    
    return re_ckpt, re_clip

def getLbwPresets():

    presets = {}
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "ComfyUI-Inspire-Pack/resources/lbw-preset.txt"),
        os.path.join(os.path.dirname(__file__), "..", "ComfyUI-Inspire-Pack/resources/lbw-preset.custom.txt"),
        os.path.join(os.path.dirname(__file__), "resources/lbw-preset.txt"),
    ]

    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as file:
                for line in file:
                    l = line.split(':')
                    if len(l) > 1:
                        if l[0] not in presets:
                            presets[l[0]] = l[1]
        else:
            print("")

    return presets

def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None

def validate(vectors):
    
    if len(vectors) < 12:
        return False
    
    for x in vectors:
        if x in ['R', 'r', 'U', 'u'] or is_numeric_string(x):
            continue
        else:
            subvectors = x.strip().split(' ')
            for y in subvectors:
                y = y.strip()
                if y not in ['R', 'r', 'U', 'u'] and not is_numeric_string(y):
                    return False

    return True

def convert_vector_value(vector_value):

    def simple_vector(x):
        if x in ['U', 'u']:
            ratio = np.random.uniform(-1.5, 1.5)
            ratio = round(ratio, 2)
        elif x in ['R', 'r']:
            ratio = np.random.uniform(0, 3.0)
            ratio = round(ratio, 2)
        elif is_numeric_string(x):
            ratio = float(x)
        else:
            ratio = None

        return ratio

    v = simple_vector(vector_value)
    if v is not None:
        ratios = [v]
    else:
        ratios = [simple_vector(x) for x in vector_value.split(" ")]

    return ratios

def norm_value(value):
    if value == 1:
        return 1
    elif value == 0:
        return 0
    else:
        return value
    
def extractLoras(prompt):
    
    pattern = "<lora:([^:>]*?)(?::(-?\d*(?:\.\d*)?))?(?::lbw=([^:>]*))?(?::(start|stop|step)=(-?\d+(?:-\d+)?))?>"
    loras_folder_path = folder_paths.folder_names_and_paths["loras"][0][0]
    types = folder_paths.folder_names_and_paths["loras"][1]
    lbw_presets = getLbwPresets()
    loras = []
    
    matches = re.findall(pattern, prompt)
    
    for match in matches:
        
        lora_name_noext = ""
        lora_path = ""
        lora_fullpath = ""
        lora_strength = 0
        lbw_vector = []
        sss = ""
        sssparam = ""
        
        lora_strength = float(match[1] if len(match) > 1 and len(match[1]) else 0)

        if lora_strength != 0:
            
            lora_match_name = re.search(r'([^/]+?)(?:\.\w+)?$', match[0])
            if lora_match_name:
                lora_name_noext = lora_match_name.group(1)
                
            lora_fullpath_list = glob.glob(f'{loras_folder_path}/**/{lora_name_noext}.*', recursive=True)
            
            lpaths = []
            for l in lora_fullpath_list:
                split_extension = os.path.splitext(l)
                if len(split_extension) >1:
                    for t in types:
                        if split_extension[1] == t:
                            lpaths.append(l)
                            
            if lpaths:
                lora_fullpath = lpaths[0]
                
                lora_path = os.path.relpath(lora_fullpath, loras_folder_path)
                if lora_path.startswith('/'): lora_path = lora_path[1:]
                
                lbw_match = match[2] if match[2] else ""
                if lbw_match:
                    lbw_vector = lbw_match.split(",")
                    
                    if not validate(lbw_vector):
                        
                        if len(lbw_vector) > 0:
                            vk = lbw_vector[0].strip()
                            if vk in lbw_presets:
                                lbw_vector = lbw_presets[vk].rstrip('\r\n').split(",")
                            else:
                                lbw_vector = []
                        else:
                            lbw_vector = []
                            
                    sss = match[3] if match[3] else ""
                    sssparam = match[4] if match[4] else ""

                loras.append({'name': lora_name_noext, 'path': lora_path, 'fullpath': lora_fullpath, 'strength': lora_strength, 'vector': lbw_vector, 'sss': sss, 'sssparam': sssparam})

    return (re.sub(pattern, '', prompt), loras)

def load_lora_for_models(model, clip, lora_fullpath, strength_model, strength_clip, seed, vector):
    
    lora_torch = None
    lora_torch = comfy.utils.load_torch_file(lora_fullpath, safe_load=True)   
    key_map = comfy.lora.model_lora_keys_unet(model.model)
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    loaded = comfy.lora.load_lora(lora_torch, key_map)
    
    vector_i = 1

    last_k_unet_num = None
    new_modelpatcher = model.clone()
    populated_ratio = strength_model

    def parse_unet_num(s):
        if s[1] == '.':
            return int(s[0])
        else:
            return int(s)

    input_blocks = []
    middle_blocks = []
    output_blocks = []
    others = []
    for k, v in loaded.items():
        k_unet = k[len("diffusion_model."):]

        if k_unet.startswith("input_blocks."):
            k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
            input_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
        elif k_unet.startswith("middle_block."):
            k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
            middle_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
        elif k_unet.startswith("output_blocks."):
            k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
            output_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
        else:
            others.append((k, v, k_unet))
            
    input_blocks = sorted(input_blocks, key=lambda x: x[2])
    middle_blocks = sorted(middle_blocks, key=lambda x: x[2])
    output_blocks = sorted(output_blocks, key=lambda x: x[2])
    
    np.random.seed(seed % (2**31))
    populated_vector_list = []
    ratios = []
    for k, v, k_unet_num, k_unet in (input_blocks + middle_blocks + output_blocks):

        if last_k_unet_num != k_unet_num and len(vector) > vector_i:
            ratios = convert_vector_value(vector[vector_i].strip())
            ratio = ratios.pop(0)
            populated_ratio = ratio
            populated_vector_list.append(norm_value(populated_ratio))
            vector_i += 1

        else:
            if len(ratios) > 0:
                ratio = ratios.pop(0)

            populated_ratio = ratio

        last_k_unet_num = k_unet_num

        new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)

    ratios = convert_vector_value(vector[0].strip())
    ratio = ratios.pop(0)
    populated_ratio = 1

    populated_vector_list.insert(0, norm_value(populated_ratio))

    for k, v, k_unet in others:
        new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)

    new_clip = clip.clone()
    new_clip.add_patches(loaded, strength_clip)
    populated_vector = ','.join(map(str, populated_vector_list))
    return (new_modelpatcher, new_clip, populated_vector)

def getNewTomlnameExt(tomlname, folderpath, savetype):

    tomlnameExt = tomlname + ".toml"
    
    if savetype == "new save":

        filename_list = []
        tmp_list = []
        tmp_list += glob.glob(f"{folderpath}/**/*.toml", recursive=True)
        for l in tmp_list:
            filename_list.append(os.path.relpath(l, folderpath))
        
        duplication_flag = False
        for f in filename_list:
            if tomlnameExt == f:
                duplication_flag = True
                
        if duplication_flag:
            count = 1
            while duplication_flag:
                new_tomlnameExt = f"{tomlname}_{count}.toml"
                if not new_tomlnameExt in filename_list:
                    tomlnameExt = new_tomlnameExt
                    duplication_flag = False
                count += 1
                
    return tomlnameExt

def convert_backslashes_anglebrackets(s):

    result = ''
    inside = False
    temp = ''

    for c in s:
        if c == '<':
            inside = True
            result += c
        elif c == '>':
            inside = False
            result += temp + c
            temp = ''
        elif inside:
            if c == '\\':
                temp += '/'
            else:
                temp += c
        else:
            result += c

    return result
    

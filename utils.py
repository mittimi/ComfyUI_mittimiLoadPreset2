import os
import folder_paths
import glob
import re


def extractLoras(prompt):
    
    pattern = "<lora:([^:>]*?)(?::(-?\d*(?:\.\d*)?))?(?::lbw=([^:>]*))?(?::(start|stop|step)=(-?\d+(?:-\d+)?))?>"
    loras_folder_path = folder_paths.folder_names_and_paths["loras"][0][0]
    types = folder_paths.folder_names_and_paths["loras"][1]
    loras = []

    matches = re.findall(pattern, prompt)
    for match in matches:
        lora_name = ""
        lora_name_g = re.search(r'([^/]+?)(?:\.\w+)?$', match[0])
        if lora_name_g:
            lora_name = lora_name_g.group(1)

        strength = float(match[1] if len(match) > 1 and len(match[1]) else 1.0)
        
        if strength != 0:
            lora_full_path = glob.glob(f'{loras_folder_path}/**/{lora_name}.*', recursive=True)
            lpaths = []
            for l in lora_full_path:
                split_extension = os.path.splitext(l)
                for t in types:
                    if len(split_extension) >1:
                        if split_extension[1] == t:
                            lpaths.append(l)
            if lpaths:
                lpath = lpaths[0]
                lora_path = lpath[len(loras_folder_path):]
                if lora_path.startswith('/'):
                    lora_path = lora_path[1:]

                loras.append({'lora': lora_path, 'strength': strength, 'lbw': "", 'sss': "", 'sssparam': ""})

    return (re.sub(pattern, '', prompt), loras)


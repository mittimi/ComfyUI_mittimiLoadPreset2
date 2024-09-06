# ComfyUI_mittimiLoadPreset2

This node can easily switch between models and prompts by saving presets. Compatible with SD1.5 and XL series.

When using various types of models, isn't it troublesome to reset negative prompts and samplers every time you change models? With this node, you can instantly recall predefined prompts and parameters. Let’s take a look at how to use it.  
　  

## Usage
### 1) Create a Preset
First, create preset data in advance. There are samples in the presets folder within the custom node, so please refer to those.

Presets are saved in TOML format, and the file contains the following 16 items to set:

- **CheckpointName**: Enter the model name, including the file extension. Please make sure to fill out the form.
- **ClipSet**: Value of Clip Skip.
- **VAE**: Enter the VAE name, including the file extension. If you are using the VAE built into the model, write "Use_merged_vae"
- **PositivePromptA**: Enter the positive prompt.
- **PositivePromptB**: Also a positive prompt.
- **PositivePromptC**: Also a positive prompt.
- **NegativePromptA**: Enter the negative prompt.
- **NegativePromptB**: Also a negative prompt.
- **NegativePromptC**: Also a negative prompt.
- **Width**: Image size.
- **Height**: Image size.
- **BatchSize**: Enter the Image batch size.
- **Steps**: Number of steps.
- **CFG**: Enter the CFG value up to the first decimal place.
- **SamplerName**: Name of the sampler.
- **Scheduler**: Name of the scheduler.

If there are items you don't need to use, enter the parameters originally included in the preset sample as they are.

Save the created TOML file in the presets folder.  
　  

### 2) Node Descriptions

Currently, there are three nodes.
- LoadSetParameters
- SaveImageWithParamText
- CombineParamData

Each is described below.  
　  

#### LoadSetParameters node  

![Screenshot of LoadSetParametersNode.](/assets/images/002.jpg)  

"LoadSetParameters" node has many input widgets and output ports, but there is no need to fill or connect them all. Use only the functions you want.  
Let's start from the top of the input and explain where explanations are needed.  

- preset widget  
This is the widget to load the presets described earlier. When a preset is selected, the contents are reflected in each widget.  
The content of each widget is not rewritten unless this is touched, so parameters such as prompt and CFG can be adjusted by directly tinkering with the widget.  

- vae widget  
If you want to use a model with merged VAE, select “Use_marged_vae”.  

- prompt widget  
PosPromptA,B,C are finally combined into a single text, clip text encoded, and output from the positive_primpt port.
Commas are not given when merging, so please be sure to anticipate this in your prompts.
The same specifications apply to negative prompts.
  
- LoRA supports the A1111WebUI method of writing <lora:xxx:1.0>, <lora:xxx.safetensors:1.0>, <lora:folder/xxx:1.0> in the prompt.  
However, if you want to maintain compatibility with A1111WebUI, <lora:xxx:1.0> is the preferred description.  
I would like to support Block Weight in the future, but my understanding has not caught up yet.  
With [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) in place, you can use LoRA's input assistant. This makes writing easier.  

- seed widget  
Seed is set up for saving parameters as described below, and the value is not automatically changed.  
If used, right click and select "Convert Widget to Input" -> "Convert Seed to input" as shown in the image to bring up the input port. Then connect the seed generator node.  
![Screenshot of LoadSetParametersNode.](/assets/images/003.jpg)

- model, clip port  
If LoRA is described in the prompt, the model and clip will be output with LoRA reflected.  
Normally, there is no problem, but if you encounter any inconvenience, do not write LoRA in the prompt, and connect to a LoRA-type node and use it.  

- positive_prompt, negative_prompt port  
The same as above, the output will be the CONDITIONING as reflected in the prompt and model,clip.  
This is often inconvenient, so please make use of the prompt_text port below.  

- positive_prompt_text, negative_prompt_text  
Output prompts in string format. Use this when the encoded form is inconvenient.  

- parameters_data  
Outputs each parameter when saved by A1111WebUI.  
As shown in the image, the parameters are saved when connected to the SaveImageWithParamText node. See the node description below for details.  
　  

#### SaveImageWithParamText node  

![Screenshot of LoadSetParametersNode.](/assets/images/004.jpg)  

This node saves images. The image saving function is equivalent to the official ComfyUI "SaveImage" node.  
The parameters_data widget is optional, and when the parameters_data from LoadSetParameters node is connected, it is stored in the image as metadata.  
　  
#### CombineParamData node  

![Screenshot of LoadSetParametersNode.](/assets/images/005.jpg)  

This node has the ability to combine two parameters_data into one output.  
Use this node when you want to store two parameter_data in an image.  
　  

### 4) For reference  

I use this node as follows.  

![Screenshot of LoadSetParametersNode.](/assets/images/007.jpg)  

Make PromptB independent and set quality system tags to A and C.  
I made width and height input ports to switch the image's height and width size with a single button.  
　  

### 5) Plans for future implementation  
- Block Weight Features.  
　  

### 6) Others  
I’m not a professional, so if there are any bugs, please kindly share how to fix them.  

I also made these nodes as a reference. Respect and thanks to the great author.  

[pythongosssss ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)  
[giriss comfy-image-saver](https://github.com/giriss/comfy-image-saver)  
[ltdrdata ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack/tree/main)  
　  

Autor by [mittimi (https://mittimi.blogspot.com)](https://mittimi.blogspot.com)


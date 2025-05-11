import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var allow_set_flag = true;

app.registerExtension({
	name: "ComfyUI_mittimiLoadPreset2",
    
    async beforeConfigureGraph() {
        allow_set_flag = false;
    },
    
    async nodeCreated(node) {
        
        function send_message(node_id, message) {
            console.log("sendMessage");
            const body = new FormData();
            body.append('message',message);
            body.append('node_id', node_id);
            api.fetchApi("/mittimi_path", { method: "POST", body, });
        }
        
        if (node.comfyClass == "LoadSetParamMittimi") {
            
            Object.defineProperty(node.widgets[0], "value", {
                
                set: (value) => {
                    node._value = value;
                    console.log("set");
                    
                    if (allow_set_flag) send_message(node.id, value);
                },
                get: () => {
                    return node._value;
    			}
    		});
            
            function messageHandler(event) {
                
                if (node.id == event.detail.node) {
                    
                    node.widgets[1].value = (event.detail.message['CheckpointName'])?event.detail.message['CheckpointName']:"no checkpoint";
                    node.widgets[2].value = (event.detail.message['ClipSet'])?event.detail.message['ClipSet']:"-1";
                    node.widgets[3].value = (event.detail.message['VAE'])?event.detail.message['VAE']:"Use_merged_vae";
                    node.widgets[4].value = (event.detail.message['PositivePromptA'])?event.detail.message['PositivePromptA']:"";
                    node.widgets[5].value = (event.detail.message['PositivePromptB'])?event.detail.message['PositivePromptB']:"";
                    node.widgets[6].value = (event.detail.message['PositivePromptC'])?event.detail.message['PositivePromptC']:"";
                    node.widgets[7].value = (event.detail.message['NegativePromptA'])?event.detail.message['NegativePromptA']:"";
                    node.widgets[8].value = (event.detail.message['NegativePromptB'])?event.detail.message['NegativePromptB']:"";
                    node.widgets[9].value = (event.detail.message['NegativePromptC'])?event.detail.message['NegativePromptC']:"";
                    node.widgets[10].value = (event.detail.message['Width'])?event.detail.message['Width']:512;
                    node.widgets[11].value = (event.detail.message['Height'])?event.detail.message['Height']:512;
                    node.widgets[12].value = (event.detail.message['BatchSize'])?event.detail.message['BatchSize']:1;
                    node.widgets[13].value = (event.detail.message['Steps'])?event.detail.message['Steps']:20;
                    node.widgets[14].value = (event.detail.message['CFG'])?event.detail.message['CFG']:7.0;
                    node.widgets[15].value = (event.detail.message['SamplerName'])?event.detail.message['SamplerName']:"euler a";
                    node.widgets[16].value = (event.detail.message['Scheduler'])?event.detail.message['Scheduler']:"normal";
                    node.widgets[17].value = (event.detail.message['Seed'])?event.detail.message['Seed']:1;
                }
            }
            api.addEventListener("my.custom.message", messageHandler);
        }
    },
    
    async afterConfigureGraph() {
        allow_set_flag = true;
    }
});



    

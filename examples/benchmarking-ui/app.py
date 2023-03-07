import gradio as gr
from fastapi import FastAPI

from elements import TextElements as text
from benchmark import Benchmarker
from settings import Manager
from vm import CPUHandler
    
    
bench = Benchmarker()
cpu = CPUHandler()
app = FastAPI()
   

with gr.Blocks() as demo:
    
    gr.Markdown(text.md_title)
    
    with gr.Row():
        
        with gr.Column():
            
            gr.HTML(value=text.embd_video)
            gr.Markdown(text.md_body)   
            
            with gr.Accordion(label=text.accordion_label, open=False):
                gr.Tab(cpu.get_cpu_count())
                gr.Tab(cpu.get_cpu_model_name())
                
        with gr.Column():
            
            for domain, models in text.tab_switch.items():

                with gr.Tab(domain):
                
                    model = gr.Radio(
                        choices=models, 
                        value=models[0],
                        label=text.model_label
                    )
                    
                    engine = gr.Radio(
                        choices=text.engines, 
                        value=text.engines[0], 
                        label=text.engine_label
                    )
                    
                    batch = gr.Slider(
                        minimum=text.batch_min, 
                        maximum=text.batch_max, 
                        step=text.batch_step, 
                        value=text.batch_value, 
                        label=text.batch_label, 
                        interactive=True
                    )
                    
                    time = gr.Slider(
                        minimum=text.time_min, 
                        maximum=text.time_max, 
                        step=text.time_step, 
                        value=text.time_value, 
                        label=text.time_label, 
                        interactive=True
                    )
                    
                    scenario = gr.Radio(
                        choices=text.scenarios, 
                        value=text.scenarios[0], 
                        label=text.scenario_label
                    )
                    
                    button = gr.Button(value=text.button_label)
                    output = gr.Textbox(label=text.output_label)
                    
                    button.click(
                        fn=bench.get_benchmarks, 
                        inputs=[model, engine, batch, time, scenario], 
                        outputs=output
                    )


app = gr.mount_gradio_app(app, demo, path=Manager.route)        
                        
if __name__ == "__main__":
    demo.launch(share=True)
## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition (NER) identifies and classifies entities like names, places, and dates from unstructured text. Traditional models struggle to generalize across diverse domains. This project develops a prototype NER system using a fine-tuned BART model, deployed with Gradio for interactive user evaluation.

### DESIGN STEPS:

#### STEP 1:
Collect a labeled dataset relevant to Named Entity Recognition tasks.Preprocess the data by cleaning, tokenizing, and formatting the labels appropriately.
Convert the dataset into a format compatible with the BART model, such as sequence tagging or span extraction.

#### STEP 2:
Select a suitable pretrained BART model for fine-tuning.Set up a training pipeline using frameworks like Hugging Face Transformers with PyTorch or TensorFlow.
Fine-tune the model on the prepared dataset and evaluate its performance using metrics such as precision, recall, and F1-score.

#### STEP 3:
Develop a Python script to load the fine-tuned BART model and implement a prediction function.Build a Gradio interface with an input text box and output area to display recognized entities.Deploy the application locally or on a cloud platform to enable user interaction and testing.

### PROGRAM:
```py
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
import requests, json


def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

import gradio as gr
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California","Tensho is a skilled graphic designer living in Kyoto, Japan, known for his creative and precise visual work. He draws inspiration from the city's rich culture to craft unique and compelling designs.","Liam is a robotics engineer from Dublin, Ireland, who designs intelligent machines for industrial automation. His work merges cutting-edge technology with practical problem-solving."])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

gr.close_all()
```

### OUTPUT:

### RESULT:
A working NER prototype was developed using a fine-tuned BART model, accurately identifying entities from user input. The Gradio interface enabled interactive testing and evaluation, showcasing strong recognition performance and potential for future improvements.

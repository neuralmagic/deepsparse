# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import pandas as pd
from deepsparse import Pipeline


# flake8: noqa
# fmt: off

# Step 1: Models
model_paths = [
    # ... add paths to other models
]

# Step 2: Loading Parameters
loading_parameters_list = [
    {"engine_type": "deepsparse", "sequence_length": 2048, "prompt_sequence_length": 1, "internal_kv_cache": False},
    {"engine_type": "deepsparse", "sequence_length": 2048, "prompt_sequence_length": 1, "internal_kv_cache": True},
    {"engine_type": "onnxruntime", "sequence_length": 2048, "prompt_sequence_length": 1, "internal_kv_cache": False},
    # ... add other loading parameters
]

# Step 3: Sampling Parameters
sampling_parameters_list = [
    {"deterministic": True, "sampling_temperature": 1.0, "max_tokens": 300},
    {"deterministic": False, "sampling_temperature": 0.8, "max_tokens": 300, "top_p": 0.95, "top_k": 50, "presence_penalty": 1.2},
    {"deterministic": False, "sampling_temperature": 0.7, "max_tokens": 300, "top_p": 0.4, "top_k": 40, "presence_penalty": 1.5},
    # ... add other sampling parameters
]

# Step 4: Prompts
prompts = [
# Instruct then Prompt.
"Complete the following sentence: The sky is ",
# Few Shot Prompt.
"This is awesome! // Positive\nThis is bad! // Negative\nWow that movie was rad! // Positive\nWhat a horrible show! //",
# Explicitly Specify the Instruction
"### Instruction ###\nTranslate the text below to Spanish:\nText: 'hello!'",
# Be Very Specific
"Extract the name of places in the following text.\nDesired format:\nPlace: <comma_separated_list_of_company_names>\nInput: 'Although these developments are encouraging to researchers, much is still a mystery. “We often have a black box between the brain and the effect we see in the periphery,” says Henrique Veiga-Fernandes, a neuroimmunologist at the Champalimaud Centre for the Unknown in Lisbon. “If we want to use it in the therapeutic context, we actually need to understand the mechanism.'",
# Precision
"Explain the concept of deep learning. Keep the explanation short, only a few sentences, and don't be too descriptive.",
# Focus on What LLM Should Do
"The following is an agent that recommends movies to a customer. The agent is responsible to recommend a movie from the top global trending movies. It should refrain from asking users for their preferences and avoid asking for personal information. If the agent doesn't have a movie to recommend, it should respond 'Sorry, couldn't find a movie to recommend today.'.\nCustomer: Please recommend a movie based on my interests.\nAgent:",
# Explain vs. Summarize
"Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body's immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.\nExplain the above in one sentence:",
# Information Extraction
"Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.\nMention the large language model based product mentioned in the paragraph above:",
# Question and Answer
"Answer the question based on the context below. Keep the answer short and concise. Respond 'Unsure about answer' if not sure about the answer.\nContext: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.\nQuestion: What was OKT3 originally sourced from?\nAnswer:",
# Text Classification
"Classify the text into neutral, negative or positive.\nText: I think the vacation is okay.\nSentiment: neutral\nText: I think the food was okay.\nSentiment:",
# Conversation
"The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.\nHuman: Hello, who are you?\nAI: Greeting! I am an AI research assistant. How can I help you today?\nHuman: Can you tell me about the creation of blackholes?\nAI:", "The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students.\nHuman: Hello, who are you?\nAI: Greeting! I am an AI research assistant. How can I help you today?\nHuman: Can you tell me about the creation of black holes?\nAI: ",
# Reasoning
"The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nA: ", 
"The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nSolve by breaking the problem into steps. First, identify the odd numbers, add them, and indicate whether the result is odd or even.",
# Zero Shot, i.e., no examples at all
"Classify the text into neutral, negative or positive.\nText: I think the vacation is okay.\nSentiment:",
# Few Shot, i.e., only a few examples
"The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.\nA: The answer is False.\n\nThe odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.\nA: The answer is True.\n\nThe odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.\nA: The answer is True.\n\nThe odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.\nA: The answer is False.\n\nThe odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nA: ",
# Chain of Thought, i.e., go through a series of rational steps
"The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.\nA: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.\n\nThe odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nA:",
# Zero Shot Chain of Thought, i.e., think step by step, but no examples provided
"I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?\nLet's think step by step.",
]

# Create an empty list to store the results
results = []

# Structured Loop for Evaluations
for model_path, loading_parameters, sampling_parameters in itertools.product(model_paths, loading_parameters_list, sampling_parameters_list):
    
    # Initialize the pipeline with the current combination of model and loading parameters
    pipe = Pipeline.create(
        task="text-generation", 
        model_path=model_path, 
        deterministic=sampling_parameters.get("deterministic", False),
        **loading_parameters
    )

    # We want all the prompts as the inner-loop as we don't need to recompile
    for prompt in prompts:

        # Get the output using the current combination of sampling parameters and prompt
        output = pipe(sequences=prompt, **sampling_parameters)
        
        # Print or store the output to analyze the quality later
        print(f"Model: {model_path}, Loading Parameters: {loading_parameters}, Sampling Parameters: {sampling_parameters}\nPrompt: {prompt}\n====\n")
        print(output.sequences[0])
        print("------")

        # Store the results in a dictionary and append to the results list
        results.append({
            "Model": model_path,
            "Loading Parameters": str(loading_parameters),
            "Sampling Parameters": str(sampling_parameters),
            "Prompt": prompt,
            "Output": output.sequences[0]
        })

# Create a DataFrame from the results
df = pd.DataFrame(results)

output_file = "model_evaluation_results.csv"
print(f"Saving output to {output_file}")
# Save the DataFrame to a CSV file for further analysis
df.to_csv(output_file, index=False)

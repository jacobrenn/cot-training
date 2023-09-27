from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import openai
import torch
import click

DATASET = 'aisquared/cot-ensemble-prompts'
THOUGHT_KEY = 'Thought:'
END_KEY = '### End'
TEST_SIZE = 5000
SEED = 42

def get_model_response(llm, prompt):
    for num_times in range(3):
        response = llm(prompt, max_new_tokens = 100, top_k = 1)[0]['generated_text']
        if END_KEY in response:
            return response.split(END_KEY)[0].strip()
        else:
            prompt = response
    return response

def validate_gpt4(response):
    messages = [
        {
            'role' : 'system',
            'content' : 'You are a validation chatbot which is used to validate the responses of other chatbots in a succinct manner.'
        },
        {
            'role' : 'user',
            'content' : f'The following text contains a question asked to and an answer provided by a chatbot, along with the chatbot\'s chain of thought. I need to know both if the answer that the chatbot provided is correct and if the logic to get to the answer is correct. Please provide ONLY yes or no responses to whether the answer and the content is correct, formatted in the following way:\n\nanswer_correct: yes/no\nlogic_correct: yes/no. Do not provide any other answer besides yes or no\n\nBegin:\n\n{response}\n\nWas the model\'s answer and logic correct?'
        }
    ]
    response = openai.ChatCompletion.create(
        model = 'gpt-4',
        messages = messages,
        temperature = 1
    )
    return response.choices[0]['message']['content']


@click.command()
@click.argument('model-id', type = str)
@click.argument('openai-key-file', type = click.Path(exists = True, file_okay = True, dir_okay = False))
@click.option('--output', '-o', type = click.Path(exists = False, file_okay = True, dir_okay = False), default = './output.csv')
def main(model_id, openai_key_file, output):
    dataset = load_dataset(DATASET)
    dataset = dataset.shuffle(seed = SEED)['train']
    dataset.train_test_split(test_size = TEST_SIZE, seed = SEED)
    dataset = dataset['test']

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,
        device_map = 'auto',
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtype = torch.float16
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast = False
    )
    tokenizer.pad_token = tokenizer.eos_token
    llm = pipeline(
        'text-generation',
        model = model,
        tokenizer = tokenizer
    )

    with open(openai_key_file, 'r') as f:
        openai.api_key = f.read().strip()

    prompt = dataset['prompt'][0].split(THOUGHT_KEY)[0]
    response = get_model_response(llm, prompt)
    gpt4_response = validate_gpt4(response)

    print(prompt)
    print('\n\n')
    print(response)
    print('\n\n')
    print(gpt4_response)

if __name__ == '__main__':
    main()

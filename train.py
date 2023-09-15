from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset, Dataset
from functools import partial
import numpy as np
import torch
import click

DATASET = 'aisquared/cot-ensemble-prompts'
MODEL_ID = 'gpt2-xl'
QUESTION_KEY = 'Question:'
THOUGHT_KEY = 'Thought:'
ANSWER_KEY = 'Final Answer:'
END_KEY = '### End'
SEED = 42
DEFAULT_MAX_LENGTH = 1024

PREFIX = 'Answer the following question to the best of your ability. Be sure to show your chain of thought leading to the final answer.\n\n'

def load_model_and_tokenizer(location):
    model = AutoModelForCausalLM.from_pretrained(
        location,
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        location,
        trust_remote_code = True,
        use_fast = False
    )
    return model, tokenizer

class DataCollatorForCompletionOnly(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        thought_tok_id = self.tokenizer.encode(THOUGHT_KEY)
        labels = batch['labels'].clone()

        for i in range(len(examples)):
            thought_tok_start_idx = None
            for idx in np.where(batch['labels'][i] == thought_tok_id[0])[0]:
                thought_tok_start_idx = idx
                break

            if thought_tok_start_idx:
                labels[i, :thought_tok_start_idx] = -100

        batch['labels'] = labels
        return batch
    
def get_model_and_tokenizer(model_id = MODEL_ID, gradient_checkpointing = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,
        use_cache = False if gradient_checkpointing else True,
        device_map = 'auto',
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtype = torch.float16
        )
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code = True, use_cache = False if gradient_checkpointing else True, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id , use_fast = False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {
            'additional_special_tokens' : [QUESTION_KEY, THOUGHT_KEY, ANSWER_KEY, END_KEY]
        }
    )
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer

def preprocess_batch(batch, tokenizer, max_length = DEFAULT_MAX_LENGTH):
    return tokenizer(
        batch['text'],
        max_length = max_length,
        truncation = True
    )

def preprocess_dataset(tokenizer, max_length, dataset_name = DATASET, seed = SEED):

    dataset = load_dataset(dataset_name)['train']
    _preproc_func = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)

    if dataset_name == DATASET:
        dataset = dataset.to_pandas()

        def create_full_prompt(text):
            return PREFIX + text + '\n\n' + END_KEY
        
        dataset['text'] = dataset.prompt.apply(create_full_prompt)
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.map(
            _preproc_func,
            batched = True,
            remove_columns = ['prompt', 'text']
        )

    else:
        raise ValueError(f'Got unsupported dataset: {dataset_name}')
    
    dataset = dataset.shuffle(seed = seed)
    return dataset

@click.command()
@click.argument('local-output-dir', type = click.Path(exists = False, dir_okay = True, file_okay = False))
@click.option('--epochs', '-e', type = int, default = 3)
@click.option('--train-batch-size', type = int, default = 8)
@click.option('--eval-batch-size', type = int, default = 8)
@click.option('--lr', type = float, default = 1e-5)
@click.option('--seed', type = int, default = SEED)
@click.option('--gradient-checkpointing/--no-gradient-checkpointing', default = True)
@click.option('--cuda/--no-cuda', default = True)
@click.option('--deepspeed', type = click.Path(exists = True, file_okay = True, dir_okay = False), default = None)
@click.option('--test-size', type = int, default = 100)
@click.option('--model-id', type = str, default = MODEL_ID)
@click.option('--local_rank', type = int, default = 0)
@click.option('--fp16/--no-fp16', default = True)
@click.option('--max-length', type = int, default = DEFAULT_MAX_LENGTH)
@click.option('--dataset', type = str, default = DATASET)
@click.option('--load-best/--no-load-best', default = True)
def train(
        local_output_dir,
        epochs,
        train_batch_size,
        eval_batch_size,
        lr,
        seed,
        gradient_checkpointing,
        cuda,
        deepspeed,
        test_size,
        model_id,
        local_rank,
        fp16,
        max_length,
        dataset,
        load_best
):
    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_id = model_id, gradient_checkpointing = gradient_checkpointing)
    
    processed_dataset = preprocess_dataset(tokenizer, max_length = max_length, dataset_name = dataset)
    split_dataset = processed_dataset.train_test_split(test_size = test_size, seed = seed)

    data_collator = DataCollatorForCompletionOnly(
        tokenizer = tokenizer,
        mlm = False,
        return_tensors = 'pt',
        pad_to_multiple_of = 8
    )

    training_args = TrainingArguments(
        output_dir = local_output_dir,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        learning_rate = lr,
        num_train_epochs = epochs,
        gradient_checkpointing = gradient_checkpointing,
        logging_dir = f'{local_output_dir}/runs',
        logging_strategy = 'steps',
        logging_steps = 10,
        evaluation_strategy = 'steps',
        eval_steps = 100,
        save_strategy = 'steps',
        save_steps = 200,
        save_total_limit = None,
        load_best_model_at_end = load_best,
        report_to = 'tensorboard',
        disable_tqdm = False,
        remove_unused_columns = False,
        no_cuda = not cuda,
        deepspeed = deepspeed,
        local_rank = local_rank,
        fp16 = fp16
    )

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = split_dataset['train'],
        eval_dataset = split_dataset['test'],
        data_collator = data_collator
    )
    trainer.train()
    trainer.save_model(local_output_dir)

if __name__ == '__main__':
    train()

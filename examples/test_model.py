from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
from peft import inject_adapter_in_model, LoraConfig

logging.set_verbosity_info()

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""


def init_model(model_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, # "unsloth/tinyllama" for 16bit loading
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    return model, tokenizer

def main():
    model_path = "output_model/model_hf-1.3b_huggingface"
    # model_path = "/data/namtd12/llm_models/internal/hf-1.3b_huggingface"
    
    # Init model
    model, tokenizer = init_model(model_path)
    # Add LoRA adapters
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",],
    #     lora_alpha = 32,
    #     lora_dropout = 0, # Currently only supports dropout = 0
    #     bias = "none",    # Currently only supports bias = "none"
    #     use_gradient_checkpointing = False, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    #     random_state = 3407,
    #     use_rslora = False,  # We support rank stabilized LoRA
    #     loftq_config = None, # And LoftQ
    # )
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Continue the fibonnaci sequence.", # instruction
            "1, 1, 2, 3, 5, 8", # input
            "", # output - leave this blank for generation!
        )
    ]*1, return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    print(tokenizer.batch_decode(outputs))

    
if __name__ == "__main__":
    main()
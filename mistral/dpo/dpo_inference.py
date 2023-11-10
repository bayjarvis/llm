from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
from mistral.dpo.config import Config

if __name__ == '__main__':
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained("Vasanth/openhermes-mistral-dpo-gptq")

    inputs = tokenizer("""I have dropped my phone in water. Now it is not working what should I do now?""", return_tensors="pt").to("cuda")

    model = AutoPeftModelForCausalLM.from_pretrained(
        config.OUTPUT_DIR,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda")
    
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.1,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )

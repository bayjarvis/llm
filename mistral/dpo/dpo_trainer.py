import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, GPTQConfig
from trl import DPOTrainer
from mistral.dpo.config import Config
from mistral.dpo.data_utils import create_dataset
import warnings
warnings.filterwarnings("ignore")

class MistralDPOTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # DPOTrainer requires a triple dataset (prompt, chosen, rejected)
    def create_triple_dataset(self):
        dataset = create_dataset(self.config.DATASET_ID, split='train_prefs')
        df = dataset.to_pandas()
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].sample(1000)
        train_dataset = Dataset.from_pandas(train_df)
        val_df = df[train_size:].sample(200)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = create_dataset(self.config.DATASET_ID, split='test_prefs')
        return train_dataset, val_dataset, test_dataset
    
    def prepare_model(self):
        gptq_config = GPTQConfig(bits=self.config.BITS, disable_exllama=self.config.DISABLE_EXLLAMA)
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, 
                                                     low_cpu_mem_usage=True, 
                                                     quantization_config=gptq_config,
                                                      device_map=self.config.DEVICE_MAP)
        model_ref = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, 
                                                         low_cpu_mem_usage=True, 
                                                         quantization_config=gptq_config,
                                                         device_map=self.config.DEVICE_MAP)
        print("Load model from pretrained checkpoint")
        print(model)

        peft_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.LORA_TARGET_MODULES,
            task_type=self.config.LORA_TASK_TYPE,
            bias=self.config.LORA_BIAS,
            inference_mode=self.config.INFERENCE_MODE)
       
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache=False
        model.gradient_checkpointing_enable()
        model.config.pretraining_tp=1
        model = get_peft_model(model, peft_config)

        print("Load model with LoRA Adapter")
        print(model)
        
        # DPOTrainer requires a reference model
        model_ref = prepare_model_for_kbit_training(model_ref)
        model_ref.config.use_cache=False
        model_ref.gradient_checkpointing_enable()
        model_ref.config.pretraining_tp=1
        model_ref = get_peft_model(model_ref, peft_config)

        print("Load reference model with LoRA Adapter")
        print(model_ref)

        return model, model_ref, peft_config
    
    def set_training_arguments(self):

        '''
        Sets the arguments for the training loop in TrainingArguments class
        '''

        training_arguments = TrainingArguments(
        per_device_train_batch_size=self.config.BATCH_SIZE,
        max_steps=self.config.MAX_STEPS,
        remove_unused_columns=self.config.REMOVE_UNUSED_COLUMNS,
        gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
        learning_rate=self.config.LEARNING_RATE,
        evaluation_strategy=self.config.EVALUATION_STRATEGY,
        logging_first_step=self.config.LOGGING_FIRST_STEP,
        logging_steps=self.config.LOGGING_STEPS,
        output_dir=self.config.OUTPUT_DIR,
        optim=self.config.OPTIM,
        warmup_steps=self.config.WARMUP_STEPS,
        fp16=self.config.FP16,
        push_to_hub=self.config.PUSH_TO_HUB
        )
        return training_arguments

    def train(self):
        train_dataset, val_dataset, test_dataset = self.create_triple_dataset()
        print('triple dataset for DPO', '*'*20)
        print('train_dataset', train_dataset)
        print('val_dataset', val_dataset)
        print('test_dataset', test_dataset)
        print('train_dataset', '*'*20)
        model, model_ref, peft_config = self.prepare_model()

        training_args = self.set_training_arguments()

        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            max_length=256,
            max_target_length=128,
            max_prompt_length=128
        )
        dpo_trainer.train()
        dpo_trainer.push_to_hub("jamesliu23/" + config.OUTPUT_DIR)
 
if __name__ == '__main__':
    config = Config()
    dpo_trainer = MistralDPOTrainer(config)
    dpo_trainer.train()



from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer
from llm.zephyr.finetune_gptq.config import Config 
from llm.zephyr.finetune_gptq.utils import to_chat_text

class ZephyrTrainer:

    def __init__(self, config):

        '''
        A Trainer used to train the Zephyr 7B model which beats Llama2-70b-chat model for your custom usecase

        Initialized:
        config: Parameters required for the trainer to create and process dataset, train and save model finally
        tokenizer: Tokenizer required in training loop
        '''

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_data_sample(self, example):

       processed_example = to_chat_text(example, self.config.INSTRUCTION_FIELD, self.config.TARGET_FIELD)
       return processed_example

    def create_dataset(self):

        '''
        Downloads and processes the dataset

        Returns:
        processed_data: Training ready processed dataset
        '''

        data = load_dataset(self.config.DATASET_ID, split="train")

        print("\n====================================================================\n")
        print("\t\t\tDOWNLOADED DATASET")
        print("\n====================================================================\n")

        df = data.to_pandas()
        df[self.config.DATASET_TEXT_FIELD] = df[[self.config.INSTRUCTION_FIELD, self.config.TARGET_FIELD]].apply(lambda x: self.process_data_sample(x), axis=1)

        print("\n====================================================================\n")
        print("\t\t\tPROCESSED DATASET")
        print(df.iloc[0])
        print("\n====================================================================\n")

        processed_data = Dataset.from_pandas(df[[self.config.DATASET_TEXT_FIELD]])
        return processed_data
    
    def prepare_model(self):

        '''
        Prepares model for finetuning by quantizing it and attaching lora modules to the model

        Returns:
        model - Model ready for finetuning
        peft_config - LoRA Adapter config
        '''

        bnb_config = GPTQConfig(
                                    bits=self.config.BITS,
                                    disable_exllama=self.config.DISABLE_EXLLAMA,
                                    tokenizer=self.tokenizer
                                )

        model = AutoModelForCausalLM.from_pretrained(
                                                        self.config.MODEL_ID,
                                                        quantization_config=bnb_config,
                                                        device_map=self.config.DEVICE_MAP
                                                    )

        print("\n====================================================================\n")
        print("DOWNLOADED MODEL")
        print(model)
        print("\n====================================================================\n")

        model.config.use_cache=self.config.USE_CACHE
        model.config.pretraining_tp=1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        print("\n====================================================================\n")
        print("MODEL CONFIG UPDATED")
        print("\n====================================================================\n")

        peft_config = LoraConfig(
                                    r=self.config.LORA_R,
                                    lora_alpha=self.config.LORA_ALPHA,
                                    lora_dropout=self.config.LORA_DROPOUT,
                                    bias=self.config.BIAS,
                                    task_type=self.config.TASK_TYPE,
                                    target_modules=self.config.TARGET_MODULES
                                )

        model = get_peft_model(model, peft_config)

        print("\n====================================================================\n")
        print("PREPARED MODEL FOR FINETUNING")
        print(model)
        print("\n====================================================================\n")

        return model, peft_config

    def set_training_arguments(self):

        '''
        Sets the arguments for the training loop in TrainingArguments class
        '''

        training_arguments = TrainingArguments(
                                                output_dir=self.config.OUTPUT_DIR,
                                                per_device_train_batch_size=self.config.BATCH_SIZE,
                                                gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
                                                optim=self.config.OPTIMIZER,
                                                learning_rate=self.config.LR,
                                                lr_scheduler_type=self.config.LR_SCHEDULER,
                                                save_strategy=self.config.SAVE_STRATEGY,
                                                logging_steps=self.config.LOGGING_STEPS,
                                                num_train_epochs=self.config.NUM_TRAIN_EPOCHS,
                                                max_steps=self.config.MAX_STEPS,
                                                fp16=self.config.FP16,
                                                push_to_hub=self.config.PUSH_TO_HUB
                                            )

        return training_arguments
    
    def train(self):

        '''
        Trains the model on the specified dataset in config
        '''

        data = self.create_dataset()
        model, peft_config = self.prepare_model()
        training_args = self.set_training_arguments()

        print("\n====================================================================\n")
        print("\t\t\tPREPARED FOR FINETUNING")
        print("\n====================================================================\n")

        trainer = SFTTrainer(
                                model=model,
                                train_dataset=data,
                                peft_config=peft_config,
                                dataset_text_field=self.config.DATASET_TEXT_FIELD,
                                args=training_args,
                                tokenizer=self.tokenizer,
                                packing=self.config.PACKING,
                                max_seq_length=self.config.MAX_SEQ_LENGTH
                            )
        trainer.train()

        print("\n====================================================================\n")
        print("\t\t\tFINETUNING COMPLETED")
        print("\n====================================================================\n")

        trainer.push_to_hub()
     

if __name__ == "__main__":
    config = Config()
    zephyr_trainer = ZephyrTrainer(config)
    zephyr_trainer.train()
     

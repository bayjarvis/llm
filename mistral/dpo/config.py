from pydantic_settings import BaseSettings

class Config(BaseSettings):
    MODEL_ID: str = "TheBloke/OpenHermes-2-Mistral-7B-GPTQ"
    DATASET_ID: str = "HuggingFaceH4/ultrafeedback_binarized"

    # GPTQ config
    BITS:int = 4
    DISABLE_EXLLAMA:bool = True

    # AutoModelForCausalLM config
    DEVICE_MAP:str = "auto"

    # Lora config
    LORA_R: int = 4
    LORA_ALPHA: int = 8
    LORA_DROPOUT: float = 0.1
    LORA_TARGET_MODULES: list = ["q_proj", "v_proj"]
    LORA_TASK_TYPE:str ="CAUSAL_LM"
    LORA_BIAS:str = "none"
    INFERENCE_MODE:bool = False

    # DPOTrainer config
    BATCH_SIZE: int = 1
    MAX_STEPS: int = 50
    REMOVE_UNUSED_COLUMNS: bool = False
    GRAD_ACCUMULATION_STEPS: int = 1
    LEARNING_RATE: float = 3e-4
    EVALUATION_STRATEGY: str = "steps"
    LOGGING_FIRST_STEP: bool = True
    LOGGING_STEPS: int = 10
    OUTPUT_DIR:str = "openhermes-mistral-gptq-dpo"
    OPTIM:str = "paged_adamw_32bit"
    WARMUP_STEPS:int = 2
    FP16:bool = True
    PUSH_TO_HUB:bool = True

    class Config:
        env_prefix = ''  # defaults to no prefix, i.e. ""

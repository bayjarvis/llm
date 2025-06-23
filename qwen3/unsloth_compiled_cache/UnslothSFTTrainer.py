"""
2025.6.1
2025.6.2
4.52.4
0.14.0
__UNSLOTH_VERSIONING__
"""
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from trl.trainer.sft_trainer import (AutoLigerKernelForCausalLM, AutoModelForCausalLM, AutoTokenizer, BaseImageProcessor, Callable, ConstantLengthDataset, DataCollator, DataCollatorForCompletionOnlyLM, DataCollatorForLanguageModeling, Dataset, DatasetGenerationError, EvalPrediction, FeatureExtractionMixin, Optional, PartialState, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTConfig, SFTTrainer, SchemaInferenceError, Trainer, TrainerCallback, Union, dataclasses, datasets, deprecate_kwarg, generate_model_card, get_comet_experiment_url, get_formatting_func_from_dataset, get_peft_model, inspect, is_peft_available, is_wandb_available, nn, os, peft_module_casting_to_bf16, prepare_model_for_kbit_training, torch, wandb, warnings, Callable, ConstantLengthDataset, Dataset, Optional, datasets, os, torch, warnings, Callable, DataCollator, DataCollatorForLanguageModeling, Dataset, Optional, datasets, inspect, os, warnings)


import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
@dataclass
class UnslothSFTConfig(SFTConfig):
    """
    
    Configuration class for the [`SFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the text field of the dataset. If provided, the trainer will automatically create a
            [`ConstantLengthDataset`] based on `dataset_text_field`.
        packing (`bool`, *optional*, defaults to `False`):
            Controls whether the [`ConstantLengthDataset`] packs the sequences of the dataset.
        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of [`~transformers.TrainingArguments`].
        max_seq_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum sequence length for the [`ConstantLengthDataset`] and for automatically creating the dataset. If
            `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset. Only used when `packing=False`.
        dataset_batch_size (`Union[int, None]`, *optional*, defaults to `1000`):
            Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is None`,
            tokenizes the full dataset as a single batch.
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of sequences to use for the [`ConstantLengthDataset`].
        chars_per_token (`float`, *optional*, defaults to `3.6`):
            Number of characters per token to use for the [`ConstantLengthDataset`]. See
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53) for more details.
        use_liger (`bool`, *optional*, defaults to `False`):
            Monkey patch the model with Liger kernels to increase throughput and reduce memory usage.
    
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    def __init__(
        self,
        output_dir = None,
        overwrite_output_dir = None,
        do_train = False,
        do_eval = False,
        do_predict = False,
        eval_strategy = 'no',
        prediction_loss_only = False,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        per_gpu_train_batch_size = None,
        per_gpu_eval_batch_size = None,
        gradient_accumulation_steps = 2,
        eval_accumulation_steps = 2,
        eval_delay = 0,
        torch_empty_cache_steps = 250,
        learning_rate = 5e-05,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        max_grad_norm = 1.0,
        num_train_epochs = 3.0,
        max_steps = -1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.1,
        warmup_steps = 0,
        log_level = 'passive',
        log_level_replica = 'warning',
        log_on_each_node = True,
        logging_dir = None,
        logging_strategy = 'steps',
        logging_first_step = False,
        logging_steps = 1,
        logging_nan_inf_filter = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_total_limit = None,
        save_safetensors = True,
        save_on_each_node = False,
        save_only_model = False,
        restore_callback_states_from_checkpoint = False,
        no_cuda = False,
        use_cpu = False,
        use_mps_device = False,
        seed = 3407,
        data_seed = 3407,
        jit_mode_eval = False,
        use_ipex = False,
        bf16 = False,
        fp16 = False,
        fp16_opt_level = 'O1',
        half_precision_backend = 'auto',
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        local_rank = -1,
        ddp_backend = None,
        tpu_num_cores = None,
        tpu_metrics_debug = False,
        debug = '',
        dataloader_drop_last = False,
        eval_steps = None,
        dataloader_num_workers = 0,
        dataloader_prefetch_factor = None,
        past_index = -1,
        run_name = None,
        disable_tqdm = None,
        remove_unused_columns = True,
        label_names = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        fsdp = '',
        fsdp_min_num_params = 0,
        fsdp_config = None,
        fsdp_transformer_layer_cls_to_wrap = None,
        accelerator_config = None,
        deepspeed = None,
        label_smoothing_factor = 0.0,
        optim = 'adamw_8bit',
        optim_args = None,
        adafactor = False,
        group_by_length = False,
        length_column_name = 'length',
        report_to = None,
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        skip_memory_metrics = True,
        use_legacy_prediction_loop = False,
        push_to_hub = False,
        resume_from_checkpoint = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_token = None,
        hub_private_repo = None,
        hub_always_push = False,
        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = None,
        include_inputs_for_metrics = False,
        eval_do_concat_batches = True,
        fp16_backend = 'auto',
        evaluation_strategy = None,
        push_to_hub_model_id = None,
        push_to_hub_organization = None,
        push_to_hub_token = None,
        mp_parameters = '',
        auto_find_batch_size = False,
        full_determinism = False,
        torchdynamo = None,
        ray_scope = 'last',
        ddp_timeout = 1800,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        dispatch_batches = None,
        split_batches = None,
        include_tokens_per_second = False,
        include_num_input_tokens_seen = False,
        neftune_noise_alpha = None,
        optim_target_modules = None,
        batch_eval_metrics = False,
        eval_on_start = False,
        use_liger_kernel = False,
        eval_use_gather_object = False,
        average_tokens_across_devices = False,
        dataset_text_field = 'text',
        packing = False,
        max_seq_length = None,
        dataset_num_proc = None,
        dataset_batch_size = 1000,
        model_init_kwargs = None,
        dataset_kwargs = None,
        eval_packing = None,
        num_of_sequences = 1024,
        chars_per_token = 3.6,
        use_liger = False,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
        if learning_rate < 1e-7: raise FloatingPointError(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: raise OverflowError(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        if dataset_num_proc is None:
            from multiprocessing import cpu_count
            dataset_num_proc = cpu_count()
        
        super().__init__(
            output_dir = output_dir,
            overwrite_output_dir = overwrite_output_dir,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            eval_strategy = eval_strategy,
            prediction_loss_only = prediction_loss_only,
            per_device_train_batch_size = per_device_train_batch_size,
            per_device_eval_batch_size = per_device_eval_batch_size,
            per_gpu_train_batch_size = per_gpu_train_batch_size,
            per_gpu_eval_batch_size = per_gpu_eval_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            eval_accumulation_steps = eval_accumulation_steps,
            eval_delay = eval_delay,
            torch_empty_cache_steps = torch_empty_cache_steps,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            max_grad_norm = max_grad_norm,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            lr_scheduler_type = lr_scheduler_type,
            warmup_ratio = warmup_ratio,
            warmup_steps = warmup_steps,
            log_level = log_level,
            log_level_replica = log_level_replica,
            log_on_each_node = log_on_each_node,
            logging_dir = logging_dir,
            logging_strategy = logging_strategy,
            logging_first_step = logging_first_step,
            logging_steps = logging_steps,
            logging_nan_inf_filter = logging_nan_inf_filter,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            save_safetensors = save_safetensors,
            save_on_each_node = save_on_each_node,
            save_only_model = save_only_model,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            no_cuda = no_cuda,
            use_cpu = use_cpu,
            use_mps_device = use_mps_device,
            seed = seed,
            data_seed = data_seed,
            jit_mode_eval = jit_mode_eval,
            use_ipex = use_ipex,
            bf16 = bf16,
            fp16 = fp16,
            fp16_opt_level = fp16_opt_level,
            half_precision_backend = half_precision_backend,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            local_rank = local_rank,
            ddp_backend = ddp_backend,
            tpu_num_cores = tpu_num_cores,
            tpu_metrics_debug = tpu_metrics_debug,
            debug = debug,
            dataloader_drop_last = dataloader_drop_last,
            eval_steps = eval_steps,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            past_index = past_index,
            run_name = run_name,
            disable_tqdm = disable_tqdm,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            fsdp = fsdp,
            fsdp_min_num_params = fsdp_min_num_params,
            fsdp_config = fsdp_config,
            fsdp_transformer_layer_cls_to_wrap = fsdp_transformer_layer_cls_to_wrap,
            accelerator_config = accelerator_config,
            deepspeed = deepspeed,
            label_smoothing_factor = label_smoothing_factor,
            optim = optim,
            optim_args = optim_args,
            adafactor = adafactor,
            group_by_length = group_by_length,
            length_column_name = length_column_name,
            report_to = report_to,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            skip_memory_metrics = skip_memory_metrics,
            use_legacy_prediction_loop = use_legacy_prediction_loop,
            push_to_hub = push_to_hub,
            resume_from_checkpoint = resume_from_checkpoint,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_always_push = hub_always_push,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            include_inputs_for_metrics = include_inputs_for_metrics,
            eval_do_concat_batches = eval_do_concat_batches,
            fp16_backend = fp16_backend,
            evaluation_strategy = evaluation_strategy,
            push_to_hub_model_id = push_to_hub_model_id,
            push_to_hub_organization = push_to_hub_organization,
            push_to_hub_token = push_to_hub_token,
            mp_parameters = mp_parameters,
            auto_find_batch_size = auto_find_batch_size,
            full_determinism = full_determinism,
            torchdynamo = torchdynamo,
            ray_scope = ray_scope,
            ddp_timeout = ddp_timeout,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            dispatch_batches = dispatch_batches,
            split_batches = split_batches,
            include_tokens_per_second = include_tokens_per_second,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            neftune_noise_alpha = neftune_noise_alpha,
            optim_target_modules = optim_target_modules,
            batch_eval_metrics = batch_eval_metrics,
            eval_on_start = eval_on_start,
            use_liger_kernel = use_liger_kernel,
            eval_use_gather_object = eval_use_gather_object,
            average_tokens_across_devices = average_tokens_across_devices,
            dataset_text_field = dataset_text_field,
            packing = packing,
            max_seq_length = max_seq_length,
            dataset_num_proc = dataset_num_proc,
            dataset_batch_size = dataset_batch_size,
            model_init_kwargs = model_init_kwargs,
            dataset_kwargs = dataset_kwargs,
            eval_packing = eval_packing,
            num_of_sequences = num_of_sequences,
            chars_per_token = chars_per_token,
            use_liger = use_liger,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

class _UnslothSFTTrainer(Trainer):
    r""""""

    _tag_names = ["trl", "sft"]

    @deprecate_kwarg(
        "tokenizer", "0.16.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        if args is None:
            args = SFTConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = SFTConfig(**args_as_dict)

        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            if args.use_liger:
                model = AutoLigerKernelForCausalLM.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for _, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                            break
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    model = model
                if (
                    args is not None
                    and args.bf16
                    and getattr(model, "is_loaded_in_4bit", False)
                    and not is_sharded_qlora
                ):
                    peft_module_casting_to_bf16(model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            args.max_seq_length = min(processing_class.model_max_length, 1024)

        self.dataset_num_proc = args.dataset_num_proc
        self.dataset_batch_size = args.dataset_batch_size

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        if formatting_func is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays None
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                args.dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    processing_class,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = args.packing if args.eval_packing is None else args.eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        processing_class,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if processing_class.padding_side is not None and processing_class.padding_side != "right":
            warnings.warn(
                "You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might "
                "lead to some unexpected behaviour due to overflow issues when training a model in half-precision. "
                "You might consider adding `processing_class.padding_side = 'right'` to your code.",
                UserWarning,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.train_dataset is not None:
            if self.args.max_steps > 0 and args.packing:
                self.train_dataset.infinite = True
            elif self.args.max_steps == -1 and args.packing:
                self.train_dataset.infinite = False

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        packing,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable],
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:
            return dataset
        if 'tokenizer'          not in locals(): tokenizer = processing_class
        if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')
        if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field
        if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')
        test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]
        chat_template = getattr(tokenizer, 'chat_template', None)
        chat_template = '' if chat_template is None else chat_template
        has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) if getattr(tokenizer, 'bos_token', None) is not None else False
        if 'add_special_tokens' not in locals() and has_bos_token_already:
            from functools import partial
            tokenizer_call = tokenizer.__call__
            tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)
            processing_class = tokenizer
        else:
            tokenizer_call = None
            add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "valid formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            def formatting_func(x):
                return x["input_ids"]

            if not packing:
                return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                processing_class,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                processing_class,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        processing_class,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:
            return dataset
        if 'tokenizer'          not in locals(): tokenizer = processing_class
        if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')
        if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field
        if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')
        test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]
        chat_template = getattr(tokenizer, 'chat_template', None)
        chat_template = '' if chat_template is None else chat_template
        has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) if getattr(tokenizer, 'bos_token', None) is not None else False
        if 'add_special_tokens' not in locals() and has_bos_token_already:
            from functools import partial
            tokenizer_call = tokenizer.__call__
            tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)
            processing_class = tokenizer
        else:
            tokenizer_call = None
            add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)
        if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call
        return tokenized_dataset

    def _prepare_packed_dataloader(
        self,
        processing_class,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func: Optional[Callable] = None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if processing_class is None:
            raise ValueError("You need to pass a processing_class with `SFTTrainer`.")

        constant_length_iterator = ConstantLengthDataset(
            processing_class,
            dataset,
            dataset_text_field=None if formatting_func is not None else dataset_text_field,
            formatting_func=formatting_func,
            seq_length=max_seq_length,
            infinite=False,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            eos_token_id=processing_class.eos_token_id,
            append_concat_token=append_concat_token,
            add_special_tokens=add_special_tokens,
        )

        if isinstance(dataset, datasets.IterableDataset):
            return constant_length_iterator

        def data_generator(constant_length_iterator):
            yield from constant_length_iterator

        try:
            packed_dataset = Dataset.from_generator(
                data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
            )
        except (DatasetGenerationError, SchemaInferenceError) as exc:
            raise ValueError(
                "Error occurred while packing the dataset. "
                "Make sure that your dataset has enough samples to at least yield one packed sequence."
            ) from exc
        return packed_dataset

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
class UnslothSFTTrainer(_UnslothSFTTrainer):
    """
    
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (`Optional[SFTConfig]`):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`Optional[transformers.DataCollator]`):
            The data collator to use for training.
        train_dataset (`Optional[datasets.Dataset]`):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supercedes the `tokenizer` argument, which is now deprecated.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    
    """
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        peft_config = None,
        formatting_func = None,
        **kwargs
    ):
        if args is None: args = UnslothSFTConfig()
        use_bf16 = getattr(args, 'bf16', False)
        use_fp16 = getattr(args, 'fp16', False)
        force_float32 = False
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
        if 'max_length' not in locals() and not hasattr(args, 'max_length'):
            pass
        else:
            if hasattr(args, 'max_seq_length') and args.max_seq_length is not None and args.max_seq_length > 0:
                if hasattr(args, 'max_length'):
                    args.max_length = args.max_seq_length
                    max_length = args.max_length
            else:
                model_max_length = getattr(model, 'max_seq_length', None)
                # print(model_max_length, 'mml1')
                if model_max_length is None: model_max_length = getattr(model, 'max_length', None)
                # print(model_max_length, 'mml2')
                if model_max_length is not None:
                    args.max_length = model_max_length
                    max_length = args.max_length
                elif hasattr(args, 'max_length') and args.max_length is not None:
                    max_length = args.max_length
                    # if we are here, then we are in a weird case where max_length is set but max_seq_length is not set
                    setattr(model, 'max_seq_length', max_length)
                else:
                    print('Unsloth: We did not find `max_seq_length` or `max_length` in the model or args. We will set it to 1024.')
                    args.max_length = 1024
        if model is not None and hasattr(model, 'for_training'):
            model.for_training()
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer, mlm = False, mlm_probability = 0.0)
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(__tokenizer)
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(__tokenizer.tokenizer)
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer.tokenizer, mlm = False, mlm_probability = 0.0)
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('sft_trainer', other_metrics)
        IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\n')
        from unsloth_zoo.tokenizer_utils import fix_untrained_tokens
        from unsloth_zoo.training_utils  import fix_zero_training_loss
        if 'tokenizer' not in locals(): tokenizer = processing_class
        fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)
        fix_zero_training_loss(model, tokenizer, train_dataset)
        
        super().__init__(
            model = model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            model_init = model_init,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            peft_config = peft_config,
            formatting_func = formatting_func,**kwargs)
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        
pass

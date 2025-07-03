# src/mlx_grpo/trainer.py

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
import tqdm

from.config import GRPOConfig
from.utils import pad_sequences, calculate_log_probs, generate

class GRPOTrainer:
    """
    A trainer for fine-tuning language models using Group Relative Policy Optimization (GRPO).
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: GRPOConfig,
        train_set,
        optimizer: AdamW,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_set = train_set
        self.optimizer = optimizer

        self.model_old = self._copy_model(model)
        self.model_ref = self._copy_model(model)
        
        self.model_ref.freeze()

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Creates a deep copy of the model with the same parameters."""
        import copy
        new_model = copy.deepcopy(model)
        mx.eval(new_model.parameters())
        return new_model

    def train(self):
        """
        The main training loop.
        """
        loss_and_grad_fn = nn.value_and_grad(self.model, self._grpo_loss_fn)

        losses = []
        all_rewards = []
        
        pbar = tqdm.tqdm(range(self.config.iters))
        for it in pbar:
            prompts, ground_truths = self._sample_batch()
            rollout_data = self._rollout_and_collect(prompts, ground_truths)
            advantages = self._compute_advantages(rollout_data["rewards_grouped"])
            
            old_log_probs = calculate_log_probs(
                self.model_old, 
                rollout_data["sequences_padded"], 
                rollout_data["labels_padded"]
            )

            (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
                self.model_ref,
                rollout_data["sequences_padded"],
                rollout_data["labels_padded"],
                advantages,
                old_log_probs,
            )
            
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            losses.append(loss.item())
            all_rewards.extend(mx.concatenate(rollout_data["rewards_grouped"]).tolist())
            pbar.set_description(
                f"Loss: {np.mean(losses[-10:]):.3f}, Mean Reward: {np.mean(all_rewards[-20:]):.3f}"
            )

            if (it + 1) % self.config.update_every == 0:
                self._sync_models()
                print(f"\nIter {it+1}: Synced old model weights.")

        print("Training finished.")
        return losses, all_rewards

    def _sample_batch(self):
        batch_prompts = []
        batch_answers = []
        indices = np.random.randint(0, len(self.train_set), self.config.batch_size)
        for i in indices:
            prompt_text, answer_text = self.train_set[i]["output"].rsplit(" ", maxsplit=1)
            full_prompt = [
                {"role": "user", "content": self.train_set[i]["instruction"]},
                {"role": "assistant", "content": prompt_text}
            ]
            batch_prompts.append(full_prompt)
            batch_answers.append(answer_text)
        return batch_prompts, batch_answers

    def _rollout_and_collect(self, prompts, ground_truths):
        rollout_sequences = []
        rollout_labels = []
        rollout_rewards_grouped = []

        for i in range(self.config.batch_size):
            prompt_tokens = self.tokenizer.apply_chat_template(prompts[i], add_generation_prompt=True)
            group_rewards = []
            for _ in range(self.config.group_size):
                response_text = generate(self.model_old, self.tokenizer, mx.array(prompt_tokens), max_tokens=self.config.max_ans_len)
                answer_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
                
                reward = 1.0 if ground_truths[i] in response_text else 0.0
                group_rewards.append(reward)

                full_sequence = mx.array(prompt_tokens + answer_tokens, dtype=mx.int32)
                
                prompt_labels = mx.array([self.config.ignore_index] * len(prompt_tokens), dtype=mx.int32)
                answer_labels = mx.array(answer_tokens, dtype=mx.int32)
                full_labels = mx.concatenate([prompt_labels, answer_labels])

                rollout_sequences.append(full_sequence)
                rollout_labels.append(full_labels)
            
            rollout_rewards_grouped.append(mx.array(group_rewards))
        
        sequences_padded = pad_sequences(rollout_sequences, self.tokenizer.pad_token_id)
        labels_padded = pad_sequences(rollout_labels, self.config.ignore_index)

        return {
            "sequences_padded": sequences_padded,
            "labels_padded": labels_padded,
            "rewards_grouped": rollout_rewards_grouped
        }

    def _compute_advantages(self, rewards_grouped):
        advantages = []
        for rewards in rewards_grouped:
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8 
            adv = (rewards - mean_reward) / std_reward
            advantages.append(adv)
        return mx.concatenate(advantages)

    def _sync_models(self):
        self.model_old.update(self.model.parameters())
        mx.eval(self.model_old.parameters())

    def _grpo_loss_fn(self, model_ref, sequences, labels, advantages, old_log_probs):
        log_probs = calculate_log_probs(self.model, sequences, labels)
        log_probs_ref = calculate_log_probs(model_ref, sequences, labels)

        ratio = mx.exp(log_probs - old_log_probs)
        clipped_ratio = mx.clip(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
        policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

        log_ratio_for_kl = log_probs_ref - log_probs
        ratio_for_kl = mx.exp(log_ratio_for_kl)
        kl_div = ratio_for_kl - log_ratio_for_kl - 1

        loss = -mx.mean(policy_reward - self.config.beta * kl_div)
        
        return loss, mx.mean(policy_reward), mx.mean(kl_div)

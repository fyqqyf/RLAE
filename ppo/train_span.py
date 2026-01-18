import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.distributions import Categorical
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import requests
import re
import os
from torch.utils.data import DataLoader


class SpanEnsembleWeightPolicy(nn.Module):
    def __init__(self, num_models=2, span_length=8):
        super().__init__()
        # Load mdeberta-v3 as the base model
        self.encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/mdeberta-v3-base",
            clean_up_tokenization_spaces=True,
            use_fast=True,
            model_max_length=512,
        )

        # Span length configuration
        self.span_length = span_length

        # Modify MLP head, add span position encoding
        self.mlp = nn.Sequential(
            nn.Linear(768 + 64, 256),  # Add 64-dim span position encoding
            nn.ReLU(),
            nn.Linear(256, num_models),
        )

        # Span position embedding
        self.span_position_embedding = nn.Embedding(128, 64)  # Support up to 128 spans

        self.softmax = nn.Softmax(dim=-1)

    def get_span_position_encoding(self, batch_size, device):
        """Generate span position encoding."""
        # Return current span position encoding; simplified, real use should track span position
        return self.span_position_embedding(torch.tensor([0], device=device))

    def forward(self, questions, span_position=0):
        # Encode input questions
        inputs = self.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.encoder(**inputs)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]

        # Add span position encoding
        span_pos_enc = self.span_position_embedding(
            torch.tensor([span_position], device=cls_output.device)
        ).expand(cls_output.size(0), -1)

        # Concatenate features
        combined_features = torch.cat([cls_output, span_pos_enc], dim=-1)

        # Generate ensemble weights
        logits = self.mlp(combined_features)
        weights = self.softmax(logits)

        return weights


class SpanPPOTrainer:
    def __init__(
        self,
        policy,
        lr=1e-4,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.01,
        api_url="http://0.0.0.0:8000/api/rl-train-span/",
        span_length=8,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.api_url = api_url
        self.span_length = span_length

        # Span state tracking
        self.current_span_position = 0
        self.tokens_in_current_span = 0
        self.current_weights = None

    def should_update_span_weights(self):
        """Decide whether to update span weights."""
        return self.tokens_in_current_span >= self.span_length

    def reset_span_state(self):
        """Reset span state."""
        self.tokens_in_current_span = 0
        self.current_span_position += 1
        self.current_weights = None

    def get_reward(
        self, question, answer, weights, span_position=0, max_new_tokens=1024
    ):
        """Call API to get ensemble result and compute reward."""
        data = {
            "messages_list": [[{"role": "user", "content": question}]],
            "max_new_tokens": max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights.tolist(),
            "span_position": span_position,  # Add span position info
            "span_length": self.span_length,  # Add span length info
        }

        response = requests.post(self.api_url, json=data)

        if response.status_code == 200:
            # Extract answer from generated text
            response_text = response.json()["response"][0]
            response_text_clean = response_text.replace(",", "")
            generated_answer = [
                float(num) for num in re.findall(r"-?\d*\.?\d+", response_text_clean)
            ][-1]

            # Extract number from ground-truth answer
            answer_clean = answer.replace(",", "")
            true_answer = [
                float(num) for num in re.findall(r"-?\d*\.?\d+", answer_clean)
            ][-1]

            # Return 1 if correct, otherwise 0
            reward = 1.0 if generated_answer == true_answer else 0.0

            return reward
        else:
            print(f"API Error: {response.status_code}")
            return 0.0

    def train_step(self, questions, answers):
        """Run one training step (span level)."""

        # Reset span state
        self.reset_span_state()

        batch_size = len(questions)
        total_rewards = []

        # Simulate generation process and update weights per span
        for span_pos in range(5):  # Assume at most 5 spans
            # Get current span weights
            weights = self.policy(questions, span_position=span_pos)
            old_probs = weights.detach()

            # Collect rewards within current span (simplified)
            span_rewards = []
            for i in range(batch_size):
                reward = self.get_reward(questions[i], answers[i], weights[i], span_pos)
                span_rewards.append(reward)

            rewards = torch.tensor(span_rewards)
            total_rewards.append(rewards)

            # PPO update (update once per span)
            for _ in range(3):  # Multiple updates
                new_weights = self.policy(questions, span_position=span_pos)

                # Compute ratio
                ratio = (new_weights + 1e-8) / (old_probs + 1e-8)

                # Compute surrogate loss
                expanded_rewards = rewards.unsqueeze(-1).expand_as(ratio)

                surr1 = ratio * expanded_rewards
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * expanded_rewards
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Add entropy regularization
                entropy = -(new_weights * torch.log(new_weights + 1e-8)).sum(-1).mean()

                # Total loss
                loss = policy_loss - self.entropy_coef * entropy

                print(
                    f"Span {span_pos} - Loss: {loss.item()}, "
                    f"Policy Loss: {policy_loss.item()}, Entropy: {entropy.item()}"
                )

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return torch.stack(total_rewards).mean()


def train(num_epochs=2, batch_size=128, span_length=8):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize policy network and trainer
    policy = SpanEnsembleWeightPolicy(num_models=2, span_length=span_length)
    trainer = SpanPPOTrainer(policy, span_length=span_length)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    total_steps = len(dataloader) * num_epochs
    global_step = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            questions = batch["question"]
            answers = batch["answer"]

            # Run training step (span level)
            avg_reward = trainer.train_step(questions, answers)
            global_step += 1

            # Print progress
            if global_step % 10 == 0:
                print(
                    f"Progress: {global_step}/{total_steps} steps, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

        # Save model at end of each epoch
        torch.save(policy.state_dict(), f"span_policy_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--span_length", type=int, default=8, help="Length of each span"
    )
    args = parser.parse_args()

    train(span_length=args.span_length)

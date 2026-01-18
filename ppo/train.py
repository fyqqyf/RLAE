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


class EnsembleWeightPolicy(nn.Module):
    def __init__(self, num_models=2):
        super().__init__()
        # Load mdeberta-v3 as the base model
        self.encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/mdeberta-v3-base",
            clean_up_tokenization_spaces=True,  # Explicitly set
            use_fast=True,  # Use fast tokenizer
            model_max_length=512,  # Set max length
        )

        # Add an MLP head to generate ensemble weights
        self.mlp = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, num_models)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, questions):
        # Encode input questions
        inputs = self.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.encoder(**inputs)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]

        # Generate ensemble weights
        logits = self.mlp(cls_output)
        weights = self.softmax(logits)

        return weights


class PPOTrainer:
    def __init__(
        self,
        policy,
        lr=1e-4,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.01,
        api_url="http://0.0.0.0:8000/api/rl-train/",
    ):

        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.api_url = api_url

    def get_reward(self, question, answer, weights, max_new_tokens=1024):
        """Call API to get ensemble result and compute reward."""
        data = {
            "messages_list": [[{"role": "user", "content": question}]],
            "max_new_tokens": max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights.tolist(),
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
        """Run one training step."""

        # Get current policy action probability distribution
        weights = self.policy(questions)
        old_probs = weights.detach()

        # Collect trajectories
        rewards = []
        for i in range(len(questions)):
            reward = self.get_reward(questions[i], answers[i], weights[i])
            rewards.append(reward)
        rewards = torch.tensor(rewards)

        # PPO update
        for _ in range(3):  # Multiple updates
            # Get new policy action probabilities
            new_weights = self.policy(questions)

            # Compute ratio
            ratio = (new_weights + 1e-8) / (old_probs + 1e-8)

            # Compute surrogate loss
            # Use r instead of A
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
                f"Loss: {loss.item()}, Policy Loss: {policy_loss.item()}, Entropy: {entropy.item()}"
            )

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train(num_epochs=2, batch_size=128):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize policy network and trainer
    policy = EnsembleWeightPolicy(num_models=2)
    trainer = PPOTrainer(policy)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers
        pin_memory=True,  # Speed up data transfer
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

            # Run training step
            trainer.train_step(questions, answers)
            global_step += 1

            # Print progress
            if global_step % 10 == 0:
                print(f"Progress: {global_step}/{total_steps} steps")

        # Save model at end of each epoch
        torch.save(policy.state_dict(), f"policy_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    train()

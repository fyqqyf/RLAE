import torch
from transformers import AutoModel, AutoTokenizer
from torch.distributions import Categorical
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import requests
import re
import os
import logging

import torch.nn as nn

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAEnsemblePolicy(nn.Module):
    def __init__(self, num_agents=2):
        super().__init__()
        # Base encoder shared by all agents
        self.encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/mdeberta-v3-base",
            clean_up_tokenization_spaces=True,
            use_fast=True,
            model_max_length=512,
        )

        # Separate MLP heads for each agent
        self.agent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),  # Each agent outputs one weight
                )
                for _ in range(num_agents)
            ]
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, questions):
        # Encode input questions
        inputs = self.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.encoder(**inputs)
        cls_output = outputs.last_hidden_state[:, 0]

        # Get weights from each agent
        agent_weights = []
        for head in self.agent_heads:
            weight = head(cls_output)
            agent_weights.append(weight)

        # Stack and normalize weights
        weights = torch.stack(agent_weights, dim=-1)  # [batch_size, num_agents]
        weights = self.softmax(weights)

        return weights


class MAPPOTrainer:
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
        """Get reward through API call"""
        # Convert weights to a 1D list
        weights_list = weights.detach().cpu().numpy().flatten().tolist()

        data = {
            "messages_list": [[{"role": "user", "content": str(question)}]],
            "max_new_tokens": max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights_list,  # Now a 1D list [w1, w2]
        }

        # Log detailed request data
        logger.info(f"Request data:")
        logger.info(f"- messages_list: {data['messages_list']}")
        logger.info(f"- weights shape: {weights.shape}")
        logger.info(f"- weights_list: {weights_list}")
        logger.info(f"- max_new_tokens: {max_new_tokens}")

        try:
            response = requests.post(self.api_url, json=data)

            if response.status_code == 422:
                logger.error(f"API Error 422: {response.text}")
                return 0.0

            response.raise_for_status()

            if response.status_code == 200:
                response_text = response.json()["response"][0]
                logger.info(f"API Response: {response_text}")
                response_text_clean = response_text.replace(",", "")
                generated_answer = [
                    float(num)
                    for num in re.findall(r"-?\d*\.?\d+", response_text_clean)
                ][-1]

                answer_clean = str(answer).replace(",", "")
                true_answer = [
                    float(num) for num in re.findall(r"-?\d*\.?\d+", answer_clean)
                ][-1]

                reward = 1.0 if abs(generated_answer - true_answer) < 1e-6 else 0.0
                return reward
            else:
                logger.error(f"API Error: {response.status_code}")
                return 0.0

        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            logger.exception("Detailed traceback:")
            return 0.0

    def train_step(self, questions, answers):
        # Get current policy weights for all agents
        all_weights = self.policy(questions)
        old_probs = all_weights.detach()

        # Collect rewards
        rewards = []
        for i in range(len(questions)):
            reward = self.get_reward(questions[i], answers[i], all_weights[i])
            rewards.append(reward)
        rewards = torch.tensor(rewards).unsqueeze(-1)  # [batch_size, 1]

        # MAPPO update
        for _ in range(3):
            new_weights = self.policy(questions)

            # Calculate ratios for all agents
            ratio = (new_weights + 1e-8) / (old_probs + 1e-8)

            # Calculate losses for all agents
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * rewards
            policy_loss = -torch.min(surr1, surr2).mean()

            # Add entropy regularization for all agents
            entropy = -(new_weights * torch.log(new_weights + 1e-8)).sum(-1).mean()

            # Total loss
            loss = policy_loss - self.entropy_coef * entropy

            print(
                f"Loss: {loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}, Entropy: {entropy.item():.4f}"
            )

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train(num_epochs=2, batch_size=16):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize policy and trainer
    policy = MAEnsemblePolicy(num_agents=2)
    trainer = MAPPOTrainer(policy)

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

            trainer.train_step(questions, answers)
            global_step += 1

            if global_step % 10 == 0:
                print(f"Progress: {global_step}/{total_steps} steps")

        # Save model after each epoch
        torch.save(policy.state_dict(), f"mappo_policy_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    train()

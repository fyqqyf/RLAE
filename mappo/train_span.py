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


class SpanMAEnsemblePolicy(nn.Module):
    """Multi-agent ensemble policy with span-level support."""

    def __init__(self, num_agents=2, span_length=8):
        super().__init__()
        # Base encoder shared by all agents
        self.encoder = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/mdeberta-v3-base",
            clean_up_tokenization_spaces=True,
            use_fast=True,
            model_max_length=512,
        )

        # Span configuration
        self.span_length = span_length

        # Separate MLP heads for each agent, with span position encoding
        self.agent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(768 + 64, 256),  # Add 64-dim span position encoding
                    nn.ReLU(),
                    nn.Linear(256, 1),  # Each agent outputs one weight
                )
                for _ in range(num_agents)
            ]
        )

        # Span position embedding
        self.span_position_embedding = nn.Embedding(128, 64)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, questions, span_position=0):
        # Encode input questions
        inputs = self.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.encoder(**inputs)
        cls_output = outputs.last_hidden_state[:, 0]

        # Add span position encoding
        span_pos_enc = self.span_position_embedding(
            torch.tensor([span_position], device=cls_output.device)
        ).expand(cls_output.size(0), -1)

        # Concatenate features
        combined_features = torch.cat([cls_output, span_pos_enc], dim=-1)

        # Get weights from each agent
        agent_weights = []
        for head in self.agent_heads:
            weight = head(combined_features)
            agent_weights.append(weight)

        # Stack and normalize weights
        weights = torch.stack(agent_weights, dim=-1)  # [batch_size, num_agents]
        weights = self.softmax(weights)

        return weights


class SpanMAPPOTrainer:
    """MAPPO trainer with span-level support."""

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
        """Get reward through API call"""
        # Convert weights to a 1D list
        weights_list = weights.detach().cpu().numpy().flatten().tolist()

        data = {
            "messages_list": [[{"role": "user", "content": str(question)}]],
            "max_new_tokens": max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights_list,  # Now a 1D list [w1, w2]
            "span_position": span_position,  # Add span position
            "span_length": self.span_length,  # Add span length
        }

        # Log detailed request data
        logger.info(f"Request data:")
        logger.info(f"- messages_list: {data['messages_list']}")
        logger.info(f"- weights shape: {weights.shape}")
        logger.info(f"- weights_list: {weights_list}")
        logger.info(f"- span_position: {span_position}")
        logger.info(f"- span_length: {self.span_length}")
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
        """Run span-level training step."""

        # Reset span state
        self.reset_span_state()

        batch_size = len(questions)
        total_rewards = []

        # Simulate generation process and update weights per span
        for span_pos in range(5):  # Assume at most 5 spans
            # Get current span weights (each agent generates independently)
            all_weights = self.policy(questions, span_position=span_pos)
            old_probs = all_weights.detach()

            # Collect rewards within current span
            rewards = []
            for i in range(batch_size):
                reward = self.get_reward(
                    questions[i], answers[i], all_weights[i], span_pos
                )
                rewards.append(reward)

            rewards = torch.tensor(rewards).unsqueeze(-1)  # [batch_size, 1]
            total_rewards.append(rewards)

            # MAPPO update (once per span)
            for _ in range(3):
                new_weights = self.policy(questions, span_position=span_pos)

                # Compute ratio
                ratio = (new_weights + 1e-8) / (old_probs + 1e-8)

                # Compute loss (all agents share rewards)
                surr1 = ratio * rewards
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * rewards
                policy_loss = -torch.min(surr1, surr2).mean()

                # Add entropy regularization (all agents)
                entropy = -(new_weights * torch.log(new_weights + 1e-8)).sum(-1).mean()

                # Total loss
                loss = policy_loss - self.entropy_coef * entropy

                print(
                    f"Span {span_pos} - Loss: {loss.item():.4f}, "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Entropy: {entropy.item():.4f}"
                )

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return torch.stack(total_rewards).mean()


def train(num_epochs=2, batch_size=16, span_length=8):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize policy and trainer
    policy = SpanMAEnsemblePolicy(num_agents=2, span_length=span_length)
    trainer = SpanMAPPOTrainer(policy, span_length=span_length)

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

            # Run span-level training step
            avg_reward = trainer.train_step(questions, answers)
            global_step += 1

            if global_step % 10 == 0:
                print(
                    f"Progress: {global_step}/{total_steps} steps, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

        # Save model
        torch.save(policy.state_dict(), f"span_mappo_policy_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--span_length", type=int, default=8, help="Length of each span"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    args = parser.parse_args()

    train(
        span_length=args.span_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

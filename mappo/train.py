import argparse
import os
import re
from typing import List, Optional, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_last_number(text: str) -> Optional[float]:
    numbers = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])


class MAEnsemblePolicy(nn.Module):
    def __init__(
        self,
        num_agents: int = 2,
        encoder_name: str = "microsoft/deberta-v3-large",
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'transformers'. Install requirements.txt before training."
            ) from exc

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", 768)
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            clean_up_tokenization_spaces=True,
            use_fast=True,
            model_max_length=512,
        )

        self.agent_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                )
                for _ in range(num_agents)
            ]
        )
        # Centralized critic (CTDE style training)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, questions: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0]

    def get_distribution_and_value(
        self, questions: List[str]
    ) -> Tuple[Dirichlet, torch.Tensor]:
        features = self._encode(questions)
        per_agent_alpha = [F.softplus(head(features)) + 1e-4 for head in self.agent_heads]
        alpha = torch.cat(per_agent_alpha, dim=-1)
        dist = Dirichlet(alpha)
        value = self.critic_head(features).squeeze(-1)
        return dist, value

    def forward(self, questions: List[str]) -> torch.Tensor:
        dist, _ = self.get_distribution_and_value(questions)
        return dist.mean


class MAPPOTrainer:
    def __init__(
        self,
        policy: MAEnsemblePolicy,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 3,
        api_url: str = "http://0.0.0.0:8000/api/rl-train/",
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.api_url = api_url

    def get_reward(
        self,
        question: str,
        answer: str,
        weights: torch.Tensor,
        max_new_tokens: int = 1024,
    ) -> Tuple[float, str]:
        data = {
            "messages_list": [[{"role": "user", "content": str(question)}]],
            "max_new_tokens": max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights.detach().cpu().tolist(),
        }

        try:
            response = requests.post(self.api_url, json=data, timeout=180)
            response.raise_for_status()
        except Exception as exc:
            print(f"API Error: {exc}")
            return 0.0, ""

        response_text = response.json().get("response", [""])[0]
        generated_answer = extract_last_number(response_text)
        true_answer = extract_last_number(str(answer))

        if generated_answer is not None and true_answer is not None:
            reward = 1.0 if abs(generated_answer - true_answer) < 1e-6 else 0.0
        else:
            reward = 1.0 if response_text.strip() == str(answer).strip() else 0.0

        return reward, response_text

    def train_step(self, questions: List[str], answers: List[str]) -> float:
        dist, values = self.policy.get_distribution_and_value(questions)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions).detach()

        rewards = []
        for i in range(len(questions)):
            reward, _ = self.get_reward(questions[i], answers[i], actions[i])
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.policy.device)
        returns = rewards
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            new_dist, new_values = self.policy.get_distribution_and_value(questions)
            new_log_probs = new_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, returns)
            entropy = new_dist.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return rewards.mean().item()


def train(
    num_epochs: int = 2,
    batch_size: int = 16,
    num_agents: int = 2,
    encoder_name: str = "microsoft/deberta-v3-large",
    api_url: str = "http://0.0.0.0:8000/api/rl-train/",
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install requirements.txt before training."
        ) from exc

    policy = MAEnsemblePolicy(num_agents=num_agents, encoder_name=encoder_name)
    if torch.cuda.is_available():
        policy = policy.cuda()
    trainer = MAPPOTrainer(policy=policy, api_url=api_url)

    dataset = load_dataset("gsm8k", "main", split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
    )

    total_steps = len(dataloader) * num_epochs
    global_step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            avg_reward = trainer.train_step(batch["question"], batch["answer"])
            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"Progress: {global_step}/{total_steps} steps, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

        torch.save(policy.state_dict(), f"mappo_policy_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--api_url", type=str, default="http://0.0.0.0:8000/api/rl-train/")
    args = parser.parse_args()

    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        encoder_name=args.encoder_name,
        api_url=args.api_url,
    )

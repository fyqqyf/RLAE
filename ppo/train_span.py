import argparse
import math
import os
import re
from typing import List, Optional, Sequence, Tuple, Union

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


class SpanEnsembleWeightPolicy(nn.Module):
    def __init__(
        self,
        num_models: int = 2,
        span_length: int = 4,
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

        self.span_length = span_length
        self.span_position_embedding = nn.Embedding(128, 64)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_models),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, state_texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            state_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0]

    def _normalize_span_positions(
        self,
        batch_size: int,
        span_position: Union[int, Sequence[int], torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(span_position, int):
            positions = torch.full(
                (batch_size,), span_position, dtype=torch.long, device=self.device
            )
        elif isinstance(span_position, torch.Tensor):
            positions = span_position.to(self.device, dtype=torch.long)
        else:
            positions = torch.tensor(span_position, dtype=torch.long, device=self.device)
        return torch.clamp(positions, min=0, max=127)

    def get_distribution_and_value(
        self,
        state_texts: List[str],
        span_position: Union[int, Sequence[int], torch.Tensor] = 0,
    ) -> Tuple[Dirichlet, torch.Tensor]:
        cls_output = self._encode(state_texts)
        positions = self._normalize_span_positions(len(state_texts), span_position)
        span_pos_enc = self.span_position_embedding(positions)
        features = torch.cat([cls_output, span_pos_enc], dim=-1)

        alpha = F.softplus(self.actor_head(features)) + 1e-4
        dist = Dirichlet(alpha)
        value = self.critic_head(features).squeeze(-1)
        return dist, value

    def forward(
        self,
        questions: List[str],
        span_position: Union[int, Sequence[int], torch.Tensor] = 0,
    ) -> torch.Tensor:
        dist, _ = self.get_distribution_and_value(questions, span_position)
        return dist.mean


class SpanPPOTrainer:
    def __init__(
        self,
        policy: SpanEnsembleWeightPolicy,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 3,
        api_url: str = "http://0.0.0.0:8000/api/rl-train-span/",
        span_length: int = 4,
        max_spans: Optional[int] = None,
        max_new_tokens: int = 128,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.api_url = api_url
        self.span_length = span_length
        self.max_spans = max_spans
        self.max_new_tokens = max_new_tokens

    def build_state_text(self, question: str, history: str) -> str:
        if history:
            return f"{question}\n\nPartial response:\n{history}"
        return question

    def get_reward(
        self,
        question: str,
        answer: str,
        weights: torch.Tensor,
        span_position: int = 0,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[float, str]:
        data = {
            "messages_list": [[{"role": "user", "content": question}]],
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "apply_chat_template": True,
            "new_weights": weights.detach().cpu().tolist(),
            "span_mode": True,
            "span_position": span_position,
            "span_length": self.span_length,
        }

        try:
            response = requests.post(self.api_url, json=data, timeout=180)
            response.raise_for_status()
        except Exception as exc:
            print(f"API Error: {exc}")
            return 0.0, ""

        response_text = response.json().get("response", [""])[0]
        generated_answer = extract_last_number(response_text)
        true_answer = extract_last_number(answer)

        if generated_answer is not None and true_answer is not None:
            reward = 1.0 if abs(generated_answer - true_answer) < 1e-6 else 0.0
        else:
            reward = 1.0 if response_text.strip() == answer.strip() else 0.0

        return reward, response_text

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # rewards/values: [T, B]
        t_steps, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, device=rewards.device)
        next_value = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(t_steps)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def train_step(self, questions: List[str], answers: List[str]) -> float:
        batch_size = len(questions)
        histories = ["" for _ in range(batch_size)]

        steps = (
            self.max_spans
            if self.max_spans is not None
            else max(1, math.ceil(self.max_new_tokens / self.span_length))
        )

        state_text_traj = []
        span_pos_traj = []
        action_traj = []
        old_logprob_traj = []
        value_traj = []
        reward_traj = []

        for span_pos in range(steps):
            state_texts = [
                self.build_state_text(question, history)
                for question, history in zip(questions, histories)
            ]

            dist, values = self.policy.get_distribution_and_value(state_texts, span_pos)
            actions = dist.sample()
            old_log_probs = dist.log_prob(actions)

            step_rewards = []
            new_histories = []
            for i in range(batch_size):
                reward, response_text = self.get_reward(
                    questions[i],
                    answers[i],
                    actions[i],
                    span_position=span_pos,
                )
                step_rewards.append(reward)
                new_histories.append(response_text or histories[i])

            histories = new_histories
            reward_tensor = torch.tensor(
                step_rewards, dtype=torch.float32, device=self.policy.device
            )

            state_text_traj.append(state_texts)
            span_pos_traj.append(torch.full((batch_size,), span_pos, dtype=torch.long))
            action_traj.append(actions.detach())
            old_logprob_traj.append(old_log_probs.detach())
            value_traj.append(values.detach())
            reward_traj.append(reward_tensor)

        rewards = torch.stack(reward_traj)  # [T, B]
        values = torch.stack(value_traj)  # [T, B]
        advantages, returns = self.compute_gae(rewards, values)

        flat_state_texts = [txt for step in state_text_traj for txt in step]
        flat_span_positions = torch.cat(span_pos_traj).to(self.policy.device)
        flat_actions = torch.cat(action_traj)
        flat_old_log_probs = torch.cat(old_logprob_traj)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        for _ in range(self.ppo_epochs):
            new_dist, new_values = self.policy.get_distribution_and_value(
                flat_state_texts, flat_span_positions
            )
            new_log_probs = new_dist.log_prob(flat_actions)
            ratio = torch.exp(new_log_probs - flat_old_log_probs)

            surr1 = ratio * flat_advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                * flat_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, flat_returns)
            entropy = new_dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return rewards.mean().item()


def train(
    num_epochs: int = 2,
    batch_size: int = 128,
    span_length: int = 4,
    num_models: int = 2,
    encoder_name: str = "microsoft/deberta-v3-large",
    max_spans: Optional[int] = None,
    max_new_tokens: int = 128,
    api_url: str = "http://0.0.0.0:8000/api/rl-train-span/",
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install requirements.txt before training."
        ) from exc

    policy = SpanEnsembleWeightPolicy(
        num_models=num_models,
        span_length=span_length,
        encoder_name=encoder_name,
    )
    if torch.cuda.is_available():
        policy = policy.cuda()

    trainer = SpanPPOTrainer(
        policy=policy,
        api_url=api_url,
        span_length=span_length,
        max_spans=max_spans,
        max_new_tokens=max_new_tokens,
    )

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

        torch.save(policy.state_dict(), f"span_policy_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_length", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_models", type=int, default=2)
    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_spans", type=int, default=None)
    parser.add_argument("--api_url", type=str, default="http://0.0.0.0:8000/api/rl-train-span/")
    args = parser.parse_args()

    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        span_length=args.span_length,
        num_models=args.num_models,
        encoder_name=args.encoder_name,
        max_spans=args.max_spans,
        max_new_tokens=args.max_new_tokens,
        api_url=args.api_url,
    )

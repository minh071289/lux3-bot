import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from orbitwars.features import GLOBAL_DIM, NUM_SHIP_BINS, PAIR_FEAT_DIM, PLANET_FEAT_DIM
from orbitwars.model import OrbitWarsGraphPolicy

DATASET_DIR = Path("imitation_learning/orbitwars_dataset")
AGENT_EPISODES_DIR = DATASET_DIR / "agent_episodes"
MODEL_NAME = "imitation_learning/weights/orbitwars_graph_policy.pth"


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def select_episodes(submission_ids, min_opp_score, val_ratio=0.1, num_episodes=None):
    submissions_df = pd.read_csv(DATASET_DIR / "submissions.csv")
    sid_to_score = dict(zip(submissions_df["submission_id"], submissions_df["score"]))
    games_df = pd.read_csv(DATASET_DIR / "games.csv")
    opp_scores = []
    for value in games_df["OppSubmissionIds"]:
        scores = [sid_to_score.get(int(x), 0) for x in str(value).split("|") if x]
        opp_scores.append(max(scores) if scores else 0)
    games_df["opp_score"] = opp_scores
    games_df = games_df[
        games_df["SubmissionId"].isin(submission_ids) & (games_df["opp_score"] >= min_opp_score)
    ]

    episodes = set()
    for sid, episode_id in zip(games_df["SubmissionId"], games_df["EpisodeId"]):
        path = AGENT_EPISODES_DIR / f"{sid}_{episode_id}.npz"
        if path.exists():
            episodes.add(str(path))
    episodes = sorted(episodes)
    if num_episodes is not None:
        episodes = episodes[:num_episodes]
    random.shuffle(episodes)
    split = int(len(episodes) * (1 - val_ratio))
    return episodes[:split], episodes[split:]


class OrbitWarsDataset(Dataset):
    def __init__(self, episodes):
        if not episodes:
            raise ValueError("No episodes found")
        self.episode_steps = []
        self.episode_data = {}
        for episode_id, episode_path in enumerate(episodes):
            data = np.load(episode_path)
            self.episode_data[episode_id] = {
                "global_features": data["global_features"],
                "planet_features": data["planet_features"],
                "pair_features": data["pair_features"],
                "planet_mask": data["planet_mask"],
                "owned_planet_mask": data["owned_planet_mask"],
                "launch_labels": data["launch_labels"],
                "target_labels": data["target_labels"],
                "ship_bin_labels": data["ship_bin_labels"],
            }
            for step_idx in range(len(data["global_features"])):
                self.episode_steps.append((episode_id, step_idx))

    def __len__(self):
        return len(self.episode_steps)

    def __getitem__(self, idx):
        episode_id, step_idx = self.episode_steps[idx]
        data = self.episode_data[episode_id]
        return (
            data["global_features"][step_idx],
            data["planet_features"][step_idx],
            data["pair_features"][step_idx],
            data["planet_mask"][step_idx],
            data["owned_planet_mask"][step_idx],
            data["launch_labels"][step_idx],
            data["target_labels"][step_idx],
            data["ship_bin_labels"][step_idx],
        )


def masked_mean(loss, mask):
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def compute_losses(model, batch, device):
    (
        global_features,
        planet_features,
        pair_features,
        planet_mask,
        owned_planet_mask,
        launch_labels,
        target_labels,
        ship_bin_labels,
    ) = batch

    global_features = global_features.to(device).float()
    planet_features = planet_features.to(device).float()
    pair_features = pair_features.to(device).float()
    planet_mask = planet_mask.to(device).float()
    owned_planet_mask = owned_planet_mask.to(device).float()
    launch_labels = launch_labels.to(device).long()
    target_labels = target_labels.to(device).long()
    ship_bin_labels = ship_bin_labels.to(device).long()

    launch_logits, target_logits, ship_logits = model(
        global_features,
        planet_features,
        pair_features,
        planet_mask,
    )

    owned_mask = (owned_planet_mask > 0) & (launch_labels >= 0)
    launch_targets = launch_labels.float().clamp_min(0)
    launch_loss = F.binary_cross_entropy_with_logits(
        launch_logits,
        launch_targets,
        reduction="none",
    )
    launch_loss = masked_mean(launch_loss, owned_mask.float())

    target_mask = (launch_labels == 1) & (target_labels >= 0)
    target_loss = F.cross_entropy(
        target_logits[target_mask],
        target_labels[target_mask],
    ) if target_mask.any() else launch_loss.new_zeros(())

    ship_mask = (launch_labels == 1) & (ship_bin_labels >= 0)
    ship_loss = F.cross_entropy(
        ship_logits[ship_mask],
        ship_bin_labels[ship_mask],
    ) if ship_mask.any() else launch_loss.new_zeros(())

    total_loss = launch_loss + target_loss + ship_loss
    return total_loss, launch_logits, target_logits, ship_logits, owned_mask, target_mask, ship_mask


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    launch_correct = 0
    launch_total = 0
    target_correct = 0
    target_total = 0
    ship_correct = 0
    ship_total = 0
    with torch.no_grad():
        for batch in dataloader:
            loss, launch_logits, target_logits, ship_logits, owned_mask, target_mask, ship_mask = compute_losses(
                model, batch, device
            )
            total_loss += float(loss.item()) * len(batch[0])

            (
                _global_features,
                _planet_features,
                _pair_features,
                _planet_mask,
                _owned_planet_mask,
                launch_labels,
                target_labels,
                ship_bin_labels,
            ) = batch

            launch_pred = (torch.sigmoid(launch_logits.cpu()) >= 0.5).long()
            owned_mask_cpu = owned_mask.cpu()
            launch_correct += int(((launch_pred == launch_labels.long()) & owned_mask_cpu).sum().item())
            launch_total += int(owned_mask_cpu.sum().item())

            if target_mask.any():
                target_pred = torch.argmax(target_logits[target_mask], dim=-1).cpu()
                target_true = target_labels[target_mask.cpu()].long()
                target_correct += int((target_pred == target_true).sum().item())
                target_total += len(target_true)

            if ship_mask.any():
                ship_pred = torch.argmax(ship_logits[ship_mask], dim=-1).cpu()
                ship_true = ship_bin_labels[ship_mask.cpu()].long()
                ship_correct += int((ship_pred == ship_true).sum().item())
                ship_total += len(ship_true)

    size = len(dataloader.dataset)
    return {
        "loss": total_loss / max(1, size),
        "launch_acc": launch_correct / max(1, launch_total),
        "target_acc": target_correct / max(1, target_total),
        "ship_acc": ship_correct / max(1, ship_total),
    }


def train_model(model, train_episodes, val_episodes, num_epochs=20, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(OrbitWarsDataset(train_episodes), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(OrbitWarsDataset(val_episodes), batch_size=batch_size, shuffle=False, num_workers=0)

    best_loss = float("inf")
    os.makedirs(Path(MODEL_NAME).parent, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        progress = tqdm(train_loader, leave=False)
        for batch in progress:
            loss, *_ = compute_losses(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_description(f"epoch {epoch + 1} loss {loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"val_loss={metrics['loss']:.4f} "
            f"launch_acc={metrics['launch_acc']:.4f} "
            f"target_acc={metrics['target_acc']:.4f} "
            f"ship_acc={metrics['ship_acc']:.4f}"
        )
        if metrics["loss"] < best_loss:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "global_dim": GLOBAL_DIM,
                    "planet_dim": PLANET_FEAT_DIM,
                    "pair_dim": PAIR_FEAT_DIM,
                    "ship_bins": NUM_SHIP_BINS,
                },
                MODEL_NAME,
            )
            best_loss = metrics["loss"]


@dataclass
class Args:
    submission_ids: list[int]
    min_opp_score: int = 0
    num_episodes: int | None = None
    num_epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3


def main(args: Args):
    seed_everything(42)
    train_episodes, val_episodes = select_episodes(
        args.submission_ids,
        min_opp_score=args.min_opp_score,
        num_episodes=args.num_episodes,
    )
    if not train_episodes or not val_episodes:
        raise ValueError("Need both train and validation episodes")
    train_model(
        OrbitWarsGraphPolicy(),
        train_episodes,
        val_episodes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

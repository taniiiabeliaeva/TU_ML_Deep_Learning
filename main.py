import os
from pathlib import Path
import configargparse



import torchtext
import torchmetrics

from dataset.dataset import LoaderConstructor
from dataset.dataset import create_poetryfoundation_dataset, create_poems_txt_dataset


import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm import LSTM
from models.transformer import Transformer

from trainer.trainer import Trainer
from trainer.scheduler import ChainedScheduler


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False

    config_path = Path(__file__).parent / "config.yaml"
    parser = configargparse.ArgumentParser(
        default_config_files=[config_path],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--dataset", type=str, default="wikitext-2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-text-length", type=int, default=200)
    parser.add_argument("--chained-scheduler", type=str2bool, default=False)
    return parser.parse_args()


def get_model(model, vocab_size, embed_dim, seq_len, output_dim, device):
    if model == "lstm":
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=2,
        )
    elif model == "transformer":
        return Transformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Model {model} not found")


if __name__ == "__main__":
    cfg = get_args()

    # Load the dataset
    if cfg.dataset == "poetryfoundation":
        dataset = create_poetryfoundation_dataset(os.getcwd())
    elif cfg.dataset == "poems_txt":
        dataset = create_poems_txt_dataset(os.getcwd())

    # Construct the dataloaders
    lc = LoaderConstructor(
        dataset=dataset,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        labels_sequence=False,
        min_freq=1 if cfg.dataset == "alicewonderland" else 3,
    )
    loaders = {}
    for loader in ["train", "validation", "test"]:
        loaders[loader] = lc.construct_loader(split=loader)

    input_size = loaders["train"].dataset.input_size
    vocab_size = lc.vocab_size
    output_size = lc.output_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = get_model(
        model=cfg.model,
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        seq_len=input_size,
        output_dim=output_size,
        device=device,
    )


    # Initialize the optimizer, loss function, and accuracy metric
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=output_size, top_k=5
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        accuracy=accuracy,
        batch_size=cfg.batch_size,
        output_dim=output_size,
        device=device,
    )

    scheduler = None
    if cfg.chained_scheduler:
        warmup_steps = 4
        scheduler = ChainedScheduler(
            trainer.optimizer,
            T_0=(cfg.epochs - warmup_steps),
            T_mul=1,
            eta_min=cfg.lr / 10,
            gamma=0.5,
            max_lr=cfg.lr,
            warmup_steps=warmup_steps,
        )

    # Train the model
    model_filename = f"trained_models/{cfg.model}_{cfg.dataset}_lr={str(cfg.lr).replace('.', '_')}_best.pt"

    # Create the directory if it doesn't exist
    Path("trained_models").mkdir(parents=True, exist_ok=True)

    best_valid_loss = float("inf")
    for epoch in range(cfg.epochs):

        # Training
        model.train()
        trainer.train_validate_epoch(loaders["train"], epoch, "train")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = trainer.train_validate_epoch(
                loaders["validation"], epoch, "validation"
            )

        # Save the best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), model_filename)
            print(f"Model improved, saving model")

        if scheduler:
            scheduler.step()

    torch.save(model.state_dict(), model_filename.replace("best", "lastepoch"))

    # Load the best model
    model.load_state_dict(torch.load(model_filename))

    # Test the model
    model.eval()
    with torch.no_grad():
        trainer.train_validate_epoch(loaders["test"], None, "test")

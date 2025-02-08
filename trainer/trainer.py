from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        accuracy,
        batch_size,
        output_dim,
        device,
        patience=5,  # Early stopping patience 
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy = accuracy
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.device = device
        self.model.to(self.device)

        # Early stopping variables
        self.patience = patience
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

    def train_validate_epoch(self, loader, epoch, split):
        # Reset metrics
        total_loss = 0
        self.accuracy.reset()
        
        # For calculating f1 score
        all_preds = []
        all_labels = []

        for batch in tqdm(loader, total=len(loader)):
            # Forward pass
            inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
            labels = labels.contiguous().view(-1)
            output = self.model(inputs).view(-1, self.output_dim)

            # Compute accuracy and loss
            self.accuracy.update(output, labels)
            loss = self.criterion(output, labels)

            # Collecting predictions and labels for F1 score calculation
            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.cpu().numpy())  # Add predictions to the list
            all_labels.extend(labels.cpu().numpy())  # Add true labels to the list

            # Backpropagation (only for training)
            if split == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        
        print(f"[{split.upper()} {epoch}]: Loss: {avg_loss}, Accuracy: {self.accuracy.compute()}, F1 Score: {f1:.4f}")

        # Early stopping check (only on validation)
        if split == "validation":
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_no_improve = 0  # Reset counter if we see improvement
                print(" Validation loss improved. Saving model...")
            else:
                self.epochs_no_improve += 1  # Count consecutive bad epochs
                print(f" No improvement for {self.epochs_no_improve}/{self.patience} epochs.")

            if self.epochs_no_improve >= self.patience:
                print(" Early stopping.")
                return "stop"

        return avg_loss

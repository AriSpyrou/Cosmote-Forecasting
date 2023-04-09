import os
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import parser as pars
import matplotlib.pyplot as plt
import plot_confusion_matrix as pcm

# Random seed for experiment reproducibility
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model hyper-parameters
NUM_EPOCHS = 30
BATCH_SIZE = 64
INPUT_SIZE = 13
HIDDEN_SIZE = 56
NUM_LAYER = 3
NUM_CLASSES = 10
L2 = 0.0000
BI = True
DROPOUT = True
PACK_PADDED = True

# Early stopping parameters
THRESHOLD = 0.03
INIT = 0
EARLY_STOP = False
PATIENCE = 3
EPOCHS_NO_IMPROVE = 0
FORCE_TRAIN = True

# Filesystem experiment naming
model_name = "LSTM{}_{}h_{}layers_{}".format(NUM_EPOCHS, HIDDEN_SIZE, NUM_LAYER, L2)
if DROPOUT:
    model_name += "_Dropout"
if BI:
    model_name += "_bi"


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels, max_length):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # Length of sequence is the number of windows of each wav
        self.lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.int64).to(device)

        self.feats = torch.tensor(self.zero_pad_and_stack(feats, max_length), dtype=torch.float).to(device)

        if isinstance(labels, (list, tuple)):
            self.labels = torch.tensor(np.array(labels), dtype=torch.int64).to(device)
        else:
            self.labels = torch.tensor(labels, dtype=torch.int64).to(device)

    def zero_pad_and_stack(self, x, max_length):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        for wav in x:
            # Find necessary padding for each wav
            padding = max_length - wav.shape[0]
            padded.append(np.append(wav, np.zeros(shape=(padding, x[0].shape[1])), axis=0))
        return np.array(padded)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, bidirectional=False):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        """
        input_dim: Num of features
        hidden_size: Num of nodes per layer (arbitraty)
        num_layers: Num of layers; each one contains hidden_size nodes
        num_classes: Num of classes; One for each digit
        """
        self.lstm = nn.LSTM(self.input_dim, self.feature_size, self.num_layers, dropout=0.1,
                            bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.feature_size * (2 ** self.bidirectional), self.num_classes)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        # initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * (2 ** self.bidirectional), x.size(0), self.feature_size,
                         dtype=torch.float).to(device)
        c0 = torch.zeros(self.num_layers * (2 ** self.bidirectional), x.size(0), self.feature_size,
                         dtype=torch.float).to(device)

        if PACK_PADDED: x = pack_padded_sequence(x, lengths.cpu().numpy(), batch_first=True)

        x, _ = self.lstm(x, (h0, c0))
        # Undo the packing operation for the linear layer
        if PACK_PADDED: x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.last_timestep(x, lengths, bidirectional=self.bidirectional)

        out = self.fc(x)
        return out

    def last_timestep(self, outputs, lengths, bidirectional):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()


def prepare_pack_padded(length, sequence, label):
    # sort sequences based on length
    length, perm_idx = length.sort(0, descending=True)
    # permute based on permutation of lengths
    sequence = sequence[perm_idx]
    label = label[perm_idx]
    return length, sequence, label


def score_network(model, loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        # batch of loader
        for sequence, label, length in loader:

            if PACK_PADDED:
                length, sequence, label = prepare_pack_padded(length, sequence, label)

            outputs = model(sequence, length)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += label.view(-1, ).size(0)
            n_correct += (predicted == label).sum().item()
        acc = 100.0 * n_correct / n_samples

        return acc


def confusion_matrix(model, loader):
    # Confusion matrix has shape (num_classes,num_classes)
    pred_per_category = np.zeros(shape=(10, 10), dtype=np.int32)
    with torch.no_grad():
        # batches in loader
        for sequence, label, length in loader:

            if PACK_PADDED:
                # sort sequences based on length
                length, perm_idx = length.sort(0, descending=True)
                # permute based on permutation of lengths
                sequence = sequence[perm_idx]
                label = label[perm_idx]

            outputs = model(sequence, length)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            for label_, prediction in zip(label, predicted):
                # add one to row label (true label), column prediction (predicted label)
                pred_per_category[label_.item(), prediction.item()] += 1
        return pred_per_category


def eval_model(model, test_loader, val_loader):
    """
    Used to print accuracy in both testing and validation datasets
    :param model: Model to be evaluated
    :param test_loader: test dataset loader
    :param val_loader: validation dataset loader
    :return: None
    """
    print(f'Accuracy of the network on the validation dataset: {score_network(model, val_loader)} %')
    print(f'Accuracy of the network on the test dataset: {score_network(model, test_loader)} %')


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = pars.parser("recordings/", n_mfcc=13)

    # Find max sequence length based on X_train,X_test,X_val
    max_window_length = max([max(i.shape[0], j.shape[0], k.shape[0]) for i, j, k in zip(X_train, X_test, X_val)])

    # Initialize datasets
    train_dataset = FrameLevelDataset(X_train, y_train, max_window_length)
    val_dataset = FrameLevelDataset(X_val, y_val, max_window_length)
    test_dataset = FrameLevelDataset(X_test, y_test, max_window_length)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if os.path.exists(os.path.join("models/", model_name)) and not FORCE_TRAIN:
        # Load model and evaluation
        model = torch.load(os.path.join("models/", model_name))
        model.eval()
        print(f'Accuracy of the network on the validation dataset: {score_network(model, val_loader)} %')
        print(f'Accuracy of the network on the test dataset: {score_network(model, test_loader)} %')
    else:
        # Create-train model
        model = BasicLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, NUM_CLASSES, bidirectional=BI).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=L2)

        n_total_steps = len(train_loader)
        loss_values = []
        val_loss_values = []

        t0 = time.time()
        for epoch in range(NUM_EPOCHS):
            train_loss = 0
            for i, (sequence, label, length) in enumerate(train_loader):
                # origin shape: [N, max_window_length, input_shape]
                label = label.to(device)
                if PACK_PADDED:
                    length, sequence, label = prepare_pack_padded(length, sequence, label)

                # Forward pass

                outputs = model(sequence, length)

                loss = criterion(outputs, label.view(-1, ))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss * sequence.size(0)

                # print epoch loss every 10 steps
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            # print train epoch loss
            print("Epoch {}: {}".format(epoch + 1, train_loss.item() / len(train_dataset)))
            loss_values.append(train_loss / len(train_dataset))

            model.eval()
            val_loss = 0
            for i, (sequence, label, length) in enumerate(val_loader):
                # origin shape: [N, max_window_length, input_shape]
                label = label.to(device)
                if PACK_PADDED:
                    length, sequence, label = prepare_pack_padded(length, sequence, label)

                # Forward pass

                outputs = model(sequence, length)

                loss = criterion(outputs, label.view(-1, ))

                val_loss += loss * sequence.size(0)
            val_loss = val_loss.item() / len(val_dataset)
            val_loss_values.append(val_loss)
            # print("Epoch {}: {}".format(epoch + 1, val_loss.item()/len(val_dataset)))
            model.train()
            # Early stopping check
            print(INIT - val_loss)
            if abs(INIT - val_loss) < THRESHOLD:
                # Save the model
                EPOCHS_NO_IMPROVE += 1
            else:
                torch.save(model, os.path.join("models/", model_name))
                EPOCHS_NO_IMPROVE = 0
            INIT = val_loss
            if epoch > 5 and EPOCHS_NO_IMPROVE == PATIENCE:
                print('Early stopping!')
                EARLY_STOP = True
                model.eval()
                eval_model(model, test_loader, val_loader)
                break

        t1 = time.time()
        total = t1 - t0
        print("Time for all epochs {}".format(total))

        # Plot training validation error
        plt.plot(np.array(loss_values), label="Train loss")
        plt.plot(val_loss_values, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if not EARLY_STOP:
            # should still save the model
            torch.save(model, os.path.join("models/", model_name))
            model.eval()
            eval_model(model, test_loader, val_loader)

        # Confusion matrices for validation/test
        cm_val = confusion_matrix(model, val_loader)
        cm_test = confusion_matrix(model, test_loader)

        pcm.plot_confusion_matrix(cm_val, np.unique(y_val))
        pcm.plot_confusion_matrix(cm_test, np.unique(y_test))

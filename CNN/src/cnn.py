import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import src.dataset_loader as dataset_loader

# Pour la matrice de confusion
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

#######################################################
#################CRÉATION DU RÉSEAU####################
#######################################################

plot_loss = []
plot_accuracy = []


class NeuralNetwork(nn.Module):
    def __init__(self, digits) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26 if not digits else 36),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def predict(self, x):
        # Calculer la prédiction du réseau.
        logits = self(x.unsqueeze(0).to(device))
        pred_probab = nn.Softmax(
            dim=1
        )(
            logits
        )  # On doit appliquer la softmax sur l'output du réseau pour avoir une probabilité.
        prediction = pred_probab.argmax(
            1
        )  # On récupère la probabilité la plus élévée dans le tenseur. Il faut appliquer la méthode item() pour récupérer le nombre python sous-jacent.
        return prediction.item()


# Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # On informe le modèle qu'on va l'entraîner.
    for batch, (
        X,
        y,
    ) in enumerate(dataloader):
        # Calculer la prédiction et la perte.
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # Calcul des gradients
        optimizer.step()  # Ajuster les paramètres en utilisant les gradients calculés précedemment.
        optimizer.zero_grad()  # Réinitialiser le gradient pour le recalculer à la prochaine étape.

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# Testing
def test_loop(dataloader, model, loss_fn):
    model.eval()  # On informe le modèle qu'on va le tester.
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # Pour désactiver les calculs de gradients (logique car on ne veut pas entraîner le réseau mais le tester.)
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    plot_accuracy.append(correct)
    plot_loss.append(test_loss)

    print(
        f"Test error:\nAccuracy: {(100 * correct):>0.1f}%, Avg los: {test_loss:>8f}\n"
    )


#######################################################
###################BOUCLE PRINCIPALE###################
#######################################################

if __name__ == "__main__":
    digits = False
    model = NeuralNetwork(digits).to(device)

    learning = False  # Pour avoir la phase d'entrainement
    loading = True  # Pour charger les poids
    saving = False  # Pour sauvegarder les poids
    plot = False  # Pour générer le graphique de la précision et de la perte lors de l'apprentissage
    matrix = False  # Pour générer la matrice de confusion

    learning_rate = 5e-2
    batch_size = 64
    epochs = 300

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if loading:
        if digits:
            model.load_state_dict(
                torch.load("models/model_weights_digits.pth", weights_only=True)
            )
        else:
            model.load_state_dict(
                torch.load("models/model_weights.pth", weights_only=True)
            )

    if learning:
        for t in range(epochs):
            print(f"Epoch {t + 1}\n--------------------")
            train_loop(dataset_loader.train_loader, model, loss_fn, optimizer)
            test_loop(dataset_loader.test_loader, model, loss_fn)
        if saving:
            if digits:
                torch.save(model.state_dict(), "models/model_weights_digits.pth")
            else:
                torch.save(model.state_dict(), "models/model_weights.pth")

        if plot:
            t = np.linspace(1, epochs, epochs)
            plt.plot(t, plot_accuracy)

            plt.title("Précision du MLP sur les données de test")
            plt.xlabel("Epochs")
            plt.ylabel("Précision")
            plt.savefig("assets/accuracy.png")

            plt.show()  # Pour clear le plot.

            plt.plot(t, plot_loss)

            plt.title("Valeur de la fonction de perte du MLP sur les données de test")
            plt.xlabel("Epochs")
            plt.ylabel("Perte")
            plt.savefig("assets/loss.png")

    if matrix:
        y_pred = []
        y_true = []

        with torch.no_grad():
            for X, y in dataset_loader.test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(y.cpu().numpy())

        cf_matrix = confusion_matrix(y_true, y_pred)

        axis = [chr(ord("A") + i) for i in range(26)]
        if digits:
            axis.extend([chr(ord("0") + i) for i in range(10)])

        df_cm = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=axis, columns=axis
        )
        df_cm = df_cm.round(2)

        plt.figure(figsize=(24, 14))
        sn.heatmap(df_cm, annot=True)
        if digits:
            plt.savefig("assets/confusionMatrix_digits.png")
        else:
            plt.savefig("assets/confusionMatrix.png")

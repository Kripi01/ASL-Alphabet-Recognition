import torch
import numpy as np
import src.dataset_loader as dataset_loader
import matplotlib.pyplot as plt

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


def kNN(points, p, k=3):
    """
    Paramètres -
            points: Dictionnaire avec 26 clés (toutes les lettres de l'alphabet ou 36 si on considère les chiffres) représentant
            le jeu de données. Chaque clé est associée à une liste de points d'entraînement, i.e. les landmarks dans mediapipe.
            p : Un tuple représentant le point de donnée à classifier, donc un tenseur de landmarks.
            A tuple, test data point of the form of an image
            k : Le nombre de plus proche voisins à considérer. La valeur par défault est 3.
    """

    # Initialiser une liste pour maintenir les k plus petites distances
    k_nearest = []

    for group in points:  # On itère sur les lettres.
        for feature in points[
            group
        ]:  # On itère la liste de landmarks qui correspond à une lettre
            # Calcul de la distance euclidienne entre les landmarks en argument et les landmarks du training test.
            euclidean_distance = torch.norm(feature - p).item()

            # Si la liste n'est pas encore pleine, ajouter la distance
            if len(k_nearest) < k:
                k_nearest.append((euclidean_distance, group))
                # Garder la liste triée pour faciliter l'insertion
                k_nearest.sort(key=lambda x: x[0])
            else:
                # Si la nouvelle distance est plus petite que la plus grande distance dans la liste, remplacer
                if euclidean_distance < k_nearest[-1][0]:
                    k_nearest[-1] = (euclidean_distance, group)
                    # Garder la liste triée
                    k_nearest.sort(key=lambda x: x[0])

    # Vote majoritaire
    compteur = {}  # Dictionnaire pour compter les occurrences des lettres
    for _, lettre in k_nearest:
        compteur[lettre] = compteur.get(lettre, 0) + 1

    # Trouver la lettre avec le plus d'occurrences
    return max(compteur, key=compteur.get)


train_features, train_labels = next(iter(dataset_loader.train_loader))
train_dataset = dataset_loader.train_dataset
test_dataset = dataset_loader.test_dataset


# On crée le dictionnaire qui correspond au jeu de données en utilisant le jeu de données déja formatté pour le CNN.
dataset = {}

for cle, valeur in zip(train_labels, train_features):
    c = cle.item()
    if c in dataset:
        dataset[c].append(valeur)  # On ajoute la valeur à la liste existante
    else:
        dataset[c] = [
            valeur
        ]  # On crée une nouvelle entrée avec une liste contenant la valeur


if __name__ == "__main__":
    matrix = True
    save = False
    k = 3
    test_k = True

    if matrix:
        s = 0
        # On génère la matrice de confusion sur le jeu de test.
        y_pred = []
        y_true = []
        for i in range(100):
            landmark, label = train_dataset[i]

            prediction = kNN(dataset, landmark, k)

            y_pred.append(prediction)
            y_true.append(label.item())

            if label == prediction:
                s += 1
        accuracy = s / (len(train_dataset) * 10)
        print(f"\nPour k = {k}, précision moyenne sur 10 tests de {accuracy * 100}%.")

        cf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(26)))

        axis = [chr(ord("A") + i) for i in range(26)]

        df_cm = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=axis, columns=axis
        )
        df_cm = df_cm.round(2)

        plt.figure(figsize=(24, 14))
        sn.heatmap(df_cm, annot=True)
        if save:
            plt.savefig("assets/confusionMatrix_kNN.png")
        plt.show()

    if test_k:
        for k in range(101, 1001, 200):
            s = 0
            for i in range(100):
                landmark, label = train_dataset[i]
                prediction = kNN(dataset, landmark, k)
                if label == prediction:
                    s += 1

            accuracy = s / (len(train_dataset) * 10)
            print(
                f"\nPour k = {k}, précision moyenne sur 10 tests de {accuracy * 100}%."
            )

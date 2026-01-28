import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import mediapipe as mp
import numpy as np


# Configuration de l'appareil.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set up les objets MediaPipe pour la détection des mains.
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Pour récupérer les jeux de données déjà crées dans le dossier CNN.
# On obtient le chemin absolu du dossier actuel
current_dir = os.path.dirname(__file__)

# On construit le chemin vers le dossier "data" qui est dans "../CNN"
dataset_path = os.path.join(current_dir, "../..", "CNN/")


class CustomLandmarkDataset(Dataset):
    def __init__(self, folder_labels, digits, dataset_id, transform=None):
        """
        folder_labels: Une liste de tuples (folder_path, label) pour chaque dossier.
        storing_dir: Chemin où sauvegarder les images traîtées.
        preprocessed_file: Chemin du fichier où les données pré-traîtées sont stockées.
        digits: Booléen indiquant si on prend en compte les chiffres dans le CNN.
        dataset_id: Entier indiquant le dataset représenté par l'instance.

        Crée un jeu de données contenant des données asscoiées aux landmarks d'images de lettres de A à Z
        (labélisés de 0 à 25) et de chiffres de 0 à 9 (labélisés de 26 à 35)
        """
        self.transform = transform

        self.preprocessed_file = ""
        if digits:
            self.preprocessed_file = (
                dataset_path
                + "data/preprocessed/digits/preprocessed_data_digits"
                + (str(dataset_id) if dataset_id != -1 else "")
                + ".pt"
            )
        else:
            self.preprocessed_file = (
                dataset_path
                + "data/preprocessed/letters/preprocessed_data"
                + (str(dataset_id) if dataset_id != -1 else "")
                + ".pt"
            )

        # Checker si les données pré-traîtées existent.
        if os.path.exists(self.preprocessed_file):
            print("Loading preprocessed data...")
            self.data_cache = torch.load(self.preprocessed_file, weights_only=True)
        else:
            print("Preprocessing data...")
            self.image_paths = []
            self.labels = []
            self.data_cache = []

            # Collecter tous les chemins d'images correspondant aux labels de tous les dossiers.
            for folder_path, label in folder_labels:
                if (
                    label not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                ) or digits:
                    folder_images = [
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if f.endswith(".jpg")
                    ]
                    self.image_paths.extend(folder_images)
                    self.labels.extend(
                        [label] * len(folder_images)
                    )  # Assigner le label (la lettre) à chaque image.

            self.dataset_size = len(self.labels)

            # Pré-traîter toutes les données et les stocker.
            self.data_cache = [
                self._process_image(idx) for idx in range(len(self.image_paths))
            ]

            # Sauvegarder les données pré-traîtées.
            torch.save(self.data_cache, self.preprocessed_file)
            print(f"Preprocessed data saved to {self.preprocessed_file}")

    def _process_image(self, idx):
        print(idx / self.dataset_size * 100)

        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        # Redimensionner l'image à la taille courante (280, 280).
        image = image.resize((280, 280))

        # Convertir l'image en un tableau numpy pour le passer dans la détection des mains MediaPipe.
        image_rgb = np.array(image)

        # Calculer les landmarks en utilisant MediaPipe.
        results = mp_hands.process(image_rgb)

        landmarks_pos = []

        # Extraire les landmarks si une (ou plusieurs) main a été détectée.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_pos.append(
                    [landmark.x for landmark in hand_landmarks.landmark]
                    + [landmark.y for landmark in hand_landmarks.landmark]
                    + [landmark.z for landmark in hand_landmarks.landmark]
                )

        # Si aucun landmarks n'a été détecté, ajouter un tableau vide
        if not landmarks_pos:
            landmarks_pos = [
                [0] * 63
            ]  # Placeholder pour les landmarks manquants (21 points * 3 coordonées)

        # Convertir les landmarks en tenseur.
        max_landmarks = 21  # Max landmarks (21 points par main).
        num_coordinates = 3  # (x, y, z) pour chaque point
        padded_landmarks = torch.zeros(max_landmarks * num_coordinates)
        if landmarks_pos:
            padded_landmarks[: len(landmarks_pos[0])] = torch.tensor(landmarks_pos[0])

        # Convertir chaque label en tenseur.
        if ord("0") <= ord(label) <= ord("9"):
            label_tensor = torch.tensor(
                ord(label) - ord("0") + 26
            )  # On numérote les chiffres de 26 (inlcus) à 35
        elif ord("A") <= ord(label) <= ord("Z"):  # Si le label est une lettre
            label_tensor = torch.tensor(
                ord(label) - ord("A")
            )  # Convertir le label (la lettre) en valeur numérique (0-25)

        return padded_landmarks.float(), label_tensor

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


folder_labels0 = [
    (f"{dataset_path}data/raw/dataset0/{chr(i)}", chr(i))
    for i in range(ord("A"), ord("Z") + 1)
]
folder_labels1 = [
    (f"{dataset_path}data/raw/dataset1/{chr(i)}", chr(i))
    for i in range(ord("A"), ord("Z") + 1)
]
folder_labels2 = [
    (f"{dataset_path}data/raw/dataset2/{chr(i)}", chr(i))
    for i in range(ord("A"), ord("Z") + 1)
]
folder_labels3 = [
    (f"{dataset_path}data/raw/dataset3/{chr(i)}", chr(i))
    for i in range(ord("A"), ord("Z") + 1)
]
folder_labels4 = [
    (f"{dataset_path}data/raw/dataset4/{chr(i)}", chr(i))
    for i in range(ord("A"), ord("Z") + 1)
]

# Création du dataset combiné avec les landmarks de tous les fichiers
dataset_complet = CustomLandmarkDataset(
    folder_labels=folder_labels1,  # ou folder_labels0 / 1 / 2 / etc selon le dataset que l'on veut utiliser
    digits=False,
    dataset_id=1,  # 0, 1, 2, 3, 4 selon le dataset que l'on veut utiliser.
)

# Définir la proportion de données pour le test
test_ratio = 0.2  # 20% pour le test, 80% pour l'entraînement
total_size = len(dataset_complet)
test_size = int(test_ratio * total_size)
train_size = total_size - test_size

# Split aléatoire du dataset
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset_complet, [train_size, test_size]
)

# Création des DataLoaders pour pouvoir charger le dataset. Il n'y a qu'un seul batch car on utilise le kNN et non le réseau de neurones.
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=False)

print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")


#####################################################################################
##################################BOUCLE PRINCIPALE##################################
#####################################################################################

if __name__ == "__main__":
    # Tester le train_dataset en chargeant les données.
    for i, (data, label) in enumerate(train_dataset):
        print(f"Processed batch {i}, data (shape): {data.shape}, label: {label}")

    # Tester le test_dataset en chargeant les données.
    for i, (data, label) in enumerate(test_dataset):
        print(f"Processed batch {i}, data (shape): {data.shape}, label: {label}")

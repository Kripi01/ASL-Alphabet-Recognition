import cv2
import mediapipe as mp
import torch
import src.knn as knn


if __name__ == "__main__":
    k = 1

    # Set up les objets mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Pour l'entrée caméra
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

        # Pour améliorer les performances, marquer l'image comme non écrivable pour la passer en référence.
        image.flags.writeable = False
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # Convertir l'image de BGR à RGB.
        results = hand.process(image)  # Appliquer le réseau.

        image.flags.writeable = True
        image = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR
        )  # On convertit aux couleurs normales.
        image_height, image_width, channels = image.shape

        if results.multi_hand_landmarks:
            for (
                hand_landmarks
            ) in results.multi_hand_landmarks:  # Pour chaque main dans l'image.
                # Calculer toutes les coordonnées des landmarks sachant que la position des landmarks est normalisée (par rapport à la taille de l'image).
                # On donne une marge de 50 pixels pour avoir la main entière dans la boîte.
                x_coords = [
                    minMax
                    for landmark in hand_landmarks.landmark
                    for minMax in (
                        int(landmark.x * image_width) - 20,
                        int(landmark.x * image_width) + 20,
                    )
                ]
                y_coords = [
                    minMax
                    for landmark in hand_landmarks.landmark
                    for minMax in (
                        int(landmark.y * image_height) - 25,
                        int(landmark.y * image_height) + 25,
                    )
                ]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                image = cv2.rectangle(
                    image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2
                )

                # Dessiner les annotations des mains sur l'image.
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks_pos = (
                    [landmark.x for landmark in hand_landmarks.landmark]
                    + [landmark.y for landmark in hand_landmarks.landmark]
                    + [landmark.z for landmark in hand_landmarks.landmark]
                )

                landmarks_pos = torch.tensor(landmarks_pos)

                # Calculer la prédiction du modèle avec les positions des landmarks en argument.
                prediction = knn.kNN(knn.dataset, landmarks_pos, k)
                predicted_text = ""

                if 0 <= prediction <= 25:
                    predicted_text = chr(ord("A") + prediction)
                    print(
                        f"Predicted result: {chr(ord('A') + prediction)}"
                    )  # La prédiction est une lettre, on convertit en char
                elif 26 <= prediction <= 36:
                    predicted_text = chr(ord("0") + prediction - 26)
                    print(
                        f"Predicted result: {chr(ord('0') + prediction - 26)}"
                    )  # La prédiction est un chiffre, on convertit en char

                cv2.putText(
                    image,
                    predicted_text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Hands bounding boxes using MediaPipe", image)

        # Pour fermer la vidéo.
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()

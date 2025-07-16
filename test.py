import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Initialise la webcam (index 0 pour la webcam par défaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

print("Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture frame webcam")
        break

    # Détecte les QR codes avec pyzbar
    decoded_objects = decode(frame)

    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        print("QR code détecté :", data)

        # Points du contour
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([(p.x, p.y) for p in points], dtype=np.int32))
        else:
            hull = np.array([(p.x, p.y) for p in points], dtype=np.int32)

        # Dessine le contour et le texte
        cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        x, y = hull[0]
        cv2.putText(frame, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Affiche la frame
    cv2.imshow("QR Detection - Webcam (pyzbar)", frame)

    # Sortie si touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère les ressources
cap.release()
cv2.destroyAllWindows()

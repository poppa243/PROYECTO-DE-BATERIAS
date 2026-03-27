import cv2
import numpy as np

# Contadores
contador_AAA = 0
contador_AA = 0
contador_9V = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame")
        break

    cv2.imshow("Camara", frame)

    key = cv2.waitKey(1) & 0xFF

    # Presiona ESPACIO para capturar
    if key == 32:
        print("Imagen capturada")

        # ROI (ajusta según tu cámara)
        x, y, w, h = 350, 150, 600, 450
        pila_roi = frame[y:y+h, x:x+w]

        # Convertir a HSV
        pila_hsv = cv2.cvtColor(pila_roi, cv2.COLOR_BGR2HSV)

        # Rangos rojo
        lim_inf1 = np.array([0, 50, 120])
        lim_sup1 = np.array([4, 255, 255])

        lim_inf2 = np.array([165, 50, 100])
        lim_sup2 = np.array([179, 255, 255])

        mascara1 = cv2.inRange(pila_hsv, lim_inf1, lim_sup1)
        mascara2 = cv2.inRange(pila_hsv, lim_inf2, lim_sup2)

        mascara = cv2.bitwise_or(mascara1, mascara2)

        # Morfología
        kernel = np.ones((7, 7), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contornos) > 0:
            contorno_max = max(contornos, key=cv2.contourArea)
            area = cv2.contourArea(contorno_max)

            # Clasificación
            if area < 13000:
                tipo = "AAA"
                contador_AAA += 1
            elif area <= 20000:
                tipo = "AA"
                contador_AA += 1
            else:
                tipo = "9V"
                contador_9V += 1

            # Dibujar contorno
            cv2.drawContours(pila_roi, [contorno_max], -1, (0,255,0), 2)

            # Mostrar info
            cv2.putText(pila_roi, f"{tipo} Area:{int(area)}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Mostrar conteo total
        cv2.putText(pila_roi, f"AAA: {contador_AAA}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(pila_roi, f"AA: {contador_AA}", (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(pila_roi, f"9V: {contador_9V}", (20,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Deteccion", pila_roi)

    # Presiona ESC para salir
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
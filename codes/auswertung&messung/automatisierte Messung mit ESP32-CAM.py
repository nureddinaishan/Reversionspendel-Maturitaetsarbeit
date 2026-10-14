import cv2
import numpy as np
import time
import csv
from datetime import datetime
from collections import deque

# ============================================================
# EINSTELLUNGEN
# ============================================================

STREAM_URL = "http://172.20.10.2:81/stream"

# Mindestfläche des roten Markers
MIN_MARKER_AREA = 80

# Glättung der Markerposition
GLAETTUNG_FRAMES = 5

# Anzahl Frames für die Ruhe-/Nullkalibrierung
NULL_FRAMES = 50

# Ab dieser Auslenkung kann ein Loslassen erkannt werden
START_SCHWELLE_PX = 8

# Wie weit der Marker nach dem grössten Ausschlag wieder Richtung
# Nullpunkt laufen muss, damit "Loslassen" erkannt wird
LOSLASS_SCHWELLE_PX = 4

# Anzahl aufeinanderfolgender Frames zur Bestätigung des Loslassens
LOSLASS_BESTAETIGUNG_FRAMES = 3

# Hysterese um den Nullpunkt gegen Mehrfachzählungen
NULL_HYSTERESE_PX = 2

# Gewünschte Anzahl vollständiger Perioden
ANZAHL_SCHWINGUNGEN = 10

# Für die automatische Kreis-/Winkelbestimmung
MIN_BOGENPUNKTE = 12
MIN_BOGEN_SPANNWEITE_PX = 10
MAX_BOGENPUNKTE = 250

# Abstand vom Drehpunkt bis zur Mitte des roten Markers
MARKER_ABSTAND_CM = 12.8


# ============================================================
# VARIABLEN
# ============================================================

marker_x_history = deque(maxlen=GLAETTUNG_FRAMES)
marker_y_history = deque(maxlen=GLAETTUNG_FRAMES)

x_null = None
y_null = None

kalibrierung_aktiv = False
kalibrierwerte = []

# Automatisch bestimmter Kreismittelpunkt = Aufhängepunkt
pivot_x = None
pivot_y = None
kreisradius_px = None

bogenpunkte = deque(maxlen=MAX_BOGENPUNKTE)

aktueller_winkel = 0.0
startwinkel = None

max_abs_dx = 0.0
winkel_am_maximum = 0.0
loslass_counter = 0

bewegung_erkannt = False
messung_aktiv = False
messung_fertig = False

letzte_seite = None
vorherige_zeit = None
vorheriges_dx = None

nulldurchgaenge = []
zeiten = []
positionen = []


# ============================================================
# ROTEN MARKER ERKENNEN
# ============================================================

def finde_roten_marker(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tolerante Rot-Erkennung:
    # auch dunkleres und helleres/ausgewascheneres Rot zulassen
    lower1 = np.array([0, 70, 30])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([165, 70, 30])
    upper2 = np.array([180, 255, 255])

    maske1 = cv2.inRange(hsv, lower1, upper1)
    maske2 = cv2.inRange(hsv, lower2, upper2)
    maske = cv2.bitwise_or(maske1, maske2)

    kernel = np.ones((3, 3), np.uint8)

    maske = cv2.morphologyEx(
        maske,
        cv2.MORPH_OPEN,
        kernel
    )

    maske = cv2.morphologyEx(
        maske,
        cv2.MORPH_CLOSE,
        kernel
    )

    konturen, _ = cv2.findContours(
        maske,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not konturen:
        return None

    groesste = max(
        konturen,
        key=cv2.contourArea
    )

    if cv2.contourArea(groesste) < MIN_MARKER_AREA:
        return None

    M = cv2.moments(groesste)

    if M["m00"] == 0:
        return None

    x = M["m10"] / M["m00"]
    y = M["m01"] / M["m00"]

    return x, y


# ============================================================
# MARKER GLÄTTEN
# ============================================================

def glaette_marker(marker):

    marker_x_history.append(marker[0])
    marker_y_history.append(marker[1])

    x = float(np.mean(marker_x_history))
    y = float(np.mean(marker_y_history))

    return x, y


# ============================================================
# KREIS AUS MARKERBAHN BESTIMMEN
# ============================================================

def bestimme_kreis(punkte):
    """
    Bestimmt den Kreis der Markerbahn per Least-Squares-Fit.
    Der Kreismittelpunkt entspricht bei einer Pendelbewegung
    dem Aufhängepunkt.
    """

    if len(punkte) < MIN_BOGENPUNKTE:
        return None

    arr = np.asarray(punkte, dtype=float)
    xs = arr[:, 0]
    ys = arr[:, 1]

    # Ein sehr kleiner Bogen ist numerisch zu unsicher.
    spannweite = max(np.ptp(xs), np.ptp(ys))

    if spannweite < MIN_BOGEN_SPANNWEITE_PX:
        return None

    A = np.column_stack(
        (
            2.0 * xs,
            2.0 * ys,
            np.ones_like(xs)
        )
    )

    b = xs * xs + ys * ys

    try:
        loesung, _, _, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    cx, cy, c = loesung

    r2 = c + cx * cx + cy * cy

    if r2 <= 0:
        return None

    radius = float(np.sqrt(r2))

    # Unplausible Fits verwerfen
    if not np.isfinite(radius) or radius < 20 or radius > 10000:
        return None

    return float(cx), float(cy), radius


# ============================================================
# WINKEL RELATIV ZUR RUHEPOSITION
# ============================================================

def berechne_winkel(x, y):
    """
    Berechnet den Auslenkwinkel über die Arkus-Sinus-Funktion.

    Der rote Marker ist 12.8 cm vom Drehpunkt entfernt.
    Der automatisch bestimmte Kreisradius in Pixeln dient zur
    Umrechnung der horizontalen Pixelverschiebung in Zentimeter.

    Ruheposition x_null entspricht definitionsgemäss 0°.
    """

    if (
        x_null is None
        or kreisradius_px is None
        or kreisradius_px <= 0
    ):
        return 0.0

    # Horizontale Verschiebung relativ zur Ruheposition
    dx_px = x - x_null

    # Umrechnung Pixel -> cm anhand des bekannten Radius von 12.8 cm
    cm_pro_pixel = MARKER_ABSTAND_CM / kreisradius_px
    dx_cm = dx_px * cm_pro_pixel

    # Geometrie:
    # sin(theta) = Gegenkathete / Radius = dx / 12.8 cm
    argument = dx_cm / MARKER_ABSTAND_CM

    # Wegen Messrauschen numerisch auf den gültigen arcsin-Bereich begrenzen
    argument = np.clip(argument, -1.0, 1.0)

    winkel_rad = np.arcsin(argument)

    return float(np.degrees(winkel_rad))


# ============================================================
# CSV SPEICHERN
# ============================================================

def speichere_csv():

    zeitstempel = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    dateiname = (
        f"pendelmessung_{zeitstempel}.csv"
    )

    with open(
        dateiname,
        "w",
        newline=""
    ) as datei:

        writer = csv.writer(
            datei,
            delimiter=";"
        )

        writer.writerow([
            "Zeit_s",
            "x_relativ_px"
        ])

        if len(nulldurchgaenge) > 0:
            t0 = nulldurchgaenge[0]
        else:
            t0 = 0

        for t, x in zip(
            zeiten,
            positionen
        ):

            writer.writerow([
                f"{t - t0:.6f}",
                f"{x:.6f}"
            ])

    print()
    print(
        f"Messdaten gespeichert: {dateiname}"
    )


# ============================================================
# MESSUNG AUSWERTEN
# ============================================================

def auswertung():

    if len(nulldurchgaenge) < 2:
        return

    perioden = []

    for i in range(
        len(nulldurchgaenge) - 1
    ):

        T = (
            nulldurchgaenge[i + 1]
            - nulldurchgaenge[i]
        )

        perioden.append(T)

    gesamtzeit = (
        nulldurchgaenge[-1]
        - nulldurchgaenge[0]
    )

    T_mittel = (
        gesamtzeit
        / ANZAHL_SCHWINGUNGEN
    )

    print()
    print("======================================")
    print("          MESSUNG ABGESCHLOSSEN")
    print("======================================")
    print()

    print(
        f"Schwingungen: "
        f"{ANZAHL_SCHWINGUNGEN}"
    )

    if startwinkel is not None:
        print(
            f"Startwinkel: "
            f"{abs(startwinkel):.2f} Grad"
        )

    print(
        f"Gesamtzeit: "
        f"{gesamtzeit:.6f} s"
    )

    print(
        f"Mittlere Periodendauer: "
        f"{T_mittel:.6f} s"
    )

    if len(perioden) > 1:

        std = np.std(
            perioden,
            ddof=1
        )

        print(
            f"Standardabweichung: "
            f"{std:.6f} s"
        )

    print()

    for i, T in enumerate(
        perioden,
        start=1
    ):

        print(
            f"T{i:02d} = {T:.6f} s"
        )

    speichere_csv()

    print()
    print(
        "Programm bleibt geöffnet. "
        "Für eine neue Messung r drücken."
    )


# ============================================================
# MESSUNG ZURÜCKSETZEN
# ============================================================

def reset_messung():

    global pivot_x
    global pivot_y
    global kreisradius_px
    global aktueller_winkel
    global startwinkel
    global max_abs_dx
    global winkel_am_maximum
    global loslass_counter
    global bewegung_erkannt
    global messung_aktiv
    global messung_fertig
    global letzte_seite
    global vorherige_zeit
    global vorheriges_dx

    # Kreis wird für jede neue Messung erneut automatisch gelernt.
    pivot_x = None
    pivot_y = None
    kreisradius_px = None

    bogenpunkte.clear()

    aktueller_winkel = 0.0
    startwinkel = None

    max_abs_dx = 0.0
    winkel_am_maximum = 0.0
    loslass_counter = 0

    bewegung_erkannt = False
    messung_aktiv = False
    messung_fertig = False

    letzte_seite = None
    vorherige_zeit = None
    vorheriges_dx = None

    nulldurchgaenge.clear()
    zeiten.clear()
    positionen.clear()

    print()
    print(
        "Messung zurückgesetzt."
    )

    if x_null is not None:
        print(
            "Pendel langsam auslenken und anschließend loslassen."
        )


# ============================================================
# KAMERA ÖFFNEN
# ============================================================

print(
    "Verbinde mit ESP32-CAM ..."
)

cap = cv2.VideoCapture(
    STREAM_URL
)

if not cap.isOpened():

    print(
        "FEHLER: Stream konnte "
        "nicht geöffnet werden."
    )

    input(
        "Enter zum Schliessen ..."
    )
    raise SystemExit

print(
    "Kamera verbunden."
)

print()
print("======================================")
print("        REVERSIONSPENDEL-MESSUNG")
print("======================================")
print()
print("z = Nullposition kalibrieren")
print("r = neue Messung")
print("q = Programm beenden")
print()
print(
    "1. Pendel ruhig senkrecht hängen lassen."
)
print(
    "2. z drücken."
)
print(
    "3. Pendel langsam auslenken."
)
print(
    "4. Winkel im Bild ablesen und Pendel loslassen."
)
print(
    "5. 10 Perioden werden automatisch gemessen."
)

cv2.namedWindow(
    "Pendel Messung"
)


# ============================================================
# HAUPTSCHLEIFE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "WARNUNG: Kein Kamerabild empfangen. "
            "Versuche weiter ..."
        )

        time.sleep(0.05)
        continue

    marker = finde_roten_marker(
        frame
    )

    aktuelle_zeit = time.perf_counter()

    # ========================================================
    # MARKER ERKANNT
    # ========================================================

    if marker is not None:

        marker = glaette_marker(
            marker
        )

        x, y = marker

        marker_int = (
            int(round(x)),
            int(round(y))
        )

        cv2.circle(
            frame,
            marker_int,
            14,
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            marker_int,
            3,
            (0, 255, 0),
            -1
        )


        # ====================================================
        # NULLKALIBRIERUNG
        # ====================================================

        if kalibrierung_aktiv:

            kalibrierwerte.append(
                (x, y)
            )

            cv2.putText(
                frame,
                (
                    f"Kalibrierung "
                    f"{len(kalibrierwerte)}"
                    f"/{NULL_FRAMES}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            if (
                len(kalibrierwerte)
                >= NULL_FRAMES
            ):

                arr = np.asarray(
                    kalibrierwerte,
                    dtype=float
                )

                x_null = float(
                    np.mean(arr[:, 0])
                )

                y_null = float(
                    np.mean(arr[:, 1])
                )

                kalibrierung_aktiv = False

                reset_messung()

                print()
                print(
                    "Nullposition kalibriert."
                )

                print(
                    f"x0 = {x_null:.3f} px"
                )

                print(
                    f"y0 = {y_null:.3f} px"
                )

                print(
                    "Ruheposition entspricht jetzt 0.00 Grad."
                )

                print()
                print(
                    "Pendel langsam auslenken."
                )


        # ====================================================
        # NACH KALIBRIERUNG
        # ====================================================

        elif x_null is not None:

            dx = x - x_null

            # -----------------------------------------------
            # Vor dem Loslassen: Markerbahn sammeln und
            # Aufhängepunkt automatisch aus Kreisbogen lernen
            # -----------------------------------------------

            if (
                not bewegung_erkannt
                and not messung_fertig
            ):

                # Ruhepunkt immer mit im Fit behalten.
                if len(bogenpunkte) == 0:
                    for _ in range(5):
                        bogenpunkte.append(
                            (x_null, y_null)
                        )

                # Nur neue, ausreichend unterschiedliche Punkte aufnehmen
                if (
                    len(bogenpunkte) == 0
                    or np.hypot(
                        x - bogenpunkte[-1][0],
                        y - bogenpunkte[-1][1]
                    ) >= 0.7
                ):
                    bogenpunkte.append(
                        (x, y)
                    )

                kreis = bestimme_kreis(
                    bogenpunkte
                )

                if kreis is not None:

                    pivot_x, pivot_y, kreisradius_px = kreis

                    aktueller_winkel = berechne_winkel(
                        x,
                        y
                    )

                else:
                    # Solange der Kreisbogen noch zu klein ist, bleibt
                    # die Anzeige nahe der Ruheposition bei 0°.
                    aktueller_winkel = 0.0


                # -------------------------------------------
                # Maximalen Ausschlag / Startwinkel merken
                # -------------------------------------------

                if abs(dx) > max_abs_dx:

                    max_abs_dx = abs(dx)

                    if pivot_x is not None:
                        winkel_am_maximum = (
                            aktueller_winkel
                        )

                    loslass_counter = 0

                # -------------------------------------------
                # LOSLASSEN AUTOMATISCH ERKENNEN
                # -------------------------------------------

                if (
                    max_abs_dx >= START_SCHWELLE_PX
                    and abs(dx)
                    <= max_abs_dx - LOSLASS_SCHWELLE_PX
                ):

                    loslass_counter += 1

                else:

                    loslass_counter = 0


                if (
                    loslass_counter
                    >= LOSLASS_BESTAETIGUNG_FRAMES
                ):

                    bewegung_erkannt = True

                    if pivot_x is not None:
                        startwinkel = (
                            winkel_am_maximum
                        )
                    else:
                        startwinkel = None

                    print()
                    print(
                        "Pendel automatisch als losgelassen erkannt."
                    )

                    if startwinkel is not None:
                        print(
                            f"Startwinkel: "
                            f"{abs(startwinkel):.2f} Grad"
                        )

                    print(
                        "Warte auf ersten Nulldurchgang ..."
                    )

                    # Aktuellen Zustand als Ausgangspunkt setzen.
                    if dx <= -NULL_HYSTERESE_PX:
                        letzte_seite = -1
                    elif dx >= NULL_HYSTERESE_PX:
                        letzte_seite = 1
                    else:
                        letzte_seite = None

                    vorherige_zeit = aktuelle_zeit
                    vorheriges_dx = dx


            # -----------------------------------------------
            # Nach Loslassen: aktuellen Winkel weiter anzeigen
            # -----------------------------------------------

            elif (
                bewegung_erkannt
                and pivot_x is not None
            ):

                aktueller_winkel = berechne_winkel(
                    x,
                    y
                )


            # ================================================
            # ANZEIGE
            # ================================================

            # Kleine Winkelanzeige oben links.
            # Die kalibrierte Ruheposition ist definitionsgemäss 0°.
            if pivot_x is not None:

                cv2.putText(
                    frame,
                    f"{aktueller_winkel:+.2f} deg",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            else:

                cv2.putText(
                    frame,
                    "Winkel: 0.00 deg",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1
                )

            # Nullposition klein markieren
            cv2.circle(
                frame,
                (
                    int(round(x_null)),
                    int(round(y_null))
                ),
                4,
                (255, 255, 255),
                1
            )

            # Optional: automatisch gefundener Aufhängepunkt klein markieren
            if pivot_x is not None:

                cv2.circle(
                    frame,
                    (
                        int(round(pivot_x)),
                        int(round(pivot_y))
                    ),
                    4,
                    (255, 0, 255),
                    1
                )


            # ================================================
            # NULLDURCHGÄNGE / 10 PERIODEN AUTOMATISCH MESSEN
            # ================================================

            if (
                bewegung_erkannt
                and not messung_fertig
            ):

                zeiten.append(
                    aktuelle_zeit
                )

                positionen.append(
                    dx
                )

                aktuelle_seite = None

                if dx <= -NULL_HYSTERESE_PX:
                    aktuelle_seite = -1

                elif dx >= NULL_HYSTERESE_PX:
                    aktuelle_seite = 1


                # Nur negativ -> positiv zählen:
                # derselbe Nulldurchgang einmal pro voller Periode.
                if aktuelle_seite is not None:

                    if (
                        letzte_seite == -1
                        and aktuelle_seite == 1
                        and vorheriges_dx is not None
                        and vorherige_zeit is not None
                        and abs(dx - vorheriges_dx) > 1e-9
                    ):

                        t_cross = (
                            vorherige_zeit
                            - vorheriges_dx
                            * (
                                aktuelle_zeit
                                - vorherige_zeit
                            )
                            / (
                                dx
                                - vorheriges_dx
                            )
                        )

                        nulldurchgaenge.append(
                            t_cross
                        )

                        if len(nulldurchgaenge) == 1:

                            messung_aktiv = True

                            print()
                            print(
                                "Messung automatisch gestartet."
                            )

                            print(
                                f"Schwingung 0 / "
                                f"{ANZAHL_SCHWINGUNGEN}"
                            )

                        else:

                            n = (
                                len(nulldurchgaenge)
                                - 1
                            )

                            print(
                                f"Schwingung "
                                f"{n} / "
                                f"{ANZAHL_SCHWINGUNGEN}"
                            )

                            if (
                                n
                                >= ANZAHL_SCHWINGUNGEN
                            ):

                                messung_aktiv = False
                                messung_fertig = True

                                print()
                                print(
                                    "10 Schwingungen erreicht."
                                )

                                auswertung()

                    letzte_seite = (
                        aktuelle_seite
                    )

                vorherige_zeit = (
                    aktuelle_zeit
                )

                vorheriges_dx = dx


        # ====================================================
        # STATUS
        # ====================================================

        if x_null is None:

            status = (
                "NICHT KALIBRIERT"
            )

        elif messung_fertig:

            status = (
                "MESSUNG FERTIG - r FUER NEU"
            )

        elif messung_aktiv:

            n = max(
                0,
                len(nulldurchgaenge) - 1
            )

            status = (
                f"MESSUNG {n}/"
                f"{ANZAHL_SCHWINGUNGEN}"
            )

        elif bewegung_erkannt:

            status = (
                "LOSGELASSEN"
            )

        elif pivot_x is None:

            status = (
                "LANGSAM AUSLENKEN"
            )

        else:

            status = (
                "STARTWINKEL EINSTELLEN"
            )

        cv2.putText(
            frame,
            status,
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )


    # ========================================================
    # MARKER NICHT ERKANNT
    # ========================================================

    else:

        cv2.putText(
            frame,
            "ROTER MARKER NICHT ERKANNT",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            1
        )


    # ========================================================
    # BILD
    # ========================================================

    cv2.imshow(
        "Pendel Messung",
        frame
    )

    taste = (
        cv2.waitKey(1)
        & 0xFF
    )


    # ========================================================
    # NULLPOSITION KALIBRIEREN
    # ========================================================

    if taste == ord("z"):

        if marker is None:

            print()
            print(
                "Marker wird nicht erkannt."
            )

        else:

            kalibrierwerte.clear()
            marker_x_history.clear()
            marker_y_history.clear()

            kalibrierung_aktiv = True

            print()
            print(
                "Nullkalibrierung gestartet."
            )

            print(
                "Pendel bitte ruhig senkrecht hängen lassen ..."
            )


    # ========================================================
    # NEUE MESSUNG
    # ========================================================

    elif taste == ord("r"):

        if x_null is None:

            print()
            print(
                "Zuerst mit z kalibrieren."
            )

        else:

            marker_x_history.clear()
            marker_y_history.clear()
            reset_messung()


    # ========================================================
    # PROGRAMM BEENDEN
    # ========================================================

    elif taste == ord("q"):

        break


cap.release()
cv2.destroyAllWindows()

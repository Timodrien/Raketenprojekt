# Höhenforschungsraketen-Simulator

Ein Python/Tkinter-Tool zur Auslegung und Simulation einer selbstgebauten Höhenforschungsrakete – von der Motorauslegung über den Flug bis zur Geometriezeichnung, alles in einer GUI.

## Motivation

Ziel dieses Projekts ist der Bau einer eigenen Höhenforschungsrakete (Hobby/Amateurraketentechnik). Statt Parameter einzeln durchzurechnen oder in verstreuten Excel-Sheets zu jonglieren, bündelt dieses Tool alle relevanten Berechnungen – Feststoffmotor-Auslegung, Flugbahnsimulation, Aerodynamik und Stabilität – in einer einzigen Anwendung mit direktem visuellem Feedback.

## Features

Das Tool ist in sieben Tabs organisiert:

- **Motor Design** – Simulation eines Feststoffmotors (KNSB-basierte Treibstoffe) über die Brenndauer: Schubkurve, Kammerdruck, Treibstoffmasse-Verlauf. Berechnung basiert auf Burn-Rate-Gesetz (`r = a·P^n`) und liefert automatisch die Motorklasse (A–O).
- **Flugrechner** – Einstufige Flugsimulation (Höhe, Geschwindigkeit, Beschleunigung über Zeit) auf Basis der Motor-Schubkurve, mit atmosphärischem Dichtemodell und transsonischem Cw-Anstieg.
- **Zweistufig** – Simulation einer zweistufigen Rakete inkl. Stufentrennung, Coast-Phase und getrennten Motor-/Massendaten je Stufe.
- **Aerodynamik** – Cw-Wert-Schätzung basierend auf Nosecone-Form (Kegel, Ogive, Von-Kármán-Ogive, parabolisch, elliptisch, Halbkugel), Finess-Verhältnis und Rumpflänge.
- **Stabilität** – Vereinfachte Barrowman-Methode zur Druckpunkt-Berechnung (CP) und Abgleich mit dem Schwerpunkt (CG) zur Bestimmung der Kalibern-Stabilität.
- **Treibstoff** – Datenbank und Vergleich verschiedener Feststofftreibstoff-Mischungen (Dichte, Burn-Rate-Koeffizienten, spezifischer Impuls).
- **Geometrie** – Automatisch generierte, bemaßte Seitenansicht der kompletten Rakete (Nosecone, Nutzlast, Motorsegmente, Fins, Adapter, Trennebene) inkl. Massen- und Längenübersicht.

## Physikalisches Modell

- **Motorabbrand**: Endburner-/Segment-Geometrie mit `Ab`-Abbrandfläche, Kammerdruck aus `P = (ρ·a·Ab·c*/At)^(1/(1-n))`
- **Atmosphäre**: Exponentielles Dichtemodell (`ρ(h) = 1.225·e^(-h/8500)`), höhenabhängige Schallgeschwindigkeit
- **Aerodynamik**: Cw-Modell mit transsonischem Anstieg (Widerstandsanstieg ab Mach 0.8)
- **Flugintegration**: Explizite Euler-Integration der Bewegungsgleichung `a = (F_Schub − F_Widerstand − m·g) / m`
- **Stabilität**: Barrowman-Gleichungen für Nosecone- und Finnen-Beitrag zum CN und zur CP-Lage

## Voraussetzungen

```bash
pip install numpy matplotlib scipy
```

Tkinter wird meist mit der Python-Standardinstallation mitgeliefert (unter Linux ggf. `python3-tk` separat installieren).

## Nutzung

```bash
python Simulationsprogramm.py
```

Die GUI öffnet sich direkt mit einer Beispielgeometrie. Parameter (Treibstoff, Motorgeometrie, Rumpfdurchmesser, Finnen etc.) lassen sich in den jeweiligen Tabs anpassen; die Diagramme aktualisieren sich per Knopfdruck.

## Status

Aktiver Entwicklungsstand – Auslegungswerkzeug für die eigene Höhenforschungsrakete. Ergebnisse sind vereinfachte physikalische Näherungen (kein CFD, kein OpenRocket-Niveau) und dienen der Vorauslegung; für den finalen Bau sollten kritische Werte (v. a. Stabilität, Strukturfestigkeit) zusätzlich verifiziert werden.

## Roadmap / Ideen

- [ ] Export der Simulationsergebnisse (CSV/PDF)
- [ ] Wind- und Böenmodell im Flugrechner
- [ ] Realere Cw-Datenbasis (CFD- oder Windkanaldaten statt Näherung)
- [ ] Import von echten Motor-Thrust-Curves (.eng-Dateien)


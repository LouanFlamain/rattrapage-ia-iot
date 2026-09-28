# Reconnaissance de mouvements IoT — rattrapage IA

Mini-expérimentation de reconnaissance d'activités humaines à partir de mesures
inertielles. Elle reproduit sur ordinateur une chaîne destinée, à terme, à un
microcontrôleur tel qu'un ESP32 : préparation des mesures, classification légère,
export TensorFlow Lite et inférence simulée en flux.

## 1. Dataset retenu

Le dataset public [MotionSense](https://github.com/mmalekzadeh/motion-sense)
est utilisé pour cette expérimentation. Il contient des séries temporelles issues
des capteurs inertiels d'un iPhone 6s porté dans la poche avant de 24 participants.
Les mesures sont échantillonnées à **50 Hz**. Six activités sont disponibles :

| Identifiant | Activité |
| --- | --- |
| `dws` | Descente d'escaliers |
| `ups` | Montée d'escaliers |
| `wlk` | Marche |
| `jog` | Jogging |
| `sit` | Position assise |
| `std` | Position debout |

Ce choix satisfait largement la contrainte d'au moins trois classes. Il est
pertinent dans un contexte IoT, car il fournit des séquences comparables à celles
d'une centrale inertielle : les futures fenêtres temporelles pourront être reçues
progressivement, comme depuis un capteur embarqué.

La source utilisée est l'archive `A_DeviceMotion_data.zip`, téléchargée depuis le
[répertoire de données officiel](https://github.com/mmalekzadeh/motion-sense/tree/master/data).
Elle réunit les mesures pertinentes de l'accéléromètre et du gyroscope.

## 2. Nettoyage des données

### Signaux retenus

Le modèle utilisera neuf canaux, dans cet ordre fixe :

1. `gravity.x`, `gravity.y`, `gravity.z` ;
2. `userAcceleration.x`, `userAcceleration.y`, `userAcceleration.z` ;
3. `rotationRate.x`, `rotationRate.y`, `rotationRate.z`.

Les trois premières mesures aident notamment à différencier les positions assise
et debout. Les six suivantes représentent respectivement l'accélération créée par
le mouvement et la vitesse angulaire. Les colonnes d'orientation (`attitude.*`) et
la colonne d'index technique `Unnamed: 0` ne sont pas utilisées : elles ne sont
pas nécessaires à cette première version légère.

### Méthode reproductible

Le script [`src/clean_data.py`](src/clean_data.py) traite chaque enregistrement
indépendamment. Il :

1. ignore les fichiers techniques macOS et vérifie la présence des neuf colonnes ;
2. convertit les valeurs en nombres finis ;
3. conserve la cadence et toutes les lignes — des valeurs identiques ne sont pas
   supprimées, car elles sont légitimes pendant une posture statique ;
4. remplace uniquement les cellules invalides ou manquantes par interpolation
   linéaire au sein du même enregistrement (ou par la valeur valide de bord) ;
5. ajoute les métadonnées `sample_index`, `activity`, `class_id`, `trial` et
   `subject_id`, puis écrit un CSV propre par enregistrement ;
6. génère un manifeste et un audit chiffré des corrections effectuées.

La normalisation est intentionnellement reportée à l'étape d'entraînement : ses
statistiques seront calculées **uniquement sur les participants d'entraînement**.
Cela empêche une fuite d'information depuis les données de validation ou de test.

### Exécution

Télécharger puis extraire le dataset dans `data/raw/` :

```bash
curl --fail --location --output data/raw/A_DeviceMotion_data.zip \
  https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip
unzip -q data/raw/A_DeviceMotion_data.zip -d data/raw/
python3 src/clean_data.py
```

Les résultats locaux (non versionnés) sont :

```text
data/processed/
├── clean_recordings/    # Un CSV nettoyé par activité / essai / participant
├── manifest.csv         # Index de tous les enregistrements utilisables
└── cleaning_audit.json  # Volumes, valeurs imputées et éventuels rejets
```

### Résultat obtenu

L'exécution du nettoyage sur l'archive officielle a produit **360
enregistrements** et **1 412 865 échantillons**. Les contrôles de validité n'ont
révélé aucune ligne invalide ni cellule à imputer ; aucun enregistrement n'a été
écarté. La répartition, naturellement déséquilibrée car les essais n'ont pas tous
la même durée, est la suivante :

| Activité | Échantillons |
| --- | ---: |
| Descente (`dws`) | 131 856 |
| Montée (`ups`) | 157 285 |
| Marche (`wlk`) | 344 288 |
| Jogging (`jog`) | 134 231 |
| Assis (`sit`) | 338 778 |
| Debout (`std`) | 306 427 |

L'audit complet est régénérable dans `data/processed/cleaning_audit.json`. Ces
volumes ne préjugent pas encore de l'équilibre des futures fenêtres : cette
question sera traitée lors de la préparation des jeux d'entraînement et de test.

## Étapes restantes

1. Découper les enregistrements en fenêtres temporelles et séparer les
   participants entre entraînement, validation et test.
2. Entraîner et évaluer un modèle léger ; analyser sa matrice de confusion.
3. Convertir le modèle en TensorFlow Lite, mesurer sa taille et son temps
   d'inférence, puis simuler l'arrivée progressive des mesures.

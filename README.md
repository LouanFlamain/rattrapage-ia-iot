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

## 3. Préparation des fenêtres et entraînement

### Séparation correcte des données

La séparation est faite par **participant**, avant le découpage en fenêtres. Ce
choix est plus réaliste qu'un tirage aléatoire de fenêtres : des séquences du même
participant sont proches et rendraient les résultats artificiellement optimistes
si elles figuraient à la fois dans les jeux d'entraînement et de test.

Avec une graine fixée à `42`, les 24 participants sont répartis ainsi :

| Jeu | Participants | Rôle |
| --- | --- | --- |
| Entraînement | 1, 4, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24 | Apprentissage des poids et de la normalisation |
| Validation | 3, 5, 15, 23 | Arrêt anticipé et choix du meilleur état du modèle |
| Test | 2, 9, 14, 22 | Évaluation finale, jamais utilisée pendant l'apprentissage |

Chaque enregistrement est découpé indépendamment en fenêtres de **100
échantillons** (2 secondes à 50 Hz), avec un décalage de 50 échantillons (1
seconde). Aucune fenêtre ne traverse donc la frontière entre deux essais. Les
moyennes et écarts-types de normalisation sont estimés à partir des signaux bruts
des seuls participants d'entraînement, puis intégrés directement au modèle.

### Modèle léger

Le script [`src/train_model.py`](src/train_model.py) entraîne un petit CNN 1D :

```text
Fenêtre (100 × 9)
→ Normalisation embarquée
→ Conv1D 16 filtres (noyau 5) + max-pooling
→ Conv1D 24 filtres (noyau 3)
→ GlobalAveragePooling → Dense 16 → Softmax 6 classes
```

Cette architecture ne comporte qu'environ **2 400 paramètres entraînables**. La
pondération des classes compense les durées inégales des essais pendant
l'apprentissage, sans modifier le jeu de test.

### Exécution

Créer l'environnement local, installer TensorFlow et lancer l'entraînement :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train_model.py --epochs 25
```

Le script écrit le modèle Keras dans `models/activity_cnn.keras` et toutes les
informations nécessaires à la reproductibilité (split, statistiques de
normalisation, nombre de fenêtres, historique et résultats) dans
`models/training_metadata.json`.

### Résultat de l'entraînement

L'entraînement s'est arrêté automatiquement après **16 époques** (arrêt
anticipé sur l'accuracy de validation). Le découpage a produit 18 683 fenêtres
d'entraînement, 4 487 fenêtres de validation et 4 550 fenêtres de test. Le
modèle comporte **2 414 paramètres entraînables**.

| Jeu | Accuracy mesurée |
| --- | ---: |
| Entraînement | 98,46 % |
| Validation | 90,60 % |
| Test (participants totalement écartés de l'apprentissage) | 93,69 % |

La valeur de test est conservée telle quelle : aucun réglage d'architecture ou
d'hyperparamètre n'a été sélectionné à partir de ce jeu. L'étape suivante
complétera son analyse avec le F1-score et la matrice de confusion.

## 4. Évaluation — consigne 3

Le script [`src/evaluate_model.py`](src/evaluate_model.py) recharge le modèle
entraîné et évalue uniquement les quatre participants du jeu de test (2, 9, 14
et 22). Il reconstruit exactement les mêmes fenêtres de deux secondes, puis
produit :

- l'accuracy globale ;
- precision, recall et F1-score par activité ;
- F1 macro et F1 pondéré, plus robustes que l'accuracy en présence de classes
  dont la durée totale diffère ;
- une matrice de confusion, avec les lignes correspondant aux vraies activités
  et les colonnes aux activités prédites ;
- une liste des confusions les plus fréquentes.

Exécution :

```bash
.venv/bin/python src/evaluate_model.py
```

Les artefacts sont écrits dans `reports/` : `evaluation.json`,
`classification_report.csv` et `confusion_matrix.csv`.

### Résultats sur le jeu de test

L'évaluation porte sur **4 550 fenêtres** des participants 2, 9, 14 et 22. Ces
participants ne font pas partie du jeu d'entraînement ni du jeu de validation.

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 93,69 % |
| Precision macro | 90,67 % |
| Recall macro | 90,86 % |
| F1-score macro | 90,72 % |
| F1-score pondéré | 93,69 % |

Le F1 macro est inférieur à l'accuracy car il donne le même poids aux six
activités ; il met donc mieux en évidence les performances plus faibles sur les
deux activités d'escaliers, moins représentées et plus difficiles à séparer.

| Activité | Precision | Recall | F1-score | Fenêtres |
| --- | ---: | ---: | ---: | ---: |
| Descente (`dws`) | 71,57 % | 76,44 % | 73,92 % | 382 |
| Montée (`ups`) | 82,43 % | 75,93 % | 79,05 % | 482 |
| Marche (`wlk`) | 93,99 % | 94,53 % | 94,26 % | 1 207 |
| Jogging (`jog`) | 96,30 % | 99,77 % | 98,00 % | 443 |
| Assis (`sit`) | 99,73 % | 99,91 % | 99,82 % | 1 126 |
| Debout (`std`) | 100,00 % | 98,57 % | 99,28 % | 910 |

### Matrice de confusion

Lignes : vraie activité. Colonnes : activité prédite.

| Vrai \ Prédit | dws | ups | wlk | jog | sit | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dws` | 292 | 50 | 24 | 16 | 0 | 0 |
| `ups` | 67 | 366 | 45 | 1 | 3 | 0 |
| `wlk` | 48 | 18 | 1 141 | 0 | 0 | 0 |
| `jog` | 0 | 1 | 0 | 442 | 0 | 0 |
| `sit` | 1 | 0 | 0 | 0 | 1 125 | 0 |
| `std` | 0 | 9 | 4 | 0 | 0 | 897 |

### Analyse des mouvements reconnus et confondus

Les postures `sit` et `std`, ainsi que le `jog`, sont très bien reconnues (F1
supérieur à 98 %). La gravité sur trois axes apporte un repère stable pour les
postures ; le jogging a aussi une signature dynamique beaucoup plus marquée que
la marche.

Les principales erreurs concernent les escaliers : 67 fenêtres de montée sont
prédites comme une descente (13,90 % des montées) et 50 fenêtres de descente sont
prédites comme une montée (13,09 % des descentes). Ces deux activités présentent
des cycles de marche similaires ; les variations de geste et d'orientation du
téléphone selon les participants expliquent probablement cette proximité. La
marche est parfois associée aux escaliers (48 fenêtres prédites `dws`, 18
prédites `ups`), ce qui confirme que les mouvements locomoteurs constituent la
limite principale du modèle. Une fenêtre plus longue ou des caractéristiques
fréquentielles pourraient améliorer cette distinction, au prix d'une latence et
d'un coût de calcul plus élevés.

## 5. Export Edge AI et simulation IoT — consigne 4

### Export TensorFlow Lite

Le script [`src/export_tflite.py`](src/export_tflite.py) convertit le modèle
Keras en `activity_cnn_int8.tflite`. La quantification post-entraînement est
**entièrement int8** : les poids, les activations, l'entrée et la sortie sont
quantifiés. Un échantillon équilibré de 20 fenêtres par activité, provenant
exclusivement des participants d'entraînement, sert à calibrer cette
quantification. Les ensembles de validation et de test restent donc isolés.

```bash
.venv/bin/python src/export_tflite.py
```

Le fichier `models/tflite_metadata.json` indique notamment le type et les
paramètres de quantification des tenseurs d'entrée et de sortie. Ces paramètres
sont indispensables pour convertir correctement les mesures flottantes d'un
capteur vers le format int8 du modèle.

### Simulation de flux capteur

Le script [`src/simulate_iot.py`](src/simulate_iot.py) rejoue progressivement un
CSV nettoyé comme si les neuf mesures arrivaient d'un capteur. Il mémorise 100
mesures (2 secondes), puis infère une activité toutes les 50 nouvelles mesures
(1 seconde) avec l'interpréteur TensorFlow Lite. La console affiche le mouvement
détecté et son niveau de confiance.

```bash
# Rejoue les dix premières décisions d'un enregistrement de jogging du test.
.venv/bin/python src/simulate_iot.py

# Rejoue l'intégralité d'un autre fichier à cadence réelle de 50 Hz.
.venv/bin/python src/simulate_iot.py \
  --input-file data/processed/clean_recordings/wlk_7/sub_2.csv \
  --max-predictions 0 --realtime
```

La simulation exécute exactement les étapes qu'utiliserait un objet connecté :
accumulation d'une fenêtre de données brutes, quantification int8 à l'entrée,
inférence TFLite et déquantification de la confiance en sortie. Le fichier est
lu ligne par ligne : seules les 100 dernières mesures sont conservées en mémoire.

### Résultat de l'export et de la simulation

Le modèle exporté est [`models/activity_cnn_int8.tflite`](models/activity_cnn_int8.tflite).
Il fait **10 272 octets** (environ 10,0 KiB) et utilise une entrée `(1, 100, 9)`
et une sortie `(1, 6)`, toutes deux au format `int8`. Les échelles et points zéro
de quantification sont sauvegardés dans `models/tflite_metadata.json`.

Sur les dix premières fenêtres (deux secondes, puis une décision par seconde)
de l'enregistrement `jog_9/sub_2.csv`, le simulateur a détecté `jog` dix fois
sur dix avec une confiance de 99,6 %. Cette démonstration vérifie la chaîne
complète de données : CSV capteur → buffer temporel → quantification → modèle
TFLite → libellé et confiance dans la console.

## Étape restante

Mesurer la taille du modèle TensorFlow Lite et le temps moyen d'inférence, puis
discuter sa pertinence et ses optimisations possibles pour un ESP32.

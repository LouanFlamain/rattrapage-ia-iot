# Reconnaissance d'activités avec MotionSense

Projet de rattrapage IA & IoT. Le but est de reconnaître des mouvements à partir
de mesures inertielles et de reproduire, sur ordinateur, une chaîne qui pourrait
ensuite être adaptée à un ESP32.

Le projet contient le nettoyage des données, l'entraînement, l'évaluation,
l'export TensorFlow Lite et une simulation de capteur.

Le projet a été testé avec Python 3.9 et TensorFlow 2.16.1.

## Contenu du dossier

- src/clean_data.py : nettoyage des CSV MotionSense ;
- src/train_model.py : préparation des fenêtres et entraînement ;
- src/evaluate_model.py : métriques et matrice de confusion ;
- src/export_tflite.py : export du modèle quantifié ;
- src/simulate_iot.py : simulation de données arrivant progressivement ;
- src/benchmark_tflite.py : taille et temps d'inférence ;
- models/activity_cnn_int8.tflite : modèle à utiliser pour l'inférence ;
- reports/ : résultats bruts de l'évaluation et du benchmark.

## Dataset

Le dataset choisi est [MotionSense](https://github.com/mmalekzadeh/motion-sense).
Il contient des données d'accéléromètre et de gyroscope enregistrées à 50 Hz par
un iPhone porté dans la poche avant de 24 personnes.

Les six activités utilisées sont :

- dws : descente d'escaliers ;
- ups : montée d'escaliers ;
- wlk : marche ;
- jog : jogging ;
- sit : position assise ;
- std : position debout.

Les données brutes ne sont pas incluses dans le dépôt. Elles peuvent être
téléchargées depuis le dépôt officiel :

~~~~bash
mkdir -p data/raw
curl --fail --location --output data/raw/A_DeviceMotion_data.zip \
  https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip
unzip -q data/raw/A_DeviceMotion_data.zip -d data/raw/
~~~~

## Préparation des données

Le script src/clean_data.py garde neuf mesures :

~~~~text
gravity.x, gravity.y, gravity.z
userAcceleration.x, userAcceleration.y, userAcceleration.z
rotationRate.x, rotationRate.y, rotationRate.z
~~~~

La gravité aide notamment à distinguer les positions assise et debout. Les
colonnes attitude.* et l'index technique du CSV ne sont pas utilisés.

Chaque fichier est vérifié, les valeurs non numériques sont traitées comme des
valeurs manquantes et, si besoin, interpolées dans le même enregistrement. Les
lignes ne sont pas supprimées, afin de garder la fréquence d'échantillonnage.

~~~~bash
python3 src/clean_data.py
~~~~

Sur ce dataset, le nettoyage a produit 360 enregistrements, soit 1 412 865
mesures. Aucune valeur invalide n'a été trouvée : aucun fichier n'a été rejeté
et aucune valeur n'a eu besoin d'être interpolée.

## Entraînement

Les enregistrements sont découpés en fenêtres de 100 mesures, soit 2 secondes à
50 Hz. Une nouvelle prédiction peut être faite toutes les 50 mesures, donc toutes
les secondes.

La séparation est faite par participant avant de créer les fenêtres. Cela évite
que des données d'une même personne se retrouvent à la fois dans l'entraînement
et dans le test.

| Jeu | Participants |
| --- | --- |
| Entraînement | 1, 4, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24 |
| Validation | 3, 5, 15, 23 |
| Test | 2, 9, 14, 22 |

La normalisation est calculée uniquement avec les données d'entraînement. Ses
paramètres sont intégrés au modèle, ce qui évite d'avoir un prétraitement
différent pendant la simulation.

Le modèle est un petit CNN 1D :

~~~~text
Entrée (100, 9)
→ normalisation
→ Conv1D 16 filtres + max-pooling
→ Conv1D 24 filtres
→ GlobalAveragePooling
→ Dense 16
→ sortie softmax à 6 classes
~~~~

Il contient 2 414 paramètres entraînables. Les classes les moins représentées
sont pondérées pendant l'entraînement.

~~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train_model.py --epochs 25
~~~~

L'entraînement s'est arrêté après 16 époques. Il a créé 18 683 fenêtres pour
l'entraînement, 4 487 pour la validation et 4 550 pour le test.

| Jeu | Accuracy |
| --- | ---: |
| Entraînement | 98,46 % |
| Validation | 90,60 % |
| Test | 93,69 % |

Le modèle Keras et les paramètres utilisés sont enregistrés dans models/.

## Évaluation

L'évaluation est faite sur les quatre participants du jeu de test uniquement.

~~~~bash
.venv/bin/python src/evaluate_model.py
~~~~

| Mesure | Résultat |
| --- | ---: |
| Accuracy | 93,69 % |
| Precision macro | 90,67 % |
| Recall macro | 90,86 % |
| F1-score macro | 90,72 % |
| F1-score pondéré | 93,69 % |

| Activité | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Descente | 71,57 % | 76,44 % | 73,92 % |
| Montée | 82,43 % | 75,93 % | 79,05 % |
| Marche | 93,99 % | 94,53 % | 94,26 % |
| Jogging | 96,30 % | 99,77 % | 98,00 % |
| Assis | 99,73 % | 99,91 % | 99,82 % |
| Debout | 100,00 % | 98,57 % | 99,28 % |

Matrice de confusion (lignes = vraie activité, colonnes = prédiction) :

| Vrai \ Prédit | dws | ups | wlk | jog | sit | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dws | 292 | 50 | 24 | 16 | 0 | 0 |
| ups | 67 | 366 | 45 | 1 | 3 | 0 |
| wlk | 48 | 18 | 1 141 | 0 | 0 | 0 |
| jog | 0 | 1 | 0 | 442 | 0 | 0 |
| sit | 1 | 0 | 0 | 0 | 1 125 | 0 |
| std | 0 | 9 | 4 | 0 | 0 | 897 |

Les positions assise et debout, ainsi que le jogging, sont très bien reconnus.
Le principal problème concerne les escaliers : 67 fenêtres de montée sont
prédites comme une descente, et 50 fenêtres de descente comme une montée. La
marche est parfois confondue avec ces deux activités. C'est logique, car les
trois mouvements ont un rythme de pas proche.

Les fichiers reports/classification_report.csv,
reports/confusion_matrix.csv et reports/evaluation.json contiennent les résultats
complets.

## Modèle TensorFlow Lite et simulation IoT

Le modèle est exporté dans models/activity_cnn_int8.tflite. Il est entièrement
quantifié en int8 : les poids, les activations, l'entrée et la sortie utilisent
des entiers sur 8 bits. La quantification est calibrée avec des fenêtres du jeu
d'entraînement seulement.

~~~~bash
.venv/bin/python src/export_tflite.py
~~~~

Le script de simulation lit un CSV une ligne à la fois, garde les 100 dernières
mesures en mémoire et lance une prédiction toutes les 50 nouvelles mesures. La
console affiche l'activité détectée et la confiance associée.

~~~~bash
# Exemple sur un enregistrement de jogging
.venv/bin/python src/simulate_iot.py

# Rejouer tout un fichier à la cadence de 50 Hz
.venv/bin/python src/simulate_iot.py \
  --input-file data/processed/clean_recordings/wlk_7/sub_2.csv \
  --max-predictions 0 --realtime
~~~~

Sur les dix premières fenêtres de jog_9/sub_2.csv, le simulateur a prédit jog
dix fois sur dix, avec 99,6 % de confiance.

## Taille du modèle et déploiement ESP32

~~~~bash
.venv/bin/python src/benchmark_tflite.py
~~~~

Le modèle TFLite fait 10 272 octets (10,03 KiB). Sur l'ordinateur utilisé pour
le projet, une inférence TFLite prend en moyenne 0,0122 ms (p95 : 0,0148 ms),
sur 1 000 mesures. Le détail est dans reports/edge_benchmark.json.

Ces chiffres sont utiles pour comparer des versions du modèle, mais ils ne sont
pas directement ceux d'un ESP32. La mesure a été faite sur un ordinateur macOS
arm64 avec TensorFlow 2.16.1. Sur un microcontrôleur, il faudra mesurer la taille
du tensor arena et le temps d'inférence avec TensorFlow Lite Micro.

Le modèle est suffisamment petit pour être envisagé sur un ESP32 : le fichier
fait environ 10 KiB et une fenêtre d'entrée int8 occupe 900 octets. Il faudra
toutefois conserver de la mémoire pour le programme, les buffers capteur et,
éventuellement, le Wi-Fi.

Si le modèle est encore trop coûteux sur la carte, on peut réduire le nombre de
filtres des convolutions, diminuer la taille de la couche dense, utiliser moins
de canaux capteur ou faire une prédiction moins souvent. Dans tous les cas, il
faudra vérifier que le F1-score reste acceptable, surtout pour les escaliers.

## Vidéo de présentation

Lien Loom ou YouTube non répertorié : **à ajouter avant l'envoi du rendu**.

La vidéo doit montrer le lancement de la simulation, le modèle TFLite généré et
expliquer les choix principaux : dataset, fenêtres temporelles, séparation par
participant, modèle, résultats et limites pour un ESP32.

Une trame détaillée pour préparer l'enregistrement est disponible dans
[`video.md`](video.md).

## Sources consultées

- [MotionSense](https://github.com/mmalekzadeh/motion-sense) : dataset et
  description des signaux utilisés.
- [Installation de TensorFlow](https://www.tensorflow.org/install/pip) :
  installation de l'environnement Python.
- [Quantification post-entraînement TensorFlow Lite](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) :
  export du modèle int8.
- [Documentation de tf.lite.Interpreter](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) :
  inférence TensorFlow Lite dans le simulateur.

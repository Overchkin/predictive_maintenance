# Predictive Maintenance (Industrial ML / IoT)

## Description

Ce projet implémente un pipeline complet de **maintenance prédictive industrielle** en combinant Machine Learning et IoT.
L'objectif est de **prévoir les défaillances des équipements industriels** à partir des données capteurs afin de réduire les temps d'arrêt et les coûts de maintenance.

Le projet est conçu pour démontrer les compétences d'un **Data Scientist / IA** ainsi que celles d'un **Développeur Full-Stack**, tout en constituant un portfolio professionnel complet.

---

## Fonctionnalités principales

* Pipeline ML complet : ingestion → prétraitement → entraînement → prédiction
* Classification binaire : `Machine failure` / `No failure`
* Feature engineering avancée : variables capteurs, agrégations temporelles, indicateurs personnalisés
* Visualisation interactive avec Plotly : histogrammes, boxplots, gauge de probabilité
* Interface Streamlit moderne : entrée utilisateur, prédiction en temps réel, feature importance, historique des prédictions
* Gestion de l'historique avec bouton `Réinitialiser`
* Structure professionnelle et modulable, compatible Docker pour déploiement

---

## Structure du projet

```
predictive_maintenance/
│
├── data/                 # Données brutes et préparées
│   ├── raw/              # Dataset original
│   └── processed/        # Données prétraitées pour l'entraînement
│
├── notebooks/            # Notebooks pour exploration et prototypage
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/                  # Code source
│   ├── data/
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── create_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── api/
│   │   └── main.py
│   ├── ui/
│   │   └── app.py        # Interface Streamlit
│   └── utils/
│       └── helpers.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_models.py
│
├── Dockerfile            # Conteneurisation
├── requirements.txt      # Dépendances Python
├── README.md             # Présentation du projet
└── .gitignore
```

---

## Installation

1. Cloner le projet :

```bash
git clone https://github.com/Overchkin/predictive_maintenance.git
cd predictive_maintenance
```

2. Créer un environnement Python et installer les dépendances :

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / Mac
pip install --upgrade pip
pip install -r requirements.txt
```

3. Préparer les données :

* Placer le dataset original dans `data/raw/`
* Le pipeline générera `data/processed/`

---

## Lancer le pipeline

```bash
python run_pipeline.py
```

* Teste l’ingestion, le preprocessing, l’entraînement et la prédiction.
* Exemple de sortie :

```json
{'prediction': 0, 'failure_probability': 0.0}
```

---

## Lancer l'interface Streamlit

```bash
streamlit run src/ui/app.py
```

* Entrées côté **sidebar**
* **Gauge dynamique** de probabilité de panne
* **Feature importance** interactive
* **Historique des prédictions** et **histogrammes des capteurs**
* Bouton **Réinitialiser l’historique**

---

## Fonctionnement du modèle

* Modèle : **Random Forest Classifier**
* Target : `Machine failure` (0 = pas de panne, 1 = panne)
* Features : capteurs industriels, températures, vitesse, torque, tool wear, type produit
* Prétraitement : normalisation et scaling des variables continues

---

## Contributions

* Démonstration complète d’un pipeline ML industriel
* UI moderne et interactive prête pour portfolio
* Modularité, testabilité et structure professionnelle
* Prêt pour déploiement Docker ou cloud

---

## Technologies utilisées

* Python 3.10+
* pandas, numpy
* scikit-learn
* plotly, streamlit
* Docker (optionnel)

---

## License

© 2026 Israël – Data Scientist / Full-Stack Developer

Ce projet est à usage personnel, portfolio et démonstration professionnelle. 

# Workflow - Predictive Maintenance (Industrial ML / IoT)

## 1. Vue d'ensemble

Le workflow de ce projet illustre le **parcours complet des données depuis leur acquisition jusqu'à la prédiction et la visualisation**. Il est conçu pour être modulable, testable et prêt pour une intégration front-end avec Streamlit.

## 2. Étapes principales

### Étape 1 : Ingestion des données

* Script : `src/data/ingest.py`
* Objectif :

  * Charger les datasets bruts depuis `data/raw/`
  * Vérifier la cohérence et l’intégrité des données
  * Sauvegarder une copie dans `data/processed/`

### Étape 2 : Prétraitement des données

* Script : `src/data/preprocess.py`
* Objectif :

  * Nettoyer les données (valeurs manquantes, outliers)
  * Normaliser et scaler les variables continues
  * Encoder les variables catégorielles (Type produit)
  * Sauvegarder les données prêtes pour le modèle

### Étape 3 : Feature Engineering

* Script : `src/features/create_features.py`
* Objectif :

  * Créer de nouvelles features à partir des variables capteurs
  * Agréger et transformer les indicateurs temporels
  * Générer des features supplémentaires pour améliorer la performance du modèle

### Étape 4 : Entraînement du modèle

* Script : `src/models/train.py`
* Objectif :

  * Charger les données prétraitées
  * Définir et entraîner un modèle de **Random Forest Classifier**
  * Évaluer la performance avec métriques standards (accuracy, f1-score, confusion matrix)
  * Sauvegarder le modèle entraîné et le scaler associé

### Étape 5 : Prédiction

* Script : `src/models/predict.py`
* Objectif :

  * Charger le modèle et le scaler sauvegardés
  * Recevoir de nouvelles données (entrée utilisateur ou batch)
  * Retourner la prédiction (`Machine failure` / `No failure`) et la probabilité de panne

### Étape 6 : Interface utilisateur (Streamlit)

* Script : `src/ui/app.py`
* Objectif :

  * Interface interactive pour entrer les valeurs capteurs
  * Bouton de prédiction pour calculer le risque de panne en temps réel
  * Afficher le **gauge de probabilité**, la **feature importance**, l’**historique** et les **visualisations des capteurs**
  * Bouton de réinitialisation pour nettoyer l’historique

### Étape 7 : Historique et Dashboard

* Historique des prédictions stocké dans `st.session_state.history`
* Visualisations interactives avec Plotly pour :

  * Histogrammes des valeurs capteurs
  * Boxplots des capteurs pour suivi des distributions

### Étape 8 : Tests unitaires

* Dossier : `tests/`
* Objectif :

  * Tester la robustesse des scripts de preprocessing et modèles
  * Vérifier que les prédictions sont cohérentes avec les données attendues

### Étape 9 : Déploiement (optionnel)

* Dockerfile pour créer un conteneur complet avec ML + UI
* Prêt pour déploiement cloud ou démonstration sur n’importe quelle machine

## 3. Pipeline résumé

```text
Raw Data -> Ingestion -> Preprocessing -> Feature Engineering -> Model Training -> Prediction -> Streamlit UI -> Dashboard / Visualisation
```

## 4. Notes importantes

* Les scripts sont **modulaires** et peuvent être exécutés individuellement ou via `run_pipeline.py`
* Le workflow est conçu pour être **répétable**, **testable** et **professionnel**, reflétant des standards de développement Data Science et Full-Stack.
* La structure du projet permet une **intégration facile avec Docker ou API REST** si nécessaire ultérieurement.


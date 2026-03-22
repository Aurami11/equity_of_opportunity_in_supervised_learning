# Fairness post-processing (Hardt, Price & Srebro, 2016)

Ce dépôt contient :
- un **script Python réutilisable** qui implémente le post-traitement de Hardt et al. (2016) via **programmation linéaire** ;
- un **notebook de démonstration** appliqué au dataset `ausprivauto0405` (package R **CASdatasets**), avec une comparaison de modèles de base vs. modèles post-traités.

Les deux contraintes de fairness supportées :
- **Equalized Odds (EO)** : mêmes **TPR** *et* **FPR** entre groupes (`Ŷ ⟂ A | Y`)
- **Equal Opportunity (EOpp)** : mêmes **TPR** entre groupes (conditionné sur `Y=1`)

---

## Structure du dépôt

```
.
├── fairness_postprocess_hardt.py          # Implémentation (Score + Binary post-processing) + CLI
├── fairness_CAS_ausprivauto0405_full.ipynb # Démo complète sur CASdatasets/ausprivauto0405
└── README.md
```

---

## Prérequis

- Python **3.9+** (recommandé : 3.10/3.11)
- Dépendances principales :
  - `numpy`, `pandas`
  - `scipy` (pour `scipy.optimize.linprog`)
- Dépendances pour le notebook :
  - `scikit-learn`, `matplotlib`
  - `pyreadr` (lecture du `.rda` extrait du tarball CASdatasets)
  - `jupyter` / `jupyterlab`

Installation rapide :

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib pyreadr jupyter
```

---

## Exécuter le notebook (démo `ausprivauto0405`)

1) Lancer Jupyter :

```bash
jupyter lab
# ou: jupyter notebook
```

2) Ouvrir `fairness_CAS_ausprivauto0405_full.ipynb` et exécuter les cellules dans l’ordre.

### Données : téléchargement automatique (internet requis)

Le notebook télécharge le tarball du package R **CASdatasets** et extrait `data/ausprivauto0405.rda`.
Le code pointe vers :

```text
https://dutangc.perso.math.cnrs.fr/RRepository/pub/src/contrib/CASdatasets_1.2-0.tar.gz
```

Si tu as déjà le fichier localement, tu peux éviter le téléchargement en réglant dans le notebook :

```python
local_tgz_path = "CASdatasets_1.2-0.tar.gz"
```

### Ce que fait le notebook (résumé)

- Définit :
  - `Y = ClaimOcc` (binaire 0/1 : sinistre ou non)
  - `A = Gender` (Female/Male)
  - `R` = score/proba d’un modèle
- Split en **train / calibration / test**.
- Entraîne des modèles de base (ex. LogReg, GradientBoosting) **sans utiliser `Gender` comme feature**.
- Fixe une politique de décision réaliste **top 10%** (seuil calculé sur la calibration).
- Apprend deux familles de post-traitement :
  1) **Score post-processing** : apprend une *distribution de seuils* par groupe (méthode la plus informative)
  2) **Binary post-processing** : apprend des *probabilités de flip* en fonction de `(A, Ŷ)`
- Compare les compromis perf / fairness, avec deux réglages de coûts (dataset très déséquilibré) :
  - `cost_fp=1, cost_fn=1` (erreur standard, peut donner des solutions triviales)
  - `cost_fp=1, cost_fn=10` (rater un sinistre coûte plus cher → décisions plus “actives”)

---

## Utiliser le script comme module Python

### Score post-processing

```python
import numpy as np
from fairness_postprocess_hardt import ScorePostProcessor, fairness_report

pp = ScorePostProcessor(method="equalized_odds", cost_fp=1.0, cost_fn=10.0, n_thresholds=200, random_state=0)
pp.fit(y_true=y_cal, scores=s_cal, a=A_cal)

p_test = pp.predict_proba(scores=s_test, a=A_test)   # proba de décision
y_test_fair = pp.predict(scores=s_test, a=A_test)    # décision binaire randomisée

print(fairness_report(y_true=y_test, y_pred=y_test_fair, a=A_test))
```

### Binary post-processing

```python
from fairness_postprocess_hardt import BinaryPostProcessor

bp = BinaryPostProcessor(method="equal_opportunity", cost_fp=1.0, cost_fn=10.0, random_state=0)
bp.fit(y_true=y_cal, y_pred=yhat_cal, a=A_cal)

p_test = bp.predict_proba(y_pred=yhat_test, a=A_test)
y_test_fair = bp.predict(y_pred=yhat_test, a=A_test)
```

---

## Points d’attention

- **Randomisation** : la méthode de Hardt et al. produit un prédicteur *randomisé* (garanties “en espérance” sur la distribution de données observée). Fixe `random_state` pour la reproductibilité.
- **Condition nécessaire** : chaque groupe doit avoir des **positifs et des négatifs** (`Y=1` et `Y=0`), sinon les contraintes EO/EOpp ne sont pas satisfaisables.
- **Dataset déséquilibré** : avec un `Y=1` rare, un réglage `cost_fn` plus élevé est souvent nécessaire pour éviter des solutions opérationnellement inutiles (ex : prédire toujours 0).

---

## Référence

Hardt, Price, Srebro (2016) — *Equality of Opportunity in Supervised Learning*.

---

## Auteur
Jules D'ALMEIDA
Frege Meli
Ben Soro

# Rapport de Validation du Pipeline sur 10 Échantillons BOAMP

- **Mode d'exécution IA :** Simulation calibrée (latence 450ms/requête)
- **Échantillons traités :** 10 annonces anonymisées

## 1. Métriques d'Exécution du Microservice

| Indicateur | Résultat Mesuré |
| :--- | :--- |
| **Temps de normalisation déterministe (10 annonces)** | `19.30 ms` |
| **Temps total du traitement Batch (filtrage + IA bornée)** | `488.83 ms` |
| **Annonces éligibles après pré-filtrage** | `2 / 10` |
| **Appels IA exécutés (plafond quota respecté)** | `2` (`0` différés, `8` filtrés) |
| **Réduction des appels LLM grâce au pré-filtrage** | **`80%`** |

## 2. Détail des Résultats par Échantillon

| ID Annonce | Département | Éligible | Statut Batch | Score IA | Action Recommandée |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `26-109796` | 45 | Non | `skipped` | - | Département '45' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109797` | 86 | Non | `skipped` | - | Département '86' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109701` | 95 | Non | `skipped` | - | Département '95' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109739` | 67 | Non | `skipped` | - | Département '67' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109757` | 94 | Oui | `analyzed` | 8/10 | À étudier |
| `26-109780` | 54 | Non | `skipped` | - | Département '54' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109821` | 49 | Non | `skipped` | - | Département '49' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109907` | 91 | Oui | `analyzed` | 8/10 | À étudier |
| `26-109945` | 62 | Non | `skipped` | - | Département '62' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |
| `26-109669` | 974 | Non | `skipped` | - | Département '974' hors zone cible (['75', '91', '92', '93', '94', '69', '33', '31']) |

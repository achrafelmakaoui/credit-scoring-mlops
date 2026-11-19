# Règles de contribution

## Branches

- `main` est protégée : aucun push direct, tout passe par une pull request.
- Une branche par fonctionnalité, créée depuis `main` à jour :
  - `setup/...` pour la configuration du projet
  - `feature/...` pour une nouvelle fonctionnalité
  - `fix/...` pour une correction
  - `docs/...` pour la documentation

```bash
git checkout main
git pull origin main
git checkout -b feature/nom-court
```

## Commits

- Un commit = un changement logique (commits atomiques).
- Message à l'impératif, en une phrase, commençant par une majuscule, sans préfixe de type.
  - Correct : `Add stratified train/test split`
  - Correct : `Fix input validation for loan amount`
  - À éviter : `feat: add split`, `update`, `wip`, `fix stuff`
- Jamais de données, de modèles (`.pkl`) ni de secrets (`.env`) dans un commit.

## Pull requests

1. Pousser la branche : `git push -u origin feature/nom-court`.
2. Ouvrir la PR vers `main` et remplir le modèle de description.
3. **L'autre membre du binôme relit et approuve** (au moins un commentaire utile).
4. La CI doit être verte avant la fusion.
5. Fusion par *Squash and merge* ou *Rebase and merge*, puis suppression de la branche.

## Avant de pousser

```bash
ruff check .
ruff format .
pytest
```

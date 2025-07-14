# Git Flow Strategy

## Branches:
- `main`: Stable production-ready code
- `develop`: Integration branch for features
- `feature/*`: Short-lived branches for new features (e.g., `feature/train-model`)
- `release/*`: Used to prepare release from `develop` to `main`
- `hotfix/*`: Urgent fixes applied directly to `main`

## Rules:
- Always use Pull Requests (PRs)
- Feature branches → merge to `develop` after review
- Releases are merged to `main` after testing

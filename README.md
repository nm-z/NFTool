# NFTool

Deep learning tool for modular regression analysis and training.

Quick start:
`docker compose up`

Visit: `http://localhost:3000` (frontend) and `http://localhost:8001` (backend)

Container-first workflow (recommended):
- `docker compose exec frontend npm run lint`
- `docker compose exec frontend npm run build`
- `docker compose exec backend python -m pytest`
- `docker compose exec backend python -m pylint src/`

The services run in Docker by default; the repo is bind-mounted into the containers so code changes apply immediately.

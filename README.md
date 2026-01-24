# NFTool

Deep learning tool for modular regression analysis and training.

Quick start:
`npm run tauri:dev`

Visit: `http://localhost:3000` (frontend) and `http://localhost:8001` (backend)

Local workflow:
- `npm --prefix frontend run lint`
- `npm --prefix frontend run build`
- `python -m pytest backend/tests`
- `python -m pylint backend/src`

The Tauri dev server starts both the frontend and backend sidecar for local development.

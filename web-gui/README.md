# LLM-Swarm-Brain Web GUI

Interactive React-based control panel for managing LLM-Swarm-Brain deployment and inference.

## Features

- 🖥️ Real-time server status monitoring
- 🎮 GPU monitoring with live stats
- 🚀 One-click deployment
- 🧠 Inference control with multiple modes
- 📊 Results visualization
- 📝 Live console output

## Setup

```bash
cd web-gui
npm install
npm start
```

The app will open at `http://localhost:3000`

## Backend API Required

You need to implement a backend API server at `http://localhost:3001` with these endpoints:

- `POST /api/test-connection` - Test SSH connection
- `POST /api/check-gpu` - Check GPU status
- `POST /api/deploy` - Deploy to server
- `POST /api/run-inference` - Run inference
- `POST /api/get-results` - Get results
- `POST /api/download-results` - Download results file

See `server-example.js` for implementation reference.

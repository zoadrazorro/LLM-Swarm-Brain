# LLM-Swarm-Brain API Server

Backend API for SSH communication with the remote server.

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```
   SSH_HOST=147.185.41.15
   SSH_USER=ubuntu
   SSH_PASSWORD=your_password
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the server:
   ```bash
   npm start
   ```

## Security Notes

- **Never commit `.env` file to git** (it's in `.gitignore`)
- The `.env` file contains sensitive credentials
- Password authentication is less secure than SSH keys
- Consider using SSH key authentication for production

## Authentication Methods

### Password Authentication (Current)
```env
SSH_PASSWORD=your_password
```

### SSH Key Authentication (More Secure)
```env
# Comment out SSH_PASSWORD and use:
SSH_KEY_PATH=C:\Users\jelly\SSH_01
```

## API Endpoints

- `POST /api/test-connection` - Test SSH connection
- `POST /api/check-gpu` - Check GPU status
- `POST /api/deploy` - Deploy to server
- `POST /api/run-inference` - Run inference
- `POST /api/get-results` - Get results
- `POST /api/download-results` - Download results

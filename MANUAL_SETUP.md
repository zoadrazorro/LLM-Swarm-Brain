# Manual Server Setup Required

The server needs the `python3-venv` package installed. This requires sudo access and must be done manually once.

## One-Time Setup Command

Run this command **once** via SSH:

```bash
ssh -i C:\Users\jelly\SSH_01 ubuntu@147.185.41.15 "sudo apt-get update && sudo apt-get install -y python3.10-venv python3-pip"
```

Or connect interactively and run:

```bash
ssh -i C:\Users\jelly\SSH_01 ubuntu@147.185.41.15

# Then on the server:
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip
exit
```

## After Installing

Once the package is installed, you can use the GUI normally:
1. Click "Deploy to Server" 
2. Or click "Run Inference" (will auto-deploy)

The GUI will handle everything else automatically.

## Why This is Needed

- Ubuntu servers don't include `python3-venv` by default
- Installing system packages requires sudo (admin) access
- The GUI can't run sudo commands (security restriction)
- This only needs to be done **once** per server

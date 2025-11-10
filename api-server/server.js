require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { NodeSSH } = require('node-ssh');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// SSH helper function
async function executeSSH(config, command, streaming = false) {
  const ssh = new NodeSSH();
  
  try {
    // Use environment variables if available, otherwise use config from request
    const sshConfig = {
      host: process.env.SSH_HOST || config.host,
      username: process.env.SSH_USER || config.user,
      tryKeyboard: true,  // Enable keyboard-interactive authentication
    };

    // Use both SSH key and password if both are provided
    if (process.env.SSH_KEY_PATH || config.sshKey) {
      sshConfig.privateKeyPath = process.env.SSH_KEY_PATH || config.sshKey;
    }
    
    if (process.env.SSH_PASSWORD) {
      sshConfig.password = process.env.SSH_PASSWORD;
      sshConfig.passphrase = process.env.SSH_PASSWORD;  // In case key has passphrase
      // Also try keyboard-interactive for password
      sshConfig.onKeyboardInteractive = (name, instructions, instructionsLang, prompts, finish) => {
        if (prompts.length > 0 && prompts[0].prompt.toLowerCase().includes('password')) {
          finish([process.env.SSH_PASSWORD]);
        }
      };
    }

    const authMethod = [];
    if (sshConfig.privateKeyPath) authMethod.push('SSH Key');
    if (sshConfig.password) authMethod.push('Password');
    console.log(`Connecting to ${sshConfig.host} as ${sshConfig.username} using ${authMethod.join(' + ')}...`);
    await ssh.connect(sshConfig);

    if (streaming) {
      return ssh.execCommand(command, { cwd: '/home/ubuntu' });
    } else {
      const result = await ssh.execCommand(command, { cwd: '/home/ubuntu' });
      ssh.dispose();
      return result;
    }
  } catch (error) {
    throw new Error(`SSH Error: ${error.message}`);
  }
}

// Test connection
app.post('/api/test-connection', async (req, res) => {
  try {
    const result = await executeSSH(req.body, 'echo "Connection successful"');
    res.json({ success: true, output: result.stdout });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

// Check GPU
app.post('/api/check-gpu', async (req, res) => {
  try {
    const result = await executeSSH(
      req.body,
      'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits'
    );

    const gpus = result.stdout.split('\n').filter(l => l.trim()).map(line => {
      const [index, name, total, used, free, temp, util] = line.split(', ');
      return {
        index: parseInt(index),
        name,
        memory_total: `${total}MB`,
        memory_used: `${used}MB`,
        memory_free: `${free}MB`,
        memory_used_percent: ((parseInt(used) / parseInt(total)) * 100).toFixed(1),
        temperature: parseInt(temp),
        utilization: parseInt(util)
      };
    });

    const cudaResult = await executeSSH(req.body, 'nvcc --version 2>&1 | grep "release" || echo "N/A"');
    const pytorchResult = await executeSSH(
      req.body,
      'cd LLM-Swarm-Brain && source venv/bin/activate && python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false"'
    );

    res.json({
      success: true,
      gpuInfo: {
        gpus,
        cuda_version: cudaResult.stdout.trim(),
        pytorch_cuda: pytorchResult.stdout.trim() === 'True'
      }
    });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

// Deploy
app.post('/api/deploy', async (req, res) => {
  res.setHeader('Content-Type', 'text/plain');
  res.setHeader('Transfer-Encoding', 'chunked');

  try {
    // Clone/update repo
    res.write('Cloning/updating repository...\n');
    const cloneResult = await executeSSH(
      req.body,
      'if [ -d "LLM-Swarm-Brain" ]; then cd LLM-Swarm-Brain && git pull; else git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git; fi'
    );
    res.write(cloneResult.stdout + '\n');

    // Setup Python environment
    res.write('\nSetting up Python environment...\n');
    const setupResult = await executeSSH(
      req.body,
      'cd LLM-Swarm-Brain && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt && pip install -e .'
    );
    res.write(setupResult.stdout + '\n');
    if (setupResult.stderr) res.write(setupResult.stderr + '\n');
    
    res.write('\n✓ Deployment complete!\n');
    res.end();
  } catch (error) {
    res.write(`Error: ${error.message}\n`);
    res.end();
  }
});

// Run inference
app.post('/api/run-inference', async (req, res) => {
  res.setHeader('Content-Type', 'text/plain');
  res.setHeader('Transfer-Encoding', 'chunked');

  try {
    // Check if venv exists and is valid
    const checkResult = await executeSSH(req.body, 'test -f LLM-Swarm-Brain/venv/bin/activate && echo "exists" || echo "missing"');
    
    if (checkResult.stdout.trim() === 'missing') {
      res.write('⚠️  Virtual environment not found. Setting up...\n\n');
      
      // Ensure repo exists
      const cloneResult = await executeSSH(
        req.body,
        'if [ ! -d "LLM-Swarm-Brain" ]; then git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git; fi'
      );
      if (cloneResult.stdout) res.write(cloneResult.stdout + '\n');
      
      // Create venv and install packages
      res.write('Creating virtual environment...\n');
      const venvResult = await executeSSH(req.body, 'cd LLM-Swarm-Brain && python3 -m venv venv');
      if (venvResult.stderr) res.write(venvResult.stderr + '\n');
      
      res.write('Installing PyTorch with CUDA support...\n');
      const torchResult = await executeSSH(
        req.body,
        'cd LLM-Swarm-Brain && source venv/bin/activate && pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
      );
      res.write(torchResult.stdout + '\n');
      
      res.write('Installing other packages...\n');
      const installResult = await executeSSH(
        req.body,
        'cd LLM-Swarm-Brain && source venv/bin/activate && pip install -r requirements.txt && pip install -e .'
      );
      res.write(installResult.stdout + '\n');
      if (installResult.stderr && !installResult.stderr.includes('WARNING')) {
        res.write(installResult.stderr + '\n');
      }
      res.write('✓ Setup complete\n\n');
    } else {
      res.write('✓ Environment ready\n\n');
    }
    
    // Run inference
    const modeFlag = req.body.mode === 'quick' ? '--quick' :
                     req.body.mode === 'models' ? '--load-models' : '';
    
    res.write(`Running inference test (${req.body.mode} mode)...\n\n`);
    const command = `cd LLM-Swarm-Brain && source venv/bin/activate && python inference_test.py ${modeFlag}`;
    const result = await executeSSH(req.body, command);
    
    res.write(result.stdout + '\n');
    if (result.stderr) res.write(result.stderr + '\n');
    res.end();
  } catch (error) {
    res.write(`Error: ${error.message}\n`);
    res.end();
  }
});

// Get results
app.post('/api/get-results', async (req, res) => {
  try {
    const result = await executeSSH(
      req.body,
      'cd LLM-Swarm-Brain && cat $(ls -t inference_results_*.json | head -1) 2>/dev/null || echo "{}"'
    );

    const results = JSON.parse(result.stdout || '{}');
    res.json({ success: true, results });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

// Download results
app.post('/api/download-results', async (req, res) => {
  try {
    const result = await executeSSH(
      req.body,
      'cd LLM-Swarm-Brain && cat $(ls -t inference_results_*.json | head -1)'
    );

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=inference_results.json');
    res.send(result.stdout);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Tail logs (streaming)
app.post('/api/tail-logs', async (req, res) => {
  res.setHeader('Content-Type', 'text/plain');
  res.setHeader('Transfer-Encoding', 'chunked');

  try {
    const logFile = req.body.logFile || 'inference.log';
    const lines = req.body.lines || 50;
    
    // Get recent log lines
    const result = await executeSSH(
      req.body,
      `cd LLM-Swarm-Brain && if [ -f ${logFile} ]; then tail -n ${lines} ${logFile}; else echo "Log file not found"; fi`
    );
    
    res.write(result.stdout || 'No logs available\n');
    if (result.stderr) res.write(result.stderr);
    res.end();
  } catch (error) {
    res.write(`Error: ${error.message}\n`);
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`🚀 API Server running on http://localhost:${PORT}`);
  console.log(`📡 SSH Host: ${process.env.SSH_HOST || 'Not configured'}`);
  console.log(`👤 SSH User: ${process.env.SSH_USER || 'Not configured'}`);
  
  const authMethods = [];
  if (process.env.SSH_KEY_PATH) authMethods.push('SSH Key');
  if (process.env.SSH_PASSWORD) authMethods.push('Password');
  console.log(`🔐 Auth Method: ${authMethods.length > 0 ? authMethods.join(' + ') : 'From client config'}`);
});

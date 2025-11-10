import React, { useState, useEffect } from 'react';
import { 
  Server, 
  Cpu, 
  HardDrive, 
  Zap, 
  PlayCircle, 
  Download, 
  Terminal,
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Settings
} from 'lucide-react';
import ServerInfo from './components/ServerInfo';
import InferenceControl from './components/InferenceControl';
import ConsoleOutput from './components/ConsoleOutput';
import ResultsViewer from './components/ResultsViewer';
import GPUMonitor from './components/GPUMonitor';
import LogViewer from './components/LogViewer';

function App() {
  const [serverStatus, setServerStatus] = useState('disconnected');
  const [gpuInfo, setGpuInfo] = useState(null);
  const [consoleOutput, setConsoleOutput] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);

  const serverConfig = {
    host: '147.185.41.15',
    user: 'ubuntu',
    sshKey: 'C:\\Users\\jelly\\SSH_01',
    specs: {
      gpu: '8 x H100 SXM5 (80GB)',
      cpu: '104 x Intel Xeon Platinum 8470',
      ram: '1TB',
      storage: '17.9TB',
      network: 'Ethernet'
    }
  };

  const addConsoleOutput = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setConsoleOutput(prev => [...prev, { timestamp, message, type }]);
  };

  const testConnection = async () => {
    addConsoleOutput('Testing SSH connection...', 'info');
    setServerStatus('connecting');
    
    try {
      const response = await fetch('/api/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });
      
      const data = await response.json();
      
      if (data.success) {
        setServerStatus('connected');
        addConsoleOutput('✓ Connected successfully', 'success');
      } else {
        setServerStatus('error');
        addConsoleOutput('✗ Connection failed: ' + data.error, 'error');
      }
    } catch (error) {
      setServerStatus('error');
      addConsoleOutput('✗ Connection error: ' + error.message, 'error');
    }
  };

  const checkGPU = async () => {
    addConsoleOutput('Checking GPU status...', 'info');
    
    try {
      const response = await fetch('/api/check-gpu', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });
      
      const data = await response.json();
      
      if (data.success) {
        setGpuInfo(data.gpuInfo);
        addConsoleOutput('✓ GPU check complete', 'success');
      } else {
        addConsoleOutput('✗ GPU check failed: ' + data.error, 'error');
      }
    } catch (error) {
      addConsoleOutput('✗ GPU check error: ' + error.message, 'error');
    }
  };

  const deployToServer = async () => {
    addConsoleOutput('Starting deployment...', 'info');
    setIsRunning(true);
    
    try {
      const response = await fetch('/api/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n').filter(l => l.trim());
        
        lines.forEach(line => {
          addConsoleOutput(line, 'info');
        });
      }
      
      addConsoleOutput('✓ Deployment complete', 'success');
    } catch (error) {
      addConsoleOutput('✗ Deployment error: ' + error.message, 'error');
    } finally {
      setIsRunning(false);
    }
  };

  const runInference = async (mode) => {
    addConsoleOutput(`Starting inference test (${mode} mode)...`, 'info');
    setIsRunning(true);
    
    try {
      const response = await fetch('/api/run-inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...serverConfig, mode })
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n').filter(l => l.trim());
        
        lines.forEach(line => {
          addConsoleOutput(line, 'info');
        });
      }
      
      addConsoleOutput('✓ Inference complete', 'success');
      
      // Fetch results
      const resultsResponse = await fetch('/api/get-results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });
      
      const resultsData = await resultsResponse.json();
      if (resultsData.success) {
        setResults(resultsData.results);
      }
    } catch (error) {
      addConsoleOutput('✗ Inference error: ' + error.message, 'error');
    } finally {
      setIsRunning(false);
    }
  };

  const downloadResults = async () => {
    addConsoleOutput('Downloading results...', 'info');
    
    try {
      const response = await fetch('/api/download-results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `inference_results_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      addConsoleOutput('✓ Results downloaded', 'success');
    } catch (error) {
      addConsoleOutput('✗ Download error: ' + error.message, 'error');
    }
  };

  useEffect(() => {
    addConsoleOutput('LLM-Swarm-Brain GUI initialized', 'info');
  }, []);

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="glass-effect rounded-2xl p-6 mb-6 shadow-2xl">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                🧠 LLM-Swarm-Brain Control Panel
              </h1>
              <p className="text-gray-200">
                Neural Network of 64 Phi-3 Neurons across 8× H100 GPUs
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className={`px-4 py-2 rounded-lg font-semibold flex items-center gap-2 ${
                serverStatus === 'connected' ? 'bg-green-500' :
                serverStatus === 'connecting' ? 'bg-yellow-500' :
                serverStatus === 'error' ? 'bg-red-500' : 'bg-gray-500'
              }`}>
                {serverStatus === 'connected' && <CheckCircle size={20} />}
                {serverStatus === 'connecting' && <RefreshCw size={20} className="animate-spin" />}
                {serverStatus === 'error' && <XCircle size={20} />}
                {serverStatus === 'disconnected' && <AlertCircle size={20} />}
                <span className="text-white capitalize">{serverStatus}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Server Info */}
        <ServerInfo 
          config={serverConfig} 
          status={serverStatus}
          onTestConnection={testConnection}
        />

        {/* GPU Monitor */}
        <GPUMonitor 
          gpuInfo={gpuInfo}
          onCheckGPU={checkGPU}
          isRunning={isRunning}
        />

        {/* Main Control Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Inference Control */}
          <InferenceControl
            onDeploy={deployToServer}
            onRunInference={runInference}
            onDownloadResults={downloadResults}
            isRunning={isRunning}
            serverStatus={serverStatus}
          />

          {/* Console Output */}
          <ConsoleOutput 
            output={consoleOutput}
            onClear={() => setConsoleOutput([])}
          />
        </div>

        {/* Remote Log Viewer */}
        <div className="mb-6">
          <LogViewer 
            serverConfig={serverConfig}
            isRunning={isRunning}
          />
        </div>

        {/* Results Viewer */}
        {results && (
          <ResultsViewer results={results} />
        )}
      </div>
    </div>
  );
}

export default App;

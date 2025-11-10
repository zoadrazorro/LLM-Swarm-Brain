import React, { useState } from 'react';
import { PlayCircle, Download, Upload, Zap } from 'lucide-react';

function InferenceControl({ onDeploy, onRunInference, onDownloadResults, isRunning, serverStatus }) {
  const [selectedMode, setSelectedMode] = useState('quick');

  const modes = [
    { 
      value: 'quick', 
      label: 'Quick Test', 
      description: '1 question per level, no models',
      icon: Zap,
      color: 'bg-green-500'
    },
    { 
      value: 'full', 
      label: 'Full Test', 
      description: 'All questions, no models',
      icon: PlayCircle,
      color: 'bg-blue-500'
    },
    { 
      value: 'models', 
      label: 'With Models', 
      description: 'Full test with GPU models',
      icon: Zap,
      color: 'bg-purple-500'
    },
  ];

  const isDisabled = isRunning || serverStatus !== 'connected';

  return (
    <div className="glass-effect rounded-2xl p-6 shadow-2xl">
      <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
        <PlayCircle className="text-green-400" />
        Inference Control
      </h2>

      {/* Deploy Button */}
      <div className="mb-6">
        <button
          onClick={onDeploy}
          disabled={isDisabled}
          className="w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-gray-500 disabled:to-gray-600 text-white rounded-lg font-semibold transition-all transform hover:scale-105 disabled:scale-100 flex items-center justify-center gap-2 shadow-lg"
        >
          <Upload size={20} />
          Deploy to Server
        </button>
        <p className="text-gray-300 text-xs mt-2 text-center">
          Clone repository and run setup script
        </p>
      </div>

      {/* Mode Selection */}
      <div className="mb-6">
        <label className="text-white font-semibold mb-3 block">
          Inference Mode
        </label>
        <div className="space-y-2">
          {modes.map((mode) => {
            const Icon = mode.icon;
            return (
              <button
                key={mode.value}
                onClick={() => setSelectedMode(mode.value)}
                className={`w-full p-4 rounded-lg transition-all ${
                  selectedMode === mode.value
                    ? 'bg-white bg-opacity-20 border-2 border-white'
                    : 'bg-white bg-opacity-5 border-2 border-transparent hover:bg-opacity-10'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${mode.color}`}>
                    <Icon size={20} className="text-white" />
                  </div>
                  <div className="flex-1 text-left">
                    <div className="text-white font-semibold">{mode.label}</div>
                    <div className="text-gray-300 text-xs">{mode.description}</div>
                  </div>
                  {selectedMode === mode.value && (
                    <div className="w-4 h-4 bg-white rounded-full" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Run Inference Button */}
      <div className="mb-4">
        <button
          onClick={() => onRunInference(selectedMode)}
          disabled={isDisabled}
          className="w-full px-6 py-4 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-500 disabled:to-gray-600 text-white rounded-lg font-semibold transition-all transform hover:scale-105 disabled:scale-100 flex items-center justify-center gap-2 shadow-lg"
        >
          <PlayCircle size={20} className={isRunning ? 'animate-pulse' : ''} />
          {isRunning ? 'Running...' : 'Run Inference'}
        </button>
      </div>

      {/* Download Results Button */}
      <button
        onClick={onDownloadResults}
        disabled={isDisabled}
        className="w-full px-6 py-3 bg-white bg-opacity-10 hover:bg-opacity-20 disabled:bg-opacity-5 text-white rounded-lg font-semibold transition-all flex items-center justify-center gap-2"
      >
        <Download size={18} />
        Download Results
      </button>

      {/* Status Message */}
      {serverStatus !== 'connected' && (
        <div className="mt-4 p-3 bg-yellow-500 bg-opacity-20 border border-yellow-500 rounded-lg">
          <p className="text-yellow-200 text-sm text-center">
            ⚠️ Connect to server first to enable controls
          </p>
        </div>
      )}
    </div>
  );
}

export default InferenceControl;

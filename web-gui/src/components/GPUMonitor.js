import React from 'react';
import { Activity, Zap, Thermometer, RefreshCw } from 'lucide-react';

function GPUMonitor({ gpuInfo, onCheckGPU, isRunning }) {
  return (
    <div className="glass-effect rounded-2xl p-6 mb-6 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Activity className="text-yellow-400" />
          GPU Monitor
        </h2>
        <button
          onClick={onCheckGPU}
          disabled={isRunning}
          className="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-500 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
        >
          <RefreshCw size={16} className={isRunning ? 'animate-spin' : ''} />
          Check GPUs
        </button>
      </div>

      {gpuInfo ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {gpuInfo.gpus && gpuInfo.gpus.map((gpu, index) => (
            <div key={index} className="bg-white bg-opacity-10 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-semibold">GPU {gpu.index}</span>
                <Zap size={16} className="text-yellow-400" />
              </div>
              <p className="text-gray-300 text-sm mb-2">{gpu.name}</p>
              
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Memory</span>
                    <span>{gpu.memory_used} / {gpu.memory_total}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: `${(gpu.memory_used_percent || 0)}%` }}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400 flex items-center gap-1">
                    <Thermometer size={12} />
                    Temp
                  </span>
                  <span className="text-white font-semibold">{gpu.temperature}°C</span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Utilization</span>
                  <span className="text-white font-semibold">{gpu.utilization}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <Activity size={48} className="text-gray-400 mx-auto mb-3 animate-pulse-slow" />
          <p className="text-gray-300">Click "Check GPUs" to view GPU status</p>
        </div>
      )}

      {gpuInfo && gpuInfo.cuda_version && (
        <div className="mt-4 pt-4 border-t border-white border-opacity-20">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-300">CUDA Version:</span>
            <span className="text-white font-semibold">{gpuInfo.cuda_version}</span>
          </div>
          <div className="flex items-center justify-between text-sm mt-2">
            <span className="text-gray-300">PyTorch CUDA:</span>
            <span className="text-white font-semibold">
              {gpuInfo.pytorch_cuda ? '✓ Available' : '✗ Not Available'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default GPUMonitor;

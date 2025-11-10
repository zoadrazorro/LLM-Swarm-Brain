import React from 'react';
import { Server, Cpu, HardDrive, Zap, Network } from 'lucide-react';

function ServerInfo({ config, status, onTestConnection }) {
  return (
    <div className="glass-effect rounded-2xl p-6 mb-6 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Server className="text-blue-400" />
          Server Information
        </h2>
        <button
          onClick={onTestConnection}
          className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition-colors"
        >
          Test Connection
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Host Info */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Server size={20} className="text-blue-400" />
            <span className="text-gray-300 font-semibold">Host</span>
          </div>
          <p className="text-white font-mono text-sm">
            {config.user}@{config.host}
          </p>
          <p className="text-gray-400 text-xs mt-1">portly-lavender-cheetah</p>
        </div>

        {/* GPU */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Zap size={20} className="text-yellow-400" />
            <span className="text-gray-300 font-semibold">GPU</span>
          </div>
          <p className="text-white font-semibold">{config.specs.gpu}</p>
          <p className="text-gray-400 text-xs mt-1">640GB Total VRAM</p>
        </div>

        {/* CPU */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={20} className="text-purple-400" />
            <span className="text-gray-300 font-semibold">CPU</span>
          </div>
          <p className="text-white font-semibold">{config.specs.cpu}</p>
          <p className="text-gray-400 text-xs mt-1">High-performance compute</p>
        </div>

        {/* RAM */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive size={20} className="text-green-400" />
            <span className="text-gray-300 font-semibold">RAM</span>
          </div>
          <p className="text-white font-semibold">{config.specs.ram}</p>
          <p className="text-gray-400 text-xs mt-1">System Memory</p>
        </div>

        {/* Storage */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive size={20} className="text-cyan-400" />
            <span className="text-gray-300 font-semibold">Storage</span>
          </div>
          <p className="text-white font-semibold">{config.specs.storage}</p>
          <p className="text-gray-400 text-xs mt-1">Disk Space</p>
        </div>

        {/* Network */}
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Network size={20} className="text-pink-400" />
            <span className="text-gray-300 font-semibold">Network</span>
          </div>
          <p className="text-white font-semibold">{config.specs.network}</p>
          <p className="text-gray-400 text-xs mt-1">us-central-1</p>
        </div>
      </div>
    </div>
  );
}

export default ServerInfo;

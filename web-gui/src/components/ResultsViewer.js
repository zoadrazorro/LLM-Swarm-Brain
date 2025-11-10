import React from 'react';
import { BarChart, Brain, TrendingUp } from 'lucide-react';

function ResultsViewer({ results }) {
  return (
    <div className="glass-effect rounded-2xl p-6 shadow-2xl">
      <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
        <BarChart className="text-purple-400" />
        Inference Results
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Brain size={20} className="text-blue-400" />
            <span className="text-gray-300">Consciousness Level</span>
          </div>
          <p className="text-3xl font-bold text-white">
            {(results.avg_consciousness || 0).toFixed(3)}
          </p>
        </div>

        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={20} className="text-green-400" />
            <span className="text-gray-300">Neurons Fired</span>
          </div>
          <p className="text-3xl font-bold text-white">
            {Math.round(results.avg_neurons || 0)}
          </p>
        </div>

        <div className="bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <BarChart size={20} className="text-purple-400" />
            <span className="text-gray-300">Integration Φ</span>
          </div>
          <p className="text-3xl font-bold text-white">
            {(results.avg_phi || 0).toFixed(3)}
          </p>
        </div>
      </div>

      <div className="bg-white bg-opacity-5 rounded-lg p-4">
        <h3 className="text-white font-semibold mb-2">Test Summary</h3>
        <p className="text-gray-300 text-sm">
          Total Questions: {results.total_questions || 0} | 
          Successful: {results.successful || 0} | 
          Failed: {results.failed || 0}
        </p>
      </div>
    </div>
  );
}

export default ResultsViewer;

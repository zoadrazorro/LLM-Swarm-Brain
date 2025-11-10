import React, { useEffect, useRef } from 'react';
import { Terminal, Trash2 } from 'lucide-react';

function ConsoleOutput({ output, onClear }) {
  const consoleRef = useRef(null);

  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [output]);

  const getTypeColor = (type) => {
    switch (type) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      default: return 'text-gray-300';
    }
  };

  return (
    <div className="glass-effect rounded-2xl p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Terminal className="text-cyan-400" />
          Console Output
        </h2>
        <button
          onClick={onClear}
          className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <Trash2 size={16} />
          Clear
        </button>
      </div>

      <div
        ref={consoleRef}
        className="bg-black bg-opacity-50 rounded-lg p-4 h-96 overflow-y-auto console-output"
      >
        {output.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No output yet...</p>
        ) : (
          output.map((line, index) => (
            <div key={index} className="mb-1">
              <span className="text-gray-500">[{line.timestamp}]</span>{' '}
              <span className={getTypeColor(line.type)}>{line.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ConsoleOutput;

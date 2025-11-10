import React, { useState, useEffect, useRef } from 'react';
import { FileText, RefreshCw, Download, Trash2 } from 'lucide-react';

function LogViewer({ serverConfig, isRunning }) {
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const logContainerRef = useRef(null);
  const intervalRef = useRef(null);

  const fetchLogs = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/tail-logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...serverConfig,
          logFile: 'inference.log',
          lines: 100
        })
      });

      const text = await response.text();
      const lines = text.split('\n').filter(l => l.trim());
      setLogs(lines);

      // Auto-scroll to bottom
      if (logContainerRef.current) {
        logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
      }
    } catch (error) {
      console.error('Error fetching logs:', error);
      setLogs([`Error: ${error.message}`]);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadLogs = () => {
    const logText = logs.join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inference_logs_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const clearLogs = () => {
    setLogs([]);
  };

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh || isRunning) {
      fetchLogs(); // Initial fetch
      intervalRef.current = setInterval(fetchLogs, refreshInterval);
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [autoRefresh, isRunning, refreshInterval]);

  // Parse log lines to extract level and format
  const parseLogLine = (line) => {
    // Match format: YYYY-MM-DD HH:MM:SS,mmm - module - LEVEL - message
    const match = line.match(/^(.+?) - (.+?) - (\w+) - (.+)$/);
    if (match) {
      return {
        timestamp: match[1],
        module: match[2],
        level: match[3],
        message: match[4]
      };
    }
    return { raw: line };
  };

  const getLogLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR': return 'text-red-400';
      case 'WARNING': return 'text-yellow-400';
      case 'INFO': return 'text-cyan-400';
      case 'DEBUG': return 'text-gray-400';
      default: return 'text-gray-300';
    }
  };

  return (
    <div className="glass-effect rounded-2xl p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <FileText className="text-cyan-400" />
          Remote Logs
        </h2>
        
        <div className="flex items-center gap-2">
          {/* Auto-refresh toggle */}
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4"
            />
            Auto-refresh
          </label>

          {/* Refresh interval */}
          {autoRefresh && (
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="px-2 py-1 bg-gray-700 text-white rounded text-sm"
            >
              <option value={2000}>2s</option>
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
            </select>
          )}

          {/* Manual refresh */}
          <button
            onClick={fetchLogs}
            disabled={isLoading}
            className="px-3 py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
            Refresh
          </button>

          {/* Download */}
          <button
            onClick={downloadLogs}
            disabled={logs.length === 0}
            className="px-3 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Download size={16} />
          </button>

          {/* Clear */}
          <button
            onClick={clearLogs}
            disabled={logs.length === 0}
            className="px-3 py-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div
        ref={logContainerRef}
        className="bg-black bg-opacity-50 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm"
      >
        {logs.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No logs available. Click refresh to fetch logs from remote server.
          </p>
        ) : (
          logs.map((line, index) => {
            const parsed = parseLogLine(line);
            
            if (parsed.raw) {
              // Unparsed line - display as-is
              return (
                <div key={index} className="mb-1 text-gray-300">
                  {line}
                </div>
              );
            }

            // Parsed log line
            return (
              <div key={index} className="mb-1 flex gap-2">
                <span className="text-gray-500 text-xs">{parsed.timestamp}</span>
                <span className="text-purple-400 text-xs">[{parsed.module}]</span>
                <span className={`font-semibold text-xs ${getLogLevelColor(parsed.level)}`}>
                  {parsed.level}
                </span>
                <span className="text-gray-200 flex-1">{parsed.message}</span>
              </div>
            );
          })
        )}
      </div>

      {isRunning && (
        <div className="mt-2 flex items-center gap-2 text-sm text-cyan-400">
          <RefreshCw size={14} className="animate-spin" />
          <span>Inference running - logs auto-refreshing...</span>
        </div>
      )}
    </div>
  );
}

export default LogViewer;

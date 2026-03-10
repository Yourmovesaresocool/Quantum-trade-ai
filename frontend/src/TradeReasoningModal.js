import React from 'react';

function TradeReasoningModal({ isOpen, onClose, tradeData }) {
  if (!isOpen || !tradeData) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 max-w-lg w-full mx-4 border border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">🧠 AI Reasoning</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ×
          </button>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-900 p-4 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Decision</p>
            <p className="text-2xl font-bold">{tradeData.action}</p>
          </div>

          <div className="bg-gray-900 p-4 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Confidence</p>
            <p className="text-xl font-semibold">{Math.round(tradeData.confidence * 100)}%</p>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 p-4 rounded-lg">
            <p className="text-sm text-blue-400 mb-2 font-semibold">Why this decision?</p>
            <p className="text-white">{tradeData.reason || 'AI reasoning not available'}</p>
          </div>

          <button
            onClick={onClose}
            className="w-full bg-blue-600 hover:bg-blue-700 p-3 rounded-lg font-bold"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default TradeReasoningModal;
import React, { useState } from 'react';
import { compareDocuments } from '../api';

const Compare = () => {
  const [textA, setTextA] = useState('');
  const [textB, setTextB] = useState('');
  const [threshold, setThreshold] = useState(0.8);
  const [useAgent, setUseAgent] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await compareDocuments(textA, textB, useAgent, threshold);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while comparing documents');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setTextA('');
    setTextB('');
    setResult(null);
    setError(null);
  };

  const getResultColor = (isParaphrase, confidence) => {
    if (isParaphrase) {
      return confidence > 0.7 ? 'text-emerald-700' : 'text-emerald-600';
    }
    return confidence > 0.7 ? 'text-zinc-700' : 'text-zinc-600';
  };

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-12">
        <h2 className="text-3xl font-semibold text-zinc-900 mb-3 tracking-tight">
          Compare Documents
        </h2>
        <p className="text-zinc-500 text-sm leading-relaxed max-w-2xl">
          Powered by Siamese Neural Networks with Sentence-BERT embeddings and AI agent validation.
          Enter two texts below to analyze their semantic similarity.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Text Input Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Text A */}
          <div className="space-y-3">
            <label htmlFor="textA" className="block text-sm font-medium text-zinc-700">
              Document A
            </label>
            <textarea
              id="textA"
              rows="12"
              className="w-full px-4 py-3 border border-gray-200 rounded-lg text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:border-transparent resize-none font-mono leading-relaxed"
              placeholder="Enter or paste your first document here..."
              value={textA}
              onChange={(e) => setTextA(e.target.value)}
              required
            />
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>{textA.length} characters</span>
              <span>{textA.split(/\s+/).filter(w => w).length} words</span>
            </div>
          </div>

          {/* Text B */}
          <div className="space-y-3">
            <label htmlFor="textB" className="block text-sm font-medium text-zinc-700">
              Document B
            </label>
            <textarea
              id="textB"
              rows="12"
              className="w-full px-4 py-3 border border-gray-200 rounded-lg text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:border-transparent resize-none font-mono leading-relaxed"
              placeholder="Enter or paste your second document here..."
              value={textB}
              onChange={(e) => setTextB(e.target.value)}
              required
            />
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>{textB.length} characters</span>
              <span>{textB.split(/\s+/).filter(w => w).length} words</span>
            </div>
          </div>
        </div>

        {/* Settings */}
        <div className="bg-zinc-50 border border-gray-100 rounded-lg p-6">
          <div className="flex flex-wrap items-center gap-8">
            {/* Threshold Slider */}
            <div className="flex-1 min-w-[240px]">
              <label htmlFor="threshold" className="block text-xs font-medium text-zinc-700 mb-3">
                Similarity Threshold: <span className="text-zinc-900 font-semibold">{(threshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                id="threshold"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
              />
              <div className="flex justify-between mt-2 text-[10px] text-zinc-400">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            {/* AI Agent Toggle */}
            <div className="flex items-center gap-3">
              <div className="relative inline-block">
                <input
                  type="checkbox"
                  id="useAgent"
                  checked={useAgent}
                  onChange={(e) => setUseAgent(e.target.checked)}
                  className="sr-only peer"
                />
                <label
                  htmlFor="useAgent"
                  className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-zinc-900 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-zinc-900 cursor-pointer"
                ></label>
              </div>
              <label htmlFor="useAgent" className="text-xs font-medium text-zinc-700 cursor-pointer">
                AI Agent Validation
              </label>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={clearAll}
            className="px-6 py-2.5 text-sm font-medium text-zinc-600 hover:text-zinc-900 transition-colors"
          >
            Clear All
          </button>
          <button
            type="submit"
            disabled={loading || !textA || !textB}
            className="px-8 py-2.5 bg-zinc-900 text-white text-sm font-medium rounded-lg hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </span>
            ) : (
              'Compare Documents'
            )}
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mt-8 p-4 bg-red-50 border border-red-100 rounded-lg">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-red-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="text-sm font-medium text-red-900">Error</h4>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-12 space-y-6">
          {/* Main Result Card */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {/* Result Header */}
            <div className="bg-zinc-50 border-b border-gray-100 px-8 py-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-zinc-900">
                  Analysis Results
                </h3>
                <span className={`text-4xl font-bold ${getResultColor(result.is_paraphrase, result.confidence)}`}>
                  {(result.similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Result Body */}
            <div className="px-8 py-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {/* Verdict */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Verdict</div>
                  <div className="flex items-center gap-2">
                    {result.is_paraphrase ? (
                      <>
                        <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                        <span className="text-lg font-semibold text-emerald-700">Paraphrase</span>
                      </>
                    ) : (
                      <>
                        <div className="w-3 h-3 rounded-full bg-zinc-400"></div>
                        <span className="text-lg font-semibold text-zinc-700">Unique</span>
                      </>
                    )}
                  </div>
                </div>

                {/* Confidence */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Confidence</div>
                  <div className="text-lg font-semibold text-zinc-900">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-2">
                    <div
                      className="bg-zinc-900 h-2 rounded-full transition-all"
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Threshold */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Threshold Used</div>
                  <div className="text-lg font-semibold text-zinc-900">
                    {(result.threshold * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-zinc-500">
                    Similarity {result.similarity >= result.threshold ? '≥' : '<'} threshold
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* AI Agent Validation */}
          {result.agent_validation && (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <div className="bg-zinc-50 border-b border-gray-100 px-8 py-4">
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5 text-zinc-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h4 className="text-sm font-semibold text-zinc-900">AI Agent Validation</h4>
                </div>
              </div>
              <div className="px-8 py-6 space-y-4">
                {/* Validation Status */}
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${result.agent_validation.validated ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
                  <span className="text-sm text-zinc-700">
                    {result.agent_validation.validated ? 'Validated' : 'Requires Review'}
                  </span>
                </div>

                {/* Flags */}
                {result.agent_validation.flags && result.agent_validation.flags.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Detected Issues</div>
                    <div className="flex flex-wrap gap-2">
                      {result.agent_validation.flags.map((flag, idx) => (
                        <span key={idx} className="px-3 py-1 bg-amber-50 text-amber-700 text-xs font-medium rounded-full">
                          {flag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Analysis */}
                {result.agent_validation.llm_reasoning && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">AI Analysis</div>
                    <div className="bg-zinc-50 border border-gray-100 rounded-lg p-4">
                      <p className="text-sm text-zinc-700 leading-relaxed whitespace-pre-wrap">
                        {result.agent_validation.llm_reasoning}
                      </p>
                      {result.agent_validation.llm_prediction !== undefined && (
                        <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between">
                          <span className="text-xs text-zinc-500">AI Prediction:</span>
                          <span className="text-xs font-semibold text-zinc-900">
                            {result.agent_validation.llm_prediction === 1 ? 'Paraphrase' : 'Unique'} ({(result.agent_validation.llm_confidence * 100).toFixed(0)}%)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Suggested Action */}
                {result.agent_validation.suggested_action && (
                  <div className="bg-blue-50 border border-blue-100 rounded-lg p-4">
                    <div className="text-xs font-medium text-blue-900 mb-1">Suggested Action</div>
                    <p className="text-sm text-blue-800">{result.agent_validation.suggested_action}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Compare;

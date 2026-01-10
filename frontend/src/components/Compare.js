import React, { useState } from 'react';
import { compareDocuments } from '../api';
import TextType from './TextType';

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
      setError(err.response?.data?.detail || 'Failed to compare documents. Please ensure the backend is running.');
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

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-12 text-center fade-in">
        <h2 className="text-4xl font-bold text-white mb-4">
          <TextType 
            text={["Compare Your Documents", "Analyze Similarity", "Detect Paraphrases"]}
            typingSpeed={70}
            pauseDuration={3000}
            showCursor={true}
            cursorCharacter="_"
            className="inline-block"
          />
        </h2>
        <p className="text-white/80 text-lg max-w-2xl mx-auto">
          Powered by Siamese Neural Networks with SBERT embeddings and AI agent validation
        </p>
      </div>

      {/* Main Form */}
      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Text Input Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Text A */}
          <div className="glass rounded-2xl p-6 space-y-4 fade-in">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                <span className="w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs">A</span>
                Document A
              </label>
              <span className="text-xs text-gray-500">{textA.length} chars</span>
            </div>
            <textarea
              rows="14"
              className="w-full px-4 py-3 bg-white/50 border-2 border-indigo-100 rounded-xl text-gray-800 placeholder-gray-400 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 resize-none font-mono text-sm transition-all"
              placeholder="Paste your first document here..."
              value={textA}
              onChange={(e) => setTextA(e.target.value)}
              required
            />
          </div>

          {/* Text B */}
          <div className="glass rounded-2xl p-6 space-y-4 fade-in">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs">B</span>
                Document B
              </label>
              <span className="text-xs text-gray-500">{textB.length} chars</span>
            </div>
            <textarea
              rows="14"
              className="w-full px-4 py-3 bg-white/50 border-2 border-purple-100 rounded-xl text-gray-800 placeholder-gray-400 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-200 resize-none font-mono text-sm transition-all"
              placeholder="Paste your second document here..."
              value={textB}
              onChange={(e) => setTextB(e.target.value)}
              required
            />
          </div>
        </div>

        {/* Settings Panel */}
        <div className="glass rounded-2xl p-6 space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            Analysis Settings
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Threshold Slider */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-gray-700">Similarity Threshold</label>
                <span className="text-lg font-bold text-indigo-600">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-3 bg-gradient-to-r from-indigo-200 to-purple-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Strict (0%)</span>
                <span>Moderate (50%)</span>
                <span>Lenient (100%)</span>
              </div>
            </div>

            {/* AI Agent Toggle */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-gray-700">AI Agent Validation</label>
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  onClick={() => setUseAgent(!useAgent)}
                  className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                    useAgent ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                      useAgent ? 'translate-x-7' : 'translate-x-1'
                    }`}
                  />
                </button>
                <span className={`text-sm font-medium ${useAgent ? 'text-indigo-600' : 'text-gray-500'}`}>
                  {useAgent ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <p className="text-xs text-gray-500">
                Uses AI to validate edge cases and provide detailed analysis
              </p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={clearAll}
            className="px-6 py-3 bg-white/50 text-gray-700 rounded-xl font-medium hover:bg-white/80 transition-all"
          >
            Clear All
          </button>
          <button
            type="submit"
            disabled={loading || !textA || !textB}
            className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                Compare Documents
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mt-8 glass-dark rounded-2xl p-6 border-l-4 border-red-500 fade-in">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-red-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="text-white font-semibold">Error</h4>
              <p className="text-white/80 text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-12 space-y-6 fade-in">
          {/* Main Result Card */}
          <div className="glass rounded-2xl overflow-hidden shadow-2xl">
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 px-8 py-6">
              <div className="flex items-center justify-between text-white">
                <h3 className="text-2xl font-bold">Analysis Results</h3>
                <div className="text-right">
                  <div className="text-5xl font-bold">{(result.similarity * 100).toFixed(1)}%</div>
                  <div className="text-sm opacity-90">Similarity Score</div>
                </div>
              </div>
            </div>

            <div className="p-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Verdict */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">Verdict</div>
                  <div className="flex items-center gap-3">
                    {result.is_paraphrase ? (
                      <>
                        <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-2xl font-bold text-green-600">Paraphrase</span>
                      </>
                    ) : (
                      <>
                        <div className="w-4 h-4 bg-gray-400 rounded-full"></div>
                        <span className="text-2xl font-bold text-gray-600">Unique</span>
                      </>
                    )}
                  </div>
                </div>

                {/* Confidence */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 transition-all duration-500"
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Threshold */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">Threshold</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {(result.threshold * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-gray-600">
                    Score {result.similarity >= result.threshold ? '≥' : '<'} threshold
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* AI Agent Validation */}
          {result.agent_validation && (
            <div className="glass rounded-2xl overflow-hidden">
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-4">
                <div className="flex items-center gap-3 text-white">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h4 className="text-lg font-semibold">AI Agent Analysis</h4>
                </div>
              </div>
              
              <div className="p-6 space-y-4">
                {/* Validation Status */}
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${result.agent_validation.validated ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></div>
                  <span className="font-medium text-gray-800">
                    {result.agent_validation.validated ? 'Validation Passed' : 'Requires Review'}
                  </span>
                </div>

                {/* Flags */}
                {result.agent_validation.flags && result.agent_validation.flags.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-500 uppercase">Detected Issues</div>
                    <div className="flex flex-wrap gap-2">
                      {result.agent_validation.flags.map((flag, idx) => (
                        <span key={idx} className="px-3 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">
                          {flag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Reasoning */}
                {result.agent_validation.llm_reasoning && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-500 uppercase">AI Insights</div>
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 border-l-4 border-purple-500 rounded-lg p-4">
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {result.agent_validation.llm_reasoning}
                      </p>
                      {result.agent_validation.llm_prediction !== undefined && (
                        <div className="mt-3 pt-3 border-t border-purple-200 flex justify-between items-center">
                          <span className="text-xs text-gray-600">AI Prediction:</span>
                          <span className="text-sm font-semibold text-purple-700">
                            {result.agent_validation.llm_prediction === 1 ? 'Paraphrase' : 'Unique'} 
                            ({(result.agent_validation.llm_confidence * 100).toFixed(0)}% confidence)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Suggested Action */}
                {result.agent_validation.suggested_action && (
                  <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4">
                    <div className="text-xs font-medium text-blue-900 mb-1">Recommendation</div>
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

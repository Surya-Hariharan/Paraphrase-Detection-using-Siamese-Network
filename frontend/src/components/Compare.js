import React, { useState } from 'react';
import { compareDocuments } from '../api';

const Compare = () => {
  const [docA, setDocA] = useState('');
  const [docB, setDocB] = useState('');
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [inputMode, setInputMode] = useState('text'); // 'text' or 'file'
  const [useAgents, setUseAgents] = useState(true);
  const [provider, setProvider] = useState('groq');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await compareDocuments(docA, docB, useAgents, provider);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'DUPLICATE':
        return 'text-red-600 bg-red-50';
      case 'PARAPHRASE':
        return 'text-yellow-600 bg-yellow-50';
      case 'UNIQUE':
        return 'text-green-600 bg-green-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900">Compare Documents</h2>
        <p className="mt-2 text-gray-600">
          Compare two documents for paraphrase detection using Siamese neural networks and AI agents
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="docA" className="block text-sm font-medium text-gray-700 mb-2">
              Document A
            </label>
            <textarea
              id="docA"
              rows="10"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
              placeholder="Enter first document text..."
              value={docA}
              onChange={(e) => setDocA(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="docB" className="block text-sm font-medium text-gray-700 mb-2">
              Document B
            </label>
            <textarea
              id="docB"
              rows="10"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
              placeholder="Enter second document text..."
              value={docB}
              onChange={(e) => setDocB(e.target.value)}
              required
            />
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-md">
          <div className="flex items-center space-x-6">
            <div className="flex items-center">
              <input
                id="useAgents"
                type="checkbox"
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                checked={useAgents}
                onChange={(e) => setUseAgents(e.target.checked)}
              />
              <label htmlFor="useAgents" className="ml-2 block text-sm text-gray-700">
                Use Multi-Agent Analysis
              </label>
            </div>

            <div className="flex items-center space-x-2">
              <label htmlFor="provider" className="text-sm text-gray-700">
                LLM Provider:
              </label>
              <select
                id="provider"
                className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                <option value="groq">Groq</option>
                <option value="ollama">Ollama</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Compare Documents'}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
            <div className="px-4 py-5 sm:px-6 bg-gray-50">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Analysis Results
              </h3>
            </div>
            <div className="border-t border-gray-200 px-4 py-5 sm:p-6">
              <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Similarity Score</dt>
                  <dd className="mt-1 text-2xl font-semibold text-gray-900">
                    {(result.similarity_score * 100).toFixed(2)}%
                  </dd>
                </div>

                {result.verdict && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Verdict</dt>
                    <dd className={`mt-1 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getVerdictColor(result.verdict)}`}>
                      {result.verdict}
                    </dd>
                  </div>
                )}

                {result.confidence && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Confidence</dt>
                    <dd className="mt-1 text-lg font-medium text-gray-900">
                      {result.confidence}
                    </dd>
                  </div>
                )}

                <div>
                  <dt className="text-sm font-medium text-gray-500">Processing Time</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {result.processing_time.toFixed(2)}s
                  </dd>
                </div>

                <div>
                  <dt className="text-sm font-medium text-gray-500">Document A Length</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {result.doc_a_length} characters
                  </dd>
                </div>

                <div>
                  <dt className="text-sm font-medium text-gray-500">Document B Length</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {result.doc_b_length} characters
                  </dd>
                </div>
              </dl>

              {result.reasoning && (
                <div className="mt-6">
                  <dt className="text-sm font-medium text-gray-500 mb-2">Analysis Reasoning</dt>
                  <dd className="text-sm text-gray-900 bg-gray-50 p-4 rounded-md whitespace-pre-wrap">
                    {result.reasoning}
                  </dd>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Compare;

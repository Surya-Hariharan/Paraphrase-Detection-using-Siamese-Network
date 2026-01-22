import React, { useState, useRef, useMemo } from 'react';
import { compareDocuments, compareFiles } from '../api';
import TextType from './TextType.jsx';
import Threads from './Threads.jsx';

const Compare = () => {
  const [textA, setTextA] = useState('');
  const [textB, setTextB] = useState('');
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [threshold, setThreshold] = useState(0.8);
  const [useAgent, setUseAgent] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  const handleFileUpload = async (file, setTextFunc, setFileFunc) => {
    if (!file) return;
    
    const allowedExtensions = ['.txt', '.pdf', '.docx'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = allowedExtensions.some(ext => fileName.endsWith(ext));
    
    if (!hasValidExtension) {
      alert('Please upload only .txt, .pdf, or .docx files');
      return;
    }
    
    // Store file for API upload
    setFileFunc(file);
    
    // For text files, also show preview
    if (file.type === 'text/plain') {
      const text = await file.text();
      setTextFunc(text);
    } else {
      setTextFunc(`📄 ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let data;
      
      // Use file upload API if both files are provided
      if (fileA && fileB) {
        data = await compareFiles(fileA, fileB, useAgent, threshold);
      } 
      // Otherwise use text comparison API
      else {
        data = await compareDocuments(textA, textB, useAgent, threshold);
      }
      
      setResult(data);
    } catch (err) {
      // Extract detailed error message from response
      let errorMsg = 'Failed to compare documents.';
      
      if (err.response?.data?.detail) {
        errorMsg = err.response.data.detail;
      } else if (err.message) {
        errorMsg = err.message;
      }
      
      // Add helpful context
      if (err.code === 'ERR_NETWORK' || errorMsg.includes('Network Error')) {
        errorMsg += ' Please ensure the backend is running on http://localhost:8000';
      }
      
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setTextA('');
    setTextB('');
    setFileA(null);
    setFileB(null);
    setResult(null);
    setError(null);
    if (fileInputARef.current) fileInputARef.current.value = '';
    if (fileInputBRef.current) fileInputBRef.current.value = '';
  };

  const threadsBackground = useMemo(() => (
    <div className="fixed inset-0 opacity-30 pointer-events-none">
      <Threads
        color={[0, 0, 0]}
        amplitude={0.5}
        distance={0}
        enableMouseInteraction={false}
      />
    </div>
  ), []);

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Threads Background */}
      {threadsBackground}

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        
        {/* Header with Typing Effect */}
        <div className="mb-12 text-center fade-in">
          <h2 className="text-4xl font-bold text-black mb-4">
            <TextType 
              text={["Compare Your Documents", "AI-Powered Analysis", "Detect Paraphrases"]}
              typingSpeed={75}
              pauseDuration={1500}
              showCursor={true}
              cursorCharacter="|"
            />
          </h2>
          <p className="text-gray-600 text-lg">Powered by Siamese Neural Networks with AI agent validation</p>
        </div>

        {/* Main Form */}
        <form onSubmit={handleSubmit} className="space-y-8">
          
          {/* Two Text Boxes */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Text Box A */}
            <div className="border-2 border-gray-300 rounded-xl p-6 bg-white shadow-lg hover:shadow-xl transition-all hover:border-gray-400">
              <div className="flex items-center justify-between mb-4">
                <label className="text-sm font-semibold text-black">Document A</label>
                <button
                  type="button"
                  onClick={() => fileInputARef.current?.click()}
                  className="p-2 text-gray-600 hover:text-black hover:bg-gray-100 rounded-lg transition-all"
                  title="Upload file (.txt, .pdf, .docx)"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>
                <input
                  ref={fileInputARef}
                  type="file"
                  accept=".txt,.pdf,.doc,.docx"
                  onChange={(e) => handleFileUpload(e.target.files[0], setTextA, setFileA)}
                  className="hidden"
                />
              </div>
              <textarea
                rows="12"
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg text-black placeholder-gray-400 focus:outline-none focus:border-black focus:ring-2 focus:ring-black resize-none text-sm transition-all"
                placeholder="Paste your first document here or upload a file..."
                value={textA}
                onChange={(e) => setTextA(e.target.value)}
                required
              />
              <div className="mt-2 text-xs text-gray-500 text-right">{textA.length} characters</div>
            </div>

            {/* Text Box B */}
            <div className="border-2 border-gray-300 rounded-xl p-6 bg-white shadow-lg hover:shadow-xl transition-all hover:border-gray-400">
              <div className="flex items-center justify-between mb-4">
                <label className="text-sm font-semibold text-black">Document B</label>
                <button
                  type="button"
                  onClick={() => fileInputBRef.current?.click()}
                  className="p-2 text-gray-600 hover:text-black hover:bg-gray-100 rounded-lg transition-all"
                  title="Upload file (.txt, .pdf, .docx)"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>
                <input
                  ref={fileInputBRef}
                  type="file"
                  accept=".txt,.pdf,.doc,.docx"
                  onChange={(e) => handleFileUpload(e.target.files[0], setTextB, setFileB)}
                  className="hidden"
                />
              </div>
              <textarea
                rows="12"
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg text-black placeholder-gray-400 focus:outline-none focus:border-black focus:ring-2 focus:ring-black resize-none text-sm transition-all"
                placeholder="Paste your second document here or upload a file..."
                value={textB}
                onChange={(e) => setTextB(e.target.value)}
                required
              />
              <div className="mt-2 text-xs text-gray-500 text-right">{textB.length} characters</div>
            </div>
          </div>

          {/* Options */}
          <div className="border-2 border-gray-300 rounded-xl p-6 bg-white shadow-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Threshold Slider */}
              <div className={!useAgent ? 'opacity-50' : ''}>
                <div className="flex justify-between items-center mb-3">
                  <label className="text-sm font-semibold text-black">Similarity Threshold</label>
                  <span className="text-sm font-bold text-black">{(threshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  disabled={!useAgent}
                  className={`w-full h-2 bg-gray-200 rounded-lg appearance-none accent-black ${
                    useAgent ? 'cursor-pointer' : 'cursor-not-allowed'
                  }`}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>

              {/* AI Agent Toggle */}
              <div>
                <label className="text-sm font-semibold text-black block mb-3">AI Agent Validation</label>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setUseAgent(!useAgent)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      useAgent ? 'bg-black' : 'bg-gray-300'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        useAgent ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                  <span className={`text-sm ${useAgent ? 'text-black font-medium' : 'text-gray-500'}`}>
                    {useAgent ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Buttons */}
          <div className="flex gap-4 justify-center">
            <button
              type="submit"
              disabled={loading || !textA || !textB}
              className="px-8 py-3 bg-black text-white font-semibold rounded-lg hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              {loading ? 'Comparing...' : 'Compare'}
            </button>
            <button
              type="button"
              onClick={clearAll}
              className="px-8 py-3 bg-white text-black font-semibold rounded-lg border-2 border-black hover:bg-gray-100 transition-all shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Clear
            </button>
          </div>
        </form>

        {/* Results */}
        {error && (
          <div className="mt-8 p-6 bg-red-50 border-2 border-red-300 rounded-xl shadow-lg fade-in">
            <div className="flex items-start gap-3">
              <svg className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <h4 className="text-red-900 font-semibold mb-1">Error</h4>
                <p className="text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="mt-8 border-2 border-gray-300 rounded-xl p-8 bg-white shadow-xl">
            <h3 className="text-2xl font-bold text-black mb-6">Analysis Results</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="p-6 bg-gray-50 rounded-xl border-2 border-gray-200">
                <div className="text-sm text-gray-600 mb-2">Similarity Score</div>
                <div className="text-4xl font-bold text-black">
                  {(result.similarity * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="p-6 bg-gray-50 rounded-xl border-2 border-gray-200">
                <div className="text-sm text-gray-600 mb-2">Classification</div>
                <div className={`text-2xl font-bold ${result.is_paraphrase ? 'text-green-600' : 'text-red-600'}`}>
                  {result.is_paraphrase ? 'Paraphrase' : 'Not Paraphrase'}
                </div>
              </div>

              <div className="p-6 bg-gray-50 rounded-xl border-2 border-gray-200">
                <div className="text-sm text-gray-600 mb-2">Confidence</div>
                <div className="text-4xl font-bold text-black">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Agent Validation Info */}
            {result.agent_validation && (
              <div className="mb-6 p-6 bg-blue-50 border-2 border-blue-200 rounded-xl">
                <h4 className="text-lg font-semibold text-black mb-3 flex items-center gap-2">
                  <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  AI Agent Validation
                </h4>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-700">
                    <span className="font-semibold">Status:</span> {result.agent_validation.validated ? '✓ Validated' : '⚠ Needs Review'}
                  </p>
                  {result.agent_validation.flags && result.agent_validation.flags.length > 0 && (
                    <p className="text-gray-700">
                      <span className="font-semibold">Flags:</span> {result.agent_validation.flags.join(', ')}
                    </p>
                  )}
                  {result.agent_validation.suggested_action && (
                    <p className="text-gray-700">
                      <span className="font-semibold">Suggestion:</span> {result.agent_validation.suggested_action}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* AI Agent Analysis */}
            {result.agent_analysis && (
              <div className="border-t-2 border-gray-200 pt-6">
                <h4 className="text-lg font-semibold text-black mb-3">AI Agent Reasoning</h4>
                <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{result.agent_analysis}</p>
              </div>
            )}

            {/* Metadata */}
            <div className="border-t-2 border-gray-200 pt-6 mt-6 grid grid-cols-2 gap-4">
              <div className="text-sm text-gray-500">
                <span className="font-semibold">Processing Time:</span> {result.processing_time?.toFixed(3)}s
              </div>
              <div className="text-sm text-gray-500">
                <span className="font-semibold">Threshold Used:</span> {(result.threshold * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="relative z-10 border-t-2 border-gray-200 mt-16 bg-white/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="text-center text-sm text-gray-600">
            <p>© 2026 Paraphrase Detection. Powered by Siamese Neural Networks.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Compare;

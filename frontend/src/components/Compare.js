import React, { useState, useRef } from 'react';
import { compareDocuments } from '../api';
import TextType from './TextType';
import Threads from './Threads';

const Compare = () => {
  const [textA, setTextA] = useState('');
  const [textB, setTextB] = useState('');
  const [threshold, setThreshold] = useState(0.8);
  const [useAgent, setUseAgent] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  const handleFileUpload = async (file, setTextFunc) => {
    if (!file) return;
    
    const allowedTypes = [
      'text/plain',
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword'
    ];
    
    if (!allowedTypes.includes(file.type)) {
      alert('Please upload only .txt, .pdf, or .docx files');
      return;
    }
    
    if (file.type === 'text/plain') {
      const text = await file.text();
      setTextFunc(text);
    } else {
      alert('PDF and DOCX parsing will be implemented soon. Please use .txt files for now.');
    }
  };

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
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Threads Background */}
      <div className="fixed inset-0 opacity-20 pointer-events-none">
        <Threads
          color={[0, 0, 0]}
          amplitude={1}
          distance={0}
          enableMouseInteraction
        />
      </div>

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
                  onChange={(e) => handleFileUpload(e.target.files[0], setTextA)}
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
                  onChange={(e) => handleFileUpload(e.target.files[0], setTextB)}
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
              <div>
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
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
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
          <div className="mt-8 p-6 bg-red-50 border-2 border-red-200 rounded-xl shadow-lg">
            <p className="text-red-800 font-medium">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-8 border-2 border-gray-300 rounded-xl p-8 bg-white shadow-xl">
            <h3 className="text-2xl font-bold text-black mb-6">Analysis Results</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="p-6 bg-gray-50 rounded-xl border-2 border-gray-200">
                <div className="text-sm text-gray-600 mb-2">Similarity Score</div>
                <div className="text-4xl font-bold text-black">
                  {(result.similarity * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="p-6 bg-gray-50 rounded-xl border-2 border-gray-200">
                <div className="text-sm text-gray-600 mb-2">Classification</div>
                <div className={`text-2xl font-bold ${result.is_paraphrase ? 'text-green-600' : 'text-red-600'}`}>
                  {result.is_paraphrase ? 'Paraphrase Detected' : 'Not a Paraphrase'}
                </div>
              </div>
            </div>

            {result.agent_analysis && (
              <div className="border-t-2 border-gray-200 pt-6">
                <h4 className="text-lg font-semibold text-black mb-3">AI Agent Analysis</h4>
                <p className="text-gray-700 leading-relaxed">{result.agent_analysis}</p>
              </div>
            )}

            <div className="border-t-2 border-gray-200 pt-6 mt-6">
              <div className="text-sm text-gray-500">
                Processing Time: {result.processing_time?.toFixed(2)}s
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

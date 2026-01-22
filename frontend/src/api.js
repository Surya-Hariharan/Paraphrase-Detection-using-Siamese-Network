import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Configure axios defaults
axios.defaults.headers.post['Content-Type'] = 'application/json';

// Create axios instance with timeout
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add response interceptor for error handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      // Server responded with error status
      const detail = error.response.data?.detail || error.response.data?.message || 'Server error occurred';
      throw new Error(detail);
    } else if (error.request) {
      // Request made but no response
      throw new Error('Cannot connect to backend server. Please ensure it is running on http://localhost:8000');
    } else {
      // Error in request setup
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

/**
 * Get API root information
 */
export const getApiInfo = async () => {
  const response = await api.get('/');
  return response.data;
};

/**
 * Health check endpoint
 */
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Compare two documents for paraphrase detection
 * @param {string} textA - First text to compare
 * @param {string} textB - Second text to compare
 * @param {boolean} useCache - Whether to use inference cache
 * @returns {Promise<Object>} Comparison results
 */
export const compareDocuments = async (textA, textB, useCache = true) => {
  const startTime = Date.now();
  
  const response = await api.post('/inference/compare', {
    text1: textA,
    text2: textB,
    use_cache: useCache
  });
  
  const endTime = Date.now();
  const processingTime = (endTime - startTime) / 1000; // Convert to seconds
  
  return {
    similarity: response.data.similarity,
    is_paraphrase: response.data.is_paraphrase,
    inference_time_ms: response.data.inference_time_ms,
    cached: response.data.cached,
    processing_time: processingTime
  };
};

export default {
  getApiInfo,
  healthCheck,
  compareDocuments
};


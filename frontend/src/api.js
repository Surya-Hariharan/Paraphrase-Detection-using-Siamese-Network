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
 * @param {boolean} useAgent - Whether to use AI agent validation
 * @param {number} threshold - Similarity threshold (0-1)
 * @returns {Promise<Object>} Comparison results
 */
export const compareDocuments = async (textA, textB, useAgent = true, threshold = 0.8) => {
  const startTime = Date.now();
  
  const response = await api.post('/inference/compare', {
    text1: textA,
    text2: textB,
    threshold: threshold
  });
  
  const endTime = Date.now();
  const processingTime = (endTime - startTime) / 1000; // Convert to seconds
  
  return {
    ...response.data,
    processing_time: processingTime,
    threshold: threshold
  };
};

/**
 * Compare two uploaded files for paraphrase detection
 * @param {File} fileA - First file to compare
 * @param {File} fileB - Second file to compare
 * @param {boolean} useAgent - Whether to use AI agent validation
 * @param {number} threshold - Similarity threshold (0-1)
 * @returns {Promise<Object>} Comparison results
 */
export const compareFiles = async (fileA, fileB, useAgent = true, threshold = 0.8) => {
  const startTime = Date.now();
  
  const formData = new FormData();
  formData.append('file1', fileA);
  formData.append('file2', fileB);
  formData.append('threshold', threshold.toString());
  
  const response = await api.post('/inference/compare-files', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  const endTime = Date.now();
  const processingTime = (endTime - startTime) / 1000;
  
  return {
    ...response.data,
    processing_time: processingTime,
    threshold: threshold
  };
};

export default {
  getApiInfo,
  healthCheck,
  compareDocuments,
  compareFiles
};


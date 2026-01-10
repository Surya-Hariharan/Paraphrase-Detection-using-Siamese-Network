import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const compareDocuments = async (textA, textB, useAgent = true, threshold = 0.8) => {
  const response = await api.post('/compare', {
    text_a: textA,
    text_b: textB,
    use_agent: useAgent,
    threshold: threshold,
  });
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;

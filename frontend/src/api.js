import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const compareDocuments = async (textA, textB, useAgent = true, threshold = 0.8) => {
  const response = await axios.post(`${API_BASE_URL}/compare`, {
    text_a: textA,
    text_b: textB,
    use_agent: useAgent,
    threshold: threshold
  });
  return response.data;
};

export const healthCheck = async () => {
  const response = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
};

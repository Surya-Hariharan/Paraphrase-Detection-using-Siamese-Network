import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const compareDocuments = async (docA, docB, useAgents = true, provider = 'groq') => {
  const response = await api.post('/api/compare', {
    doc_a: docA,
    doc_b: docB,
    use_agents: useAgents,
    provider: provider,
  });
  return response.data;
};

export const searchDuplicates = async (query, collection = 'documents', topK = 5, provider = 'groq') => {
  const response = await api.post('/api/search', {
    query: query,
    collection: collection,
    top_k: topK,
    provider: provider,
  });
  return response.data;
};

export const indexFiles = async (files, collection = 'documents', provider = 'groq') => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await api.post(
    `/api/index/files?collection=${collection}&provider=${provider}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

export const indexBulk = async (documents, collection = 'documents', provider = 'groq') => {
  const response = await api.post('/api/index/bulk', {
    documents: documents,
    collection: collection,
    provider: provider,
  });
  return response.data;
};

export const listCollections = async () => {
  const response = await api.get('/api/collections');
  return response.data;
};

export const deleteCollection = async (collectionName) => {
  const response = await api.delete(`/api/collections/${collectionName}`);
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/');
  return response.data;
};

export default api;

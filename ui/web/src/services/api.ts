import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased to 60 seconds for complex reasoning
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  reply: string;
  route: string;
  intent: string;
  session_id: string;
  context?: Record<string, any>;
  structured_data?: Record<string, any>;
  recommendations?: string[];
  confidence?: number;
}

export interface EquipmentItem {
  sku: string;
  name: string;
  quantity: number;
  location: string;
  reorder_point: number;
  updated_at: string;
}

export interface Task {
  id: number;
  kind: string;
  status: string;
  assignee: string;
  payload: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface SafetyIncident {
  id: number;
  severity: string;
  description: string;
  reported_by: string;
  occurred_at: string;
}

export const chatAPI = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post('/api/v1/chat', request);
    return response.data;
  },
};

export const equipmentAPI = {
  getItem: async (sku: string): Promise<EquipmentItem> => {
    const response = await api.get(`/api/v1/equipment/${sku}`);
    return response.data;
  },
  
  getAllItems: async (): Promise<EquipmentItem[]> => {
    const response = await api.get('/api/v1/equipment');
    return response.data;
  },
  
  updateItem: async (sku: string, data: Partial<EquipmentItem>): Promise<EquipmentItem> => {
    const response = await api.put(`/api/v1/equipment/${sku}`, data);
    return response.data;
  },
  
  createItem: async (data: Omit<EquipmentItem, 'updated_at'>): Promise<EquipmentItem> => {
    const response = await api.post('/api/v1/equipment', data);
    return response.data;
  },
};

export const operationsAPI = {
  getTasks: async (): Promise<Task[]> => {
    const response = await api.get('/api/v1/operations/tasks');
    return response.data;
  },
  
  getWorkforceStatus: async (): Promise<any> => {
    const response = await api.get('/api/v1/operations/workforce');
    return response.data;
  },
  
  assignTask: async (taskId: number, assignee: string): Promise<Task> => {
    const response = await api.post(`/api/v1/operations/tasks/${taskId}/assign`, {
      assignee,
    });
    return response.data;
  },
};

export const safetyAPI = {
  getIncidents: async (): Promise<SafetyIncident[]> => {
    const response = await api.get('/api/v1/safety/incidents');
    return response.data;
  },
  
  reportIncident: async (data: Omit<SafetyIncident, 'id' | 'occurred_at'>): Promise<SafetyIncident> => {
    const response = await api.post('/api/v1/safety/incidents', data);
    return response.data;
  },
  
  getPolicies: async (): Promise<any[]> => {
    const response = await api.get('/api/v1/safety/policies');
    return response.data;
  },
};

export const healthAPI = {
  check: async (): Promise<{ ok: boolean }> => {
    const response = await api.get('/api/v1/health');
    return response.data;
  },
};

export default api;

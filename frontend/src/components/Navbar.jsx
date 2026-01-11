import React, { useState, useEffect } from 'react';
import { healthCheck } from '../api';

const Navbar = () => {
  const [backendStatus, setBackendStatus] = useState({ connected: false, modelLoaded: false });

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const health = await healthCheck();
        setBackendStatus({
          connected: true,
          modelLoaded: health.model_loaded
        });
      } catch (error) {
        setBackendStatus({
          connected: false,
          modelLoaded: false
        });
      }
    };

    // Check immediately
    checkBackendHealth();

    // Check every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);

    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b-2 border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-black">Paraphrase Detection</h1>
          </div>
          
          {/* Backend Status Indicator */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${backendStatus.connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-xs font-medium text-gray-600">
              {backendStatus.connected 
                ? (backendStatus.modelLoaded ? 'Model Ready' : 'Model Loading...') 
                : 'Backend Offline'}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

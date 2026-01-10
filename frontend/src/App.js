import React from 'react';
import Navbar from './components/Navbar';
import Compare from './components/Compare';

function App() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <div className="pt-16">
        <Compare />
      </div>
    </div>
  );
}

export default App;

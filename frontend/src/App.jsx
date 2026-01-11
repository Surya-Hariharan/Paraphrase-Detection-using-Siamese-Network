import React from 'react';
import Navbar from './components/Navbar.jsx';
import Compare from './components/Compare.jsx';

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

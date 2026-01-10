import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import TextType from './TextType';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="glass border-b border-white/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">
                Paraphrase Detector
              </h1>
              <div className="text-sm text-indigo-600 font-medium">
                <TextType 
                  text={["AI-Powered Analysis", "Siamese Networks", "Real-time Detection"]}
                  typingSpeed={60}
                  pauseDuration={2000}
                  showCursor={false}
                  className="inline-block"
                />
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className={`text-sm font-medium ${
                isActive('/') 
                  ? 'text-indigo-600' 
                  : 'text-gray-600 hover:text-indigo-600'
              }`}
            >
              Compare
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium text-gray-600">Online</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

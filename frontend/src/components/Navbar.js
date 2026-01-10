import React from 'react';
import TextType from './TextType';

const Navbar = () => {
  return (
    <nav className="glass sticky top-0 z-50 border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-gray-900">
              <TextType 
                text={["Paraphrase Detection"]}
                typingSpeed={75}
                loop={false}
                showCursor={false}
              />
            </h1>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

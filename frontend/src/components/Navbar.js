import React from 'react';
import TextType from './TextType';

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b-2 border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-black">Paraphrase Detection</h1>
            <div className="text-sm text-gray-600 font-medium">
              <TextType 
                text={["AI-Powered Analysis", "Siamese Networks", "Minimalist UI"]}
                typingSpeed={70}
                pauseDuration={2000}
                showCursor={true}
                cursorCharacter="|"
              />
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

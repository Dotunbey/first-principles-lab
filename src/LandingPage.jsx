import React from 'react';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center items-center">
      <header className="w-full py-6 bg-blue-600 text-white text-center text-xl font-bold shadow-md">
        Professional Landing Page
      </header>

      <main className="flex flex-col items-center justify-center px-6 sm:px-16 py-12 bg-white shadow-lg rounded-lg max-w-sm w-full">
        <h2 className="mb-4 text-2xl font-semibold text-gray-800">Login</h2>
        <form className="w-full space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
            <input 
              type="email" 
              id="email" 
              className="w-full border border-gray-300 px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500" 
              placeholder="Enter your email" 
              required 
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input 
              type="password" 
              id="password" 
              className="w-full border border-gray-300 px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500" 
              placeholder="Enter your password" 
              required 
            />
          </div>

          <button 
            type="submit" 
            className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition duration-200">
            Login
          </button>
        </form>
      </main>

      <footer className="w-full py-4 text-center text-sm text-gray-600 mt-auto">
        &copy; 2026 Professional Landing Page. All rights reserved.
      </footer>
    </div>
  );
};

export default LandingPage;
import React, { useState } from 'react';
import CloudDevopsArchitectPage from './pages/CloudDevopsArchitectPage';
import GCPArchitectureDashboardPage from './pages/GCPArchitectureDashboardPage';
import { GCPConnectionProvider, useGCPConnection } from './context/GCPConnectionContext';
import UserMenu from './components/UserMenu';
import ServiceAccountModal from './components/ServiceAccountModal';

type Tab = 'architect' | 'canvas';

const AppContent: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('architect');
  const { isModalOpen, setIsModalOpen } = useGCPConnection();

  const getTabClass = (tabName: Tab) => {
    return `px-4 py-2 text-sm font-medium rounded-md transition-colors ${
      activeTab === tabName
        ? 'bg-blue-600 text-white'
        : 'text-gray-300 hover:bg-gray-700'
    }`;
  };

  return (
    <div className="flex flex-col h-screen font-sans bg-gray-900 text-gray-100 overflow-hidden">
      <header className="flex items-center justify-between p-4 border-b border-gray-700 shadow-md bg-gray-800 flex-shrink-0">
        <div className="flex items-center space-x-6">
          <h1 className="text-xl font-bold text-white">Vibe DevOps</h1>
          <nav className="flex items-center space-x-2 bg-gray-900 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('architect')}
              className={getTabClass('architect')}
              aria-current={activeTab === 'architect' ? 'page' : undefined}
            >
              Cloud Devops Architect
            </button>
            <button
              onClick={() => setActiveTab('canvas')}
              className={getTabClass('canvas')}
              aria-current={activeTab === 'canvas' ? 'page' : undefined}
            >
              Live Architecture Canvas
            </button>
          </nav>
        </div>
        <UserMenu />
      </header>
      <div className="flex-1 overflow-hidden">
        {activeTab === 'architect' && <CloudDevopsArchitectPage />}
        {activeTab === 'canvas' && <GCPArchitectureDashboardPage />}
      </div>
      {isModalOpen && <ServiceAccountModal onClose={() => setIsModalOpen(false)} />}
    </div>
  );
};


const App: React.FC = () => {
  return (
    <GCPConnectionProvider>
      <AppContent />
    </GCPConnectionProvider>
  );
};

export default App;

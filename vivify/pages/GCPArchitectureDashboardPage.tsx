import React, { useState } from 'react';
import { useGCPArchitectureStore } from '../context/GCPArchitectureStore';
import GCPArchitectureDashboard from '../components/gcp/GCPArchitectureDashboard';
import { GCPService } from '../types/gcp';
import { AlertTriangleIcon } from '../components/icons/AlertTriangleIcon';
import { useGCPConnection } from '../context/GCPConnectionContext';

const GCPArchitectureDashboardPage: React.FC = () => {
  const { hasGCPAccess, credentials, projectId, setIsModalOpen } = useGCPConnection();
  const { architecture, loading, error, lastFetch, fetchArchitecture } = useGCPArchitectureStore();
  const [selectedResource, setSelectedResource] = useState<GCPService | null>(null);

  // Manual refresh function
  const handleRefresh = () => {
    if (credentials && projectId) {
      fetchArchitecture(credentials, projectId);
    }
  };

  const handleResourceSelect = (resource: GCPService | null) => {
    setSelectedResource(resource);
  };

  const renderContent = () => {
    // Not connected to GCP
    if (!hasGCPAccess) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-gray-400 text-center">
          <AlertTriangleIcon className="w-12 h-12 text-yellow-500 mb-4" />
          <h2 className="text-xl font-semibold mb-2">Connect to GCP</h2>
          <p className="max-w-md">To visualize your infrastructure, please provide a read-only GCP Service Account key.</p>
          <button 
            onClick={() => setIsModalOpen(true)}
            className="mt-6 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
          >
            Connect to GCP
          </button>
        </div>
      );
    }

    // Connected but no data yet - show empty state
    if (!architecture && !loading && !error) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-gray-400 text-center p-8">
          <div className="text-6xl mb-6">🔍</div>
          <h2 className="text-2xl font-semibold mb-3 text-gray-200">Ready to Discover Your Infrastructure</h2>
          <p className="max-w-md mb-8 text-gray-400">
            Click the button below to scan your GCP project <span className="font-mono text-blue-400">{projectId}</span> and visualize all resources.
          </p>
          <button 
            onClick={handleRefresh}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <span>🔍</span>
            <span>Discover Resources</span>
          </button>
        </div>
      );
    }

    if (loading) {
      return (
        <div className="flex items-center justify-center h-full text-gray-400">
          <p>Discovering GCP resources...</p>
        </div>
      );
    }

    if (error) {
      const isApiDisabled = error.message.includes('API has not been used') || error.message.includes('is disabled');
      
      return (
        <div className="p-8 max-w-4xl mx-auto">
          <div className={`border px-6 py-4 rounded-lg ${isApiDisabled ? 'bg-yellow-900/30 border-yellow-700' : 'bg-red-900 border-red-700'}`} role="alert">
            <div className="flex items-start">
              <AlertTriangleIcon className={`w-6 h-6 mr-3 flex-shrink-0 mt-0.5 ${isApiDisabled ? 'text-yellow-500' : 'text-red-500'}`} />
              <div className="flex-1">
                <h3 className={`font-bold text-lg mb-2 ${isApiDisabled ? 'text-yellow-200' : 'text-red-200'}`}>
                  {isApiDisabled ? 'GCP APIs Not Enabled' : 'Discovery Failed'}
                </h3>
                <p className={`text-sm mb-4 ${isApiDisabled ? 'text-yellow-100' : 'text-red-200'}`}>
                  {error.message}
                </p>
                
                {isApiDisabled && (
                  <div className="bg-gray-800 p-4 rounded-md mb-4">
                    <h4 className="font-semibold text-white mb-2">To enable GCP APIs:</h4>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-gray-300">
                      <li>Go to the <a href="https://console.cloud.google.com/apis/library" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">GCP API Library</a></li>
                      <li>Search for and enable these APIs:
                        <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                          <li>Compute Engine API</li>
                          <li>Kubernetes Engine API</li>
                          <li>Cloud Storage API</li>
                        </ul>
                      </li>
                      <li>Wait 1-2 minutes for changes to propagate</li>
                      <li>Click "Retry Discovery" below</li>
                    </ol>
                  </div>
                )}
                
                <button 
                  onClick={handleRefresh} 
                  className={`px-4 py-2 rounded font-medium transition-colors ${
                    isApiDisabled 
                      ? 'bg-yellow-600 hover:bg-yellow-500 text-white' 
                      : 'bg-red-700 hover:bg-red-600 text-white'
                  }`}
                >
                  Retry Discovery
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    if (!architecture) {
        return <div className="flex items-center justify-center h-full text-gray-400">No architecture data found.</div>
    }
    
    // Handle empty state (no resources discovered)
    if (architecture.resources.length === 0) {
      return (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center max-w-2xl">
            <AlertTriangleIcon className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <h2 className="text-2xl font-semibold text-gray-300 mb-2">No Resources Found</h2>
            <p className="text-gray-400 mb-6">
              Discovery completed successfully, but no resources were found in project <span className="font-mono text-blue-400">{architecture.project}</span>.
            </p>
            <div className="bg-gray-800 p-6 rounded-lg text-left mb-6">
              <h3 className="font-semibold text-white mb-3">This could be because:</h3>
              <ul className="space-y-2 text-sm text-gray-300">
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">•</span>
                  <span>The project doesn't have any resources yet</span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">•</span>
                  <span>Required GCP APIs are not enabled (Compute Engine, GKE, Storage)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">•</span>
                  <span>Service account lacks necessary permissions (needs Viewer role)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">•</span>
                  <span>Resources are in regions that weren't scanned</span>
                </li>
              </ul>
            </div>
            <div className="flex justify-center space-x-4">
              <button 
                onClick={handleRefresh}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-md font-medium transition-colors"
              >
                Retry Discovery
              </button>
              <button 
                onClick={() => setIsModalOpen(true)}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md font-medium transition-colors"
              >
                Update Credentials
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <GCPArchitectureDashboard
        architecture={architecture}
        onRefresh={handleRefresh}
        selectedResource={selectedResource}
        onResourceSelect={handleResourceSelect}
        lastFetch={lastFetch}
      />
    );
  }

  return (
    <>
      {renderContent()}
    </>
  );
};

export default GCPArchitectureDashboardPage;

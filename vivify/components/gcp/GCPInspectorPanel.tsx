import React, { useState } from 'react';
import { GCPService } from '../../types/gcp';
import { XIcon } from '../icons/XIcon';
import { ExternalLinkIcon } from '../icons/ExternalLinkIcon';
import { getGCPConsoleLink } from '../../utils/gcp';

interface GCPInspectorPanelProps {
  resource: GCPService;
  onClose: () => void;
}

type Tab = 'details' | 'metrics' | 'cost' | 'security' | 'connections';

const GCPInspectorPanel: React.FC<GCPInspectorPanelProps> = ({ resource, onClose }) => {
  const [activeTab, setActiveTab] = useState<Tab>('details');

  const consoleLink = getGCPConsoleLink(resource);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'details':
        return (
          <div className="space-y-4">
            <div>
              <h4 className="text-xs font-semibold text-gray-400 mb-1">Configuration</h4>
              <div className="text-xs bg-gray-900 p-2 rounded-md font-mono text-gray-300 max-h-48 overflow-auto">
                <pre>{JSON.stringify(resource.configuration, null, 2)}</pre>
              </div>
            </div>
             <div>
              <h4 className="text-xs font-semibold text-gray-400 mb-1">Labels</h4>
              <div className="flex flex-wrap gap-2">
                {resource.labels && Object.entries(resource.labels).map(([key, value]) => (
                   <span key={key} className="px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded-full">
                       {key}:<span className="font-semibold text-white ml-1">{value}</span>
                   </span>
                ))}
              </div>
            </div>
          </div>
        );
      case 'metrics':
        return <div className="text-center text-gray-500 py-8">Metrics coming soon.</div>;
      case 'cost':
        return <div className="text-center text-gray-500 py-8">Cost analysis coming soon.</div>;
      default:
        return <div className="text-center text-gray-500 py-8">Information coming soon.</div>;
    }
  };
  
  const getTabClass = (tab: Tab) => `px-3 py-2 text-sm font-medium rounded-md transition-colors ${activeTab === tab ? 'bg-blue-600 text-white' : 'text-gray-400 hover:bg-gray-700'}`;

  return (
    <div className="flex flex-col h-full text-sm">
      {/* Panel Header */}
      <header className="p-4 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-bold text-lg text-white break-all">{resource.name}</h3>
          <button onClick={onClose} className="p-1 rounded-full hover:bg-gray-700">
            <XIcon className="w-5 h-5 text-gray-400" />
          </button>
        </div>
        <p className="text-xs text-gray-400 capitalize">{resource.type.replace('google_', '').replace(/_/g, ' ')}</p>
        {consoleLink && (
          <a href={consoleLink} target="_blank" rel="noopener noreferrer" className="inline-flex items-center text-xs text-blue-400 hover:text-blue-300 mt-2">
            View in GCP Console <ExternalLinkIcon className="w-3 h-3 ml-1.5" />
          </a>
        )}
      </header>

      {/* Tabs */}
      <nav className="p-2 flex-shrink-0 border-b border-gray-700">
        <div className="flex items-center space-x-1">
            <button onClick={() => setActiveTab('details')} className={getTabClass('details')}>Details</button>
            <button onClick={() => setActiveTab('metrics')} className={getTabClass('metrics')}>Metrics</button>
            <button onClick={() => setActiveTab('cost')} className={getTabClass('cost')}>Cost</button>
            <button onClick={() => setActiveTab('security')} className={getTabClass('security')}>Security</button>
            <button onClick={() => setActiveTab('connections')} className={getTabClass('connections')}>Connections</button>
        </div>
      </nav>

      {/* Tab Content */}
      <div className="flex-1 p-4 overflow-y-auto">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default GCPInspectorPanel;

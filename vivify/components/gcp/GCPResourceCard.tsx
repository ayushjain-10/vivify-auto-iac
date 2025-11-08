import React from 'react';
import { GCPService } from '../../types/gcp';
import { getGCPIcon } from '../../utils/gcp';

interface GCPResourceCardProps {
  resource: GCPService;
  isSelected: boolean;
  onSelect: () => void;
}

const HEALTH_STATUS_CONFIG = {
  healthy: { dot: 'bg-green-500', border: 'border-l-green-500' },
  warning: { dot: 'bg-yellow-500', border: 'border-l-yellow-500' },
  critical: { dot: 'bg-red-500', border: 'border-l-red-500' },
  unknown: { dot: 'bg-gray-500', border: 'border-l-gray-500' },
};

const STATUS_CONFIG = {
    running: 'text-green-400 bg-green-900/50',
    stopped: 'text-yellow-400 bg-yellow-900/50',
    deploying: 'text-blue-400 bg-blue-900/50',
    error: 'text-red-400 bg-red-900/50',
};

const GCPResourceCard: React.FC<GCPResourceCardProps> = ({ resource, isSelected, onSelect }) => {
  const Icon = getGCPIcon(resource.type);
  const healthConfig = HEALTH_STATUS_CONFIG[resource.healthStatus];
  const statusConfig = STATUS_CONFIG[resource.status];

  return (
    <div
      onClick={onSelect}
      className={`bg-gray-800 rounded-lg p-3 cursor-pointer transition-all duration-200 shadow-md hover:shadow-lg hover:scale-[1.02] border-l-4 ${healthConfig.border} ${isSelected ? 'ring-2 ring-blue-500' : 'ring-0 ring-transparent'}`}
    >
      {/* Card Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 bg-gray-700 rounded-md">
            <Icon className="w-5 h-5 text-gray-300" />
          </div>
          <div>
             <h3 className="font-semibold text-sm text-white truncate w-32" title={resource.name}>{resource.name}</h3>
             <p className="text-xs text-gray-400 capitalize">{resource.type.replace('google_', '').replace(/_/g, ' ')}</p>
          </div>
        </div>
        <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusConfig}`}>{resource.status}</span>
      </div>
      
      {/* Card Body - Key Metrics */}
      <div className="text-xs text-gray-400 space-y-1 my-3">
          {resource.metrics?.cpu !== undefined && (
            <div className="flex justify-between"><span>CPU</span> <span className="font-mono text-gray-200">{resource.metrics.cpu}%</span></div>
          )}
          {resource.metrics?.memory !== undefined && (
            <div className="flex justify-between"><span>Memory</span> <span className="font-mono text-gray-200">{resource.metrics.memory}%</span></div>
          )}
           {resource.costEstimate && (
            <div className="flex justify-between"><span>Cost/Mo</span> <span className="font-mono text-gray-200">${resource.costEstimate.monthly.toFixed(2)}</span></div>
          )}
      </div>

      {/* Card Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-700">
        <div className="flex items-center space-x-2">
           <span className={`w-2.5 h-2.5 rounded-full ${healthConfig.dot}`}></span>
           <span className="text-xs text-gray-400 capitalize">{resource.healthStatus}</span>
        </div>
        <div className="text-xs text-gray-500">
          {resource.zone || resource.region}
        </div>
      </div>
    </div>
  );
};

export default GCPResourceCard;

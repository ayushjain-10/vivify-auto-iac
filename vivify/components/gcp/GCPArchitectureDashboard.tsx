import React, { useMemo } from 'react';
import { GCPArchitecture, GCPService } from '../../types/gcp';
import { RefreshCcwIcon } from '../icons/RefreshCcwIcon';
import { FilterIcon } from '../icons/FilterIcon';
import GCPZoneGroup from './GCPZoneGroup';
import GCPInspectorPanel from './GCPInspectorPanel';

interface GCPArchitectureDashboardProps {
  architecture: GCPArchitecture;
  onRefresh: () => void;
  selectedResource: GCPService | null;
  onResourceSelect: (resource: GCPService | null) => void;
  lastFetch?: Date | null;
}

// Resource type display configuration
const RESOURCE_TYPE_CONFIG: Record<string, { label: string; icon: string; order: number }> = {
  'google_storage_bucket': { label: 'Cloud Storage Buckets', icon: '🪣', order: 1 },
  'google_compute_instance': { label: 'Compute Engine VMs', icon: '💻', order: 2 },
  'google_container_cluster': { label: 'GKE Clusters', icon: '☸️', order: 3 },
  'google_compute_network': { label: 'VPC Networks', icon: '🌐', order: 4 },
  'google_compute_firewall': { label: 'Firewall Rules', icon: '🛡️', order: 5 },
  'google_sql_database_instance': { label: 'Cloud SQL Databases', icon: '🗄️', order: 6 },
  'google_cloud_functions_function': { label: 'Cloud Functions', icon: '⚡', order: 7 },
  'google_cloud_run_service': { label: 'Cloud Run Services', icon: '🏃', order: 8 },
};

const GCPArchitectureDashboard: React.FC<GCPArchitectureDashboardProps> = ({
  architecture,
  onRefresh,
  selectedResource,
  onResourceSelect,
  lastFetch,
}) => {
  // Group resources by type instead of region/zone
  const groupedByType = useMemo(() => {
    const typeGroups: Record<string, GCPService[]> = {};

    architecture.resources.forEach(resource => {
      const type = resource.type;
      if (!typeGroups[type]) {
        typeGroups[type] = [];
      }
      typeGroups[type].push(resource);
    });

    // Convert to array and sort by order
    return Object.entries(typeGroups)
      .map(([type, resources]) => ({
        type,
        label: RESOURCE_TYPE_CONFIG[type]?.label || type,
        icon: RESOURCE_TYPE_CONFIG[type]?.icon || '📦',
        order: RESOURCE_TYPE_CONFIG[type]?.order || 999,
        resources,
      }))
      .sort((a, b) => a.order - b.order);
  }, [architecture.resources]);

  return (
    <div className="flex h-full bg-gray-900">
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header Bar */}
        <header className="flex items-center justify-between p-3 border-b border-gray-700 bg-gray-800 flex-shrink-0">
          <div className="flex items-center space-x-4">
            <h2 className="font-semibold text-lg">{architecture.project}</h2>
            <div className="flex items-center space-x-2">
              <span className="text-xs px-2 py-1 bg-gray-700 text-gray-300 rounded-full">
                {architecture.resources.length} resources
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">
              Total Cost: <span className="font-bold text-white">${architecture.totalCost.toFixed(2)}/mo</span>
            </div>
            <div className="text-xs text-gray-500">
              {lastFetch ? (
                <>Last refresh: {lastFetch.toLocaleTimeString()}</>
              ) : (
                <>Last refresh: {new Date(architecture.lastRefresh).toLocaleTimeString()}</>
              )}
            </div>
            <button 
              onClick={onRefresh} 
              className="flex items-center space-x-1 px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded-md transition-colors"
              title="Refresh architecture data"
            >
              <RefreshCcwIcon className="w-4 h-4" />
              <span className="text-sm">Refresh</span>
            </button>
          </div>
        </header>

        {/* Main Canvas */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 space-y-8">
            {/* Group by Resource Type */}
            {groupedByType.map(({ type, label, icon, resources }) => (
                <div key={type} className="space-y-4">
                    <div className="flex items-center space-x-3 border-b border-gray-700 pb-3">
                        <span className="text-2xl">{icon}</span>
                        <h3 className="text-lg font-semibold text-gray-200">{label}</h3>
                        <span className="text-sm font-mono bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">
                            {resources.length}
                        </span>
                        {/* Show total cost for this resource type */}
                        {resources.some(r => r.costEstimate) && (
                          <span className="text-sm text-gray-400">
                            ${resources.reduce((sum, r) => sum + (r.costEstimate?.monthly || 0), 0).toFixed(2)}/mo
                          </span>
                        )}
                    </div>
                    <GCPZoneGroup
                        title=""
                        resources={resources}
                        selectedResourceId={selectedResource?.id}
                        onResourceSelect={onResourceSelect}
                    />
                </div>
            ))}
            {/* Show message if no resources */}
            {groupedByType.length === 0 && (
                <div className="text-center text-gray-500 py-12">
                    <div className="text-6xl mb-4">📦</div>
                    <p className="text-lg">No resources found</p>
                    <p className="text-sm mt-2">Try refreshing or check your GCP project</p>
                </div>
            )}
        </main>
      </div>

      {/* Inspector Panel */}
      <aside
        className={`transition-all duration-300 ease-in-out bg-gray-800 border-l border-gray-700 overflow-hidden ${
          selectedResource ? 'w-full max-w-sm md:max-w-md lg:max-w-lg' : 'w-0'
        }`}
      >
        {selectedResource && <GCPInspectorPanel resource={selectedResource} onClose={() => onResourceSelect(null)} />}
      </aside>
    </div>
  );
};

export default GCPArchitectureDashboard;

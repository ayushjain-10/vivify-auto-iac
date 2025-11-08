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
  isFromCache?: boolean;
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
}) => {
  const groupedResources = useMemo(() => {
    const groups: Record<string, Record<string, GCPService[]>> = {};
    const globalResources: GCPService[] = [];

    architecture.resources.forEach(resource => {
      const region = resource.region || 'global';
      if (region === 'global' || !resource.zone) {
        globalResources.push(resource);
        return;
      }

      const zone = resource.zone;
      if (!groups[region]) groups[region] = {};
      if (!groups[region][zone]) groups[region][zone] = [];
      groups[region][zone].push(resource);
    });
    
    // Sort regions and zones
    const sortedGroups: Record<string, Record<string, GCPService[]>> = {};
    Object.keys(groups).sort().forEach(region => {
        sortedGroups[region] = {};
        Object.keys(groups[region]).sort().forEach(zone => {
            sortedGroups[region][zone] = groups[region][zone];
        });
    });

    return { global: globalResources, regional: sortedGroups };
  }, [architecture.resources]);

  return (
    <div className="flex h-full bg-gray-900">
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header Bar */}
        <header className="flex items-center justify-between p-3 border-b border-gray-700 bg-gray-800 flex-shrink-0">
          <div className="flex items-center space-x-4">
            <h2 className="font-semibold text-lg">{architecture.project}</h2>
            <div className="flex items-center space-x-2">
              <button className="flex items-center px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-md text-sm">
                <FilterIcon className="w-4 h-4 mr-2" />
                <span>All Regions</span>
              </button>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">
              Total Cost: <span className="font-bold text-white">${architecture.totalCost.toFixed(2)}/mo</span>
            </div>
            <div className="text-xs text-gray-500">
              Last refresh: {new Date(architecture.lastRefresh).toLocaleTimeString()}
            </div>
            <button onClick={onRefresh} className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md">
              <RefreshCcwIcon className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Main Canvas */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 space-y-8">
            {groupedResources.global.length > 0 && (
                <GCPZoneGroup
                    title="Global"
                    resources={groupedResources.global}
                    selectedResourceId={selectedResource?.id}
                    onResourceSelect={onResourceSelect}
                />
            )}
            {Object.entries(groupedResources.regional).map(([region, zones]) => (
                <div key={region} className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-300 capitalize border-b border-gray-700 pb-2">{region}</h3>
                    {Object.entries(zones).map(([zone, resources]) => (
                         <GCPZoneGroup
                            key={zone}
                            title={zone}
                            resources={resources}
                            selectedResourceId={selectedResource?.id}
                            onResourceSelect={onResourceSelect}
                        />
                    ))}
                </div>
            ))}
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

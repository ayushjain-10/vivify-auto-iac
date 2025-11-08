import React from 'react';
import { GCPService } from '../../types/gcp';
import GCPResourceCard from './GCPResourceCard';
import { ChevronDownIcon } from '../icons/ChevronDownIcon';

interface GCPZoneGroupProps {
  title: string;
  resources: GCPService[];
  selectedResourceId: string | null | undefined;
  onResourceSelect: (resource: GCPService) => void;
}

const GCPZoneGroup: React.FC<GCPZoneGroupProps> = ({ title, resources, selectedResourceId, onResourceSelect }) => {
  return (
    <section>
      <div className="flex items-center mb-4">
        <h4 className="text-md font-medium text-gray-400 capitalize">{title}</h4>
        <span className="ml-2 text-xs font-mono bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">{resources.length}</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
        {resources.map((resource) => (
          <GCPResourceCard
            key={resource.id}
            resource={resource}
            isSelected={selectedResourceId === resource.id}
            onSelect={() => onResourceSelect(resource)}
          />
        ))}
      </div>
    </section>
  );
};

export default GCPZoneGroup;

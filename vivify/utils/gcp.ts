import React from 'react';
import { GCPService } from '../types/gcp';

// GCP Icons
import { ServerIcon } from '../components/icons/gcp/ServerIcon';
import { DatabaseIcon } from '../components/icons/gcp/DatabaseIcon';
import { HardDriveIcon } from '../components/icons/gcp/HardDriveIcon';
import { LayersIcon } from '../components/icons/gcp/LayersIcon';
import { NetworkIcon } from '../components/icons/gcp/NetworkIcon';
import { ShieldIcon } from '../components/icons/gcp/ShieldIcon';
import { ZapIcon } from '../components/icons/gcp/ZapIcon';
import { MapPinIcon } from '../components/icons/gcp/MapPinIcon';
import { ActivityIcon } from '../components/icons/gcp/ActivityIcon';
import { MessageSquareIcon } from '../components/icons/gcp/MessageSquareIcon';
import { MessageCircleIcon } from '../components/icons/gcp/MessageCircleIcon';
import { BarChart3Icon } from '../components/icons/gcp/BarChart3Icon';
import { ContainerIcon } from '../components/icons/gcp/ContainerIcon';
import { CloudIcon } from '../components/icons/gcp/CloudIcon';


export const getGCPIcon = (resourceType: string): React.ComponentType<{ className?: string }> => {
  const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
    'google_compute_instance': ServerIcon,
    'google_storage_bucket': HardDriveIcon, // More representative of storage
    'google_container_cluster': LayersIcon,
    'google_compute_network': NetworkIcon,
    'google_compute_firewall': ShieldIcon,
    'google_cloud_functions_function': ZapIcon,
    'google_sql_database_instance': DatabaseIcon,
    'google_compute_disk': HardDriveIcon,
    'google_compute_address': MapPinIcon,
    'google_compute_forwarding_rule': ActivityIcon,
    'google_pubsub_topic': MessageSquareIcon,
    'google_pubsub_subscription': MessageCircleIcon,
    'google_bigquery_dataset': BarChart3Icon,
    'google_cloud_run_service': ContainerIcon,
    'google_app_engine_application': CloudIcon,
  };
  return iconMap[resourceType] || CloudIcon;
};


export function getGCPConsoleLink(resource: GCPService): string | null {
  const baseUrl = 'https://console.cloud.google.com';
  const project = resource.project;
  
  const linkMap: Record<string, (r: GCPService) => string> = {
    'google_compute_instance': (r) => 
      `${baseUrl}/compute/instancesDetail/zones/${r.zone}/instances/${r.name}?project=${project}`,
    'google_storage_bucket': (r) =>
      `${baseUrl}/storage/browser/${r.name}?project=${project}`,
    'google_container_cluster': (r) =>
      `${baseUrl}/kubernetes/clusters/details/${r.zone || r.region}/${r.name}?project=${project}`,
    'google_compute_network': (r) =>
      `${baseUrl}/networking/networks/details/${r.name}?project=${project}`,
    'google_cloud_functions_function': (r) =>
      `${baseUrl}/functions/details/${r.region}/${r.name}?project=${project}`,
     'google_sql_database_instance': (r) =>
      `${baseUrl}/sql/instances/${r.name}/overview?project=${project}`,
     'google_compute_firewall': (r) =>
      `${baseUrl}/networking/firewalls/details/${r.name}?project=${project}`,
  };
  
  const linkBuilder = linkMap[resource.type];
  return linkBuilder ? linkBuilder(resource) : null;
}

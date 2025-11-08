export interface GCPService {
  id: string;
  name: string;
  type: string;  // 'google_compute_instance', 'google_storage_bucket', etc.
  status: 'running' | 'stopped' | 'deploying' | 'error';
  region: string;
  zone?: string;
  project: string;
  description?: string;
  configuration?: Record<string, any>;
  costEstimate?: {
    monthly: number;
    breakdown: string;
    currency?: string;
  };
  metrics?: {
    cpu?: number;
    memory?: number;
    networkIn?: number;  // bytes
    networkOut?: number;  // bytes
    diskReadOps?: number;
    diskWriteOps?: number;
    requests?: number;
    errors?: number;
    latency?: number;  // milliseconds
    errorRate?: number;  // percentage
  };
  labels?: Record<string, string>;
  connections?: string[];  // IDs of connected resources
  healthStatus: 'healthy' | 'warning' | 'critical' | 'unknown';
  selfLink?: string;  // GCP selfLink for console navigation
  createdAt?: string;
  lastUpdated?: string;
}

export interface GCPConnection {
  id: string;
  from: string;  // Resource ID
  to: string;    // Resource ID
  type: 'network' | 'storage' | 'data' | 'api' | 'trigger';
  protocol?: string;
  port?: number;
  description?: string;
  direction?: 'inbound' | 'outbound' | 'bidirectional';
}

export interface GCPApplicationStack {
  id: string;
  name: string;
  description: string;
  services: string[]; // Array of service IDs
  primaryService?: string;  // Main service ID
  labels: Record<string, string>;
  totalCost: number;
  healthStatus: 'healthy' | 'warning' | 'critical';
  vpc?: string;  // VPC network name
  subnets?: string[];  // Subnetwork names
  region?: string;
  zones?: string[];
}

export interface GCPArchitecture {
  project: string;
  lastRefresh: string;
  regions: string[];
  zones: Record<string, GCPService[]>;  // Zone name -> resources (This is redundant if we have resources array)
  resources: GCPService[];
  connections: GCPConnection[];
  totalCost: number;
  costBreakdown: Record<string, number>;  // Stack name -> cost
  applicationStacks: GCPApplicationStack[];
  hasGCPAccess: boolean;
}

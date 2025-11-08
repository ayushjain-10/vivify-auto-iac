/**
 * GCP API Service
 * Handles all GCP-related API calls
 */

import { apiClient } from './api';
import { GCPArchitecture } from '../types/gcp';

export interface ValidationResult {
  valid: boolean;
  projectId?: string;
  error?: string;
}

export interface DiscoveryRequest {
  credentials: any;
  project?: string;
  regions?: string[];
}

export const gcpApi = {
  /**
   * Validate GCP service account credentials
   */
  async validateCredentials(credentials: any): Promise<ValidationResult> {
    return apiClient.post<ValidationResult>('/api/gcp/validate-credentials', {
      credentials,
    });
  },

  /**
   * Discover GCP resources
   */
  async discoverResources(
    credentials: any,
    project?: string,
    regions?: string[]
  ): Promise<GCPArchitecture> {
    const request: DiscoveryRequest = {
      credentials,
      ...(project && { project }),
      ...(regions && { regions }),
    };

    return apiClient.post<GCPArchitecture>('/api/gcp/discover', request);
  },

  /**
   * Get cached architecture for a project
   */
  async getArchitecture(project: string): Promise<GCPArchitecture> {
    return apiClient.get<GCPArchitecture>(`/api/gcp/architecture/${project}`);
  },

  /**
   * Clear cached architecture
   */
  async clearArchitectureCache(project: string): Promise<void> {
    return apiClient.delete(`/api/gcp/architecture/${project}`);
  },
};

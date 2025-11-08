"""
GCP Resource Discovery Service
Discovers GCP resources and builds architecture model
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from google.cloud import compute_v1, storage, container_v1
from google.oauth2 import service_account
from api.models.gcp import (
    GCPService, GCPArchitecture, GCPConnection,
    GCPApplicationStack, GCPCostEstimate, GCPServiceMetrics
)


class GCPDiscoveryService:
    """Service for discovering GCP resources"""
    
    def __init__(self, credentials: service_account.Credentials, project_id: str):
        self.credentials = credentials
        self.project_id = project_id
        self.resources: List[GCPService] = []
        self.connections: List[GCPConnection] = []
    
    def discover_all(self, regions: Optional[List[str]] = None) -> GCPArchitecture:
        """
        Discover all GCP resources
        
        Args:
            regions: List of regions to scan (None = all regions)
            
        Returns:
            Complete GCP architecture
        """
        print(f"🔍 Starting discovery for project: {self.project_id}")
        
        # Discover resources by type
        self._discover_compute_instances(regions)
        self._discover_storage_buckets()
        self._discover_gke_clusters(regions)
        self._discover_networks()
        self._discover_firewalls()
        
        # Detect relationships
        self._detect_relationships()
        
        # Group by zones
        zones_map = self._group_by_zones()
        
        # Detect application stacks
        stacks = self._detect_application_stacks()
        
        # Calculate costs
        total_cost = sum(
            r.cost_estimate.monthly if r.cost_estimate else 0.0
            for r in self.resources
        )
        
        cost_breakdown = {}
        for stack in stacks:
            cost_breakdown[stack.name] = stack.total_cost
        
        # If no resources found, add a note
        if len(self.resources) == 0:
            print("\n⚠️  No resources discovered. This could be because:")
            print("   - The project has no resources yet")
            print("   - Required APIs are not enabled")
            print("   - Service account lacks permissions")
            print("   - Resources are in regions not scanned")
        
        # Get unique regions
        regions_list = list(set(r.region for r in self.resources if r.region != "global"))
        
        print(f"✅ Discovery complete: {len(self.resources)} resources found")
        
        return GCPArchitecture(
            project=self.project_id,
            lastRefresh=datetime.utcnow().isoformat() + "Z",
            regions=sorted(regions_list),
            zones=zones_map,
            resources=self.resources,
            connections=self.connections,
            totalCost=total_cost,
            costBreakdown=cost_breakdown,
            applicationStacks=stacks,
            hasGCPAccess=True
        )
    
    def _discover_compute_instances(self, regions: Optional[List[str]] = None):
        """Discover Compute Engine VM instances"""
        try:
            print("  📦 Discovering Compute Engine instances...")
            client = compute_v1.InstancesClient(credentials=self.credentials)
            
            # Get all zones
            zones_client = compute_v1.ZonesClient(credentials=self.credentials)
            zones_list = zones_client.list(project=self.project_id)
            
            for zone in zones_list:
                zone_name = zone.name
                
                # Filter by region if specified
                if regions and not any(zone_name.startswith(r) for r in regions):
                    continue
                
                try:
                    instances = client.list(project=self.project_id, zone=zone_name)
                    
                    for instance in instances:
                        resource = self._instance_to_resource(instance, zone_name)
                        self.resources.append(resource)
                        
                except Exception as e:
                    print(f"    ⚠️  Error listing instances in {zone_name}: {e}")
                    continue
            
            print(f"    ✓ Found {sum(1 for r in self.resources if r.type == 'google_compute_instance')} instances")
            
        except Exception as e:
            print(f"    ❌ Error discovering compute instances: {e}")
    
    def _discover_storage_buckets(self):
        """Discover Cloud Storage buckets"""
        try:
            print("  🪣 Discovering Cloud Storage buckets...")
            client = storage.Client(credentials=self.credentials, project=self.project_id)
            
            buckets = client.list_buckets()
            
            for bucket in buckets:
                resource = self._bucket_to_resource(bucket)
                self.resources.append(resource)
            
            print(f"    ✓ Found {sum(1 for r in self.resources if r.type == 'google_storage_bucket')} buckets")
            
        except Exception as e:
            print(f"    ❌ Error discovering storage buckets: {e}")
    
    def _discover_gke_clusters(self, regions: Optional[List[str]] = None):
        """Discover GKE clusters"""
        try:
            print("  ☸️  Discovering GKE clusters...")
            client = container_v1.ClusterManagerClient(credentials=self.credentials)
            
            parent = f"projects/{self.project_id}/locations/-"
            clusters = client.list_clusters(parent=parent)
            
            for cluster in clusters.clusters:
                # Filter by region if specified
                if regions and not any(cluster.location.startswith(r) for r in regions):
                    continue
                
                resource = self._cluster_to_resource(cluster)
                self.resources.append(resource)
            
            print(f"    ✓ Found {sum(1 for r in self.resources if r.type == 'google_container_cluster')} clusters")
            
        except Exception as e:
            print(f"    ❌ Error discovering GKE clusters: {e}")
    
    def _discover_networks(self):
        """Discover VPC networks"""
        try:
            print("  🌐 Discovering VPC networks...")
            client = compute_v1.NetworksClient(credentials=self.credentials)
            
            networks = client.list(project=self.project_id)
            
            for network in networks:
                resource = self._network_to_resource(network)
                self.resources.append(resource)
            
            print(f"    ✓ Found {sum(1 for r in self.resources if r.type == 'google_compute_network')} networks")
            
        except Exception as e:
            print(f"    ❌ Error discovering networks: {e}")
    
    def _discover_firewalls(self):
        """Discover firewall rules"""
        try:
            print("  🛡️  Discovering firewall rules...")
            client = compute_v1.FirewallsClient(credentials=self.credentials)
            
            firewalls = client.list(project=self.project_id)
            
            for firewall in firewalls:
                resource = self._firewall_to_resource(firewall)
                self.resources.append(resource)
            
            print(f"    ✓ Found {sum(1 for r in self.resources if r.type == 'google_compute_firewall')} firewall rules")
            
        except Exception as e:
            print(f"    ❌ Error discovering firewalls: {e}")
    
    def _instance_to_resource(self, instance: Any, zone: str) -> GCPService:
        """Convert Compute Engine instance to GCPService"""
        region = "-".join(zone.split("-")[:-1])
        
        # Determine status
        status_map = {
            "RUNNING": "running",
            "TERMINATED": "stopped",
            "STOPPING": "stopped",
            "PROVISIONING": "deploying",
            "STAGING": "deploying",
        }
        status = status_map.get(instance.status, "error")
        
        # Extract machine type
        machine_type = instance.machine_type.split("/")[-1] if instance.machine_type else "unknown"
        
        # Mock metrics (in real implementation, fetch from Cloud Monitoring)
        metrics = GCPServiceMetrics(
            cpu=23.0 if status == "running" else 0.0,
            memory=45.0 if status == "running" else 0.0,
        )
        
        # Mock cost estimate (in real implementation, use Billing API)
        cost = GCPCostEstimate(
            monthly=67.32,
            breakdown=f"{machine_type} on-demand"
        )
        
        return GCPService(
            id=f"instance-{instance.id}",
            name=instance.name,
            type="google_compute_instance",
            status=status,
            region=region,
            zone=zone,
            project=self.project_id,
            healthStatus="healthy" if status == "running" else "unknown",
            selfLink=instance.self_link,
            costEstimate=cost,
            metrics=metrics if status == "running" else None,
            configuration={
                "machineType": machine_type,
                "networkInterfaces": [
                    {"network": ni.network} for ni in instance.network_interfaces
                ] if instance.network_interfaces else []
            },
            labels=dict(instance.labels) if instance.labels else {},
            createdAt=instance.creation_timestamp,
        )
    
    def _bucket_to_resource(self, bucket: Any) -> GCPService:
        """Convert Storage bucket to GCPService"""
        cost = GCPCostEstimate(
            monthly=12.80,
            breakdown="Standard Storage"
        )
        
        return GCPService(
            id=f"bucket-{bucket.name}",
            name=bucket.name,
            type="google_storage_bucket",
            status="running",
            region=bucket.location.lower() if bucket.location else "us",
            project=self.project_id,
            healthStatus="healthy",
            selfLink=bucket.self_link,
            costEstimate=cost,
            configuration={
                "location": bucket.location,
                "storageClass": bucket.storage_class,
            },
            labels=dict(bucket.labels) if bucket.labels else {},
            createdAt=bucket.time_created.isoformat() if bucket.time_created else None,
        )
    
    def _cluster_to_resource(self, cluster: Any) -> GCPService:
        """Convert GKE cluster to GCPService"""
        # Determine if it's zonal or regional
        location = cluster.location
        is_regional = location.count("-") == 1
        
        region = location if is_regional else "-".join(location.split("-")[:-1])
        zone = None if is_regional else location
        
        status_map = {
            "RUNNING": "running",
            "PROVISIONING": "deploying",
            "STOPPING": "stopped",
            "ERROR": "error",
        }
        status = status_map.get(str(cluster.status), "unknown")
        
        metrics = GCPServiceMetrics(
            cpu=60.0 if status == "running" else 0.0,
            memory=75.0 if status == "running" else 0.0,
        )
        
        cost = GCPCostEstimate(
            monthly=250.00,
            breakdown="Cluster Mgmt Fee + Nodes"
        )
        
        return GCPService(
            id=f"gke-{cluster.name}",
            name=cluster.name,
            type="google_container_cluster",
            status=status,
            region=region,
            zone=zone,
            project=self.project_id,
            healthStatus="healthy" if status == "running" else "warning",
            selfLink=cluster.self_link,
            costEstimate=cost,
            metrics=metrics if status == "running" else None,
            configuration={
                "nodeCount": cluster.current_node_count,
                "location": location,
            },
            labels=dict(cluster.resource_labels) if cluster.resource_labels else {},
        )
    
    def _network_to_resource(self, network: Any) -> GCPService:
        """Convert VPC network to GCPService"""
        return GCPService(
            id=f"network-{network.name}",
            name=network.name,
            type="google_compute_network",
            status="running",
            region="global",
            project=self.project_id,
            healthStatus="healthy",
            selfLink=network.self_link,
            configuration={
                "autoCreateSubnetworks": network.auto_create_subnetworks,
            },
            labels={},
        )
    
    def _firewall_to_resource(self, firewall: Any) -> GCPService:
        """Convert firewall rule to GCPService"""
        return GCPService(
            id=f"firewall-{firewall.name}",
            name=firewall.name,
            type="google_compute_firewall",
            status="running",
            region="global",
            project=self.project_id,
            healthStatus="healthy",
            selfLink=firewall.self_link,
            configuration={
                "network": firewall.network,
                "direction": firewall.direction,
                "priority": firewall.priority,
            },
            labels={},
        )
    
    def _detect_relationships(self):
        """Detect connections between resources"""
        print("  🔗 Detecting resource relationships...")
        
        # Create lookup maps
        instances = [r for r in self.resources if r.type == "google_compute_instance"]
        networks = [r for r in self.resources if r.type == "google_compute_network"]
        
        # Connect instances to networks
        for instance in instances:
            if instance.configuration and "networkInterfaces" in instance.configuration:
                for ni in instance.configuration["networkInterfaces"]:
                    network_url = ni.get("network", "")
                    # Find matching network
                    for network in networks:
                        if network.name in network_url or network.self_link == network_url:
                            connection = GCPConnection(
                                id=f"conn-{uuid.uuid4().hex[:8]}",
                                **{"from": instance.id, "to": network.id},
                                type="network"
                            )
                            self.connections.append(connection)
        
        print(f"    ✓ Found {len(self.connections)} connections")
    
    def _group_by_zones(self) -> Dict[str, List[GCPService]]:
        """Group resources by zone"""
        zones_map: Dict[str, List[GCPService]] = {}
        
        for resource in self.resources:
            if resource.zone:
                if resource.zone not in zones_map:
                    zones_map[resource.zone] = []
                zones_map[resource.zone].append(resource)
        
        return zones_map
    
    def _detect_application_stacks(self) -> List[GCPApplicationStack]:
        """Detect application stacks based on labels"""
        stacks: List[GCPApplicationStack] = []
        
        # Group by 'app' or 'environment' labels
        app_groups: Dict[str, List[GCPService]] = {}
        
        for resource in self.resources:
            if resource.labels:
                app_name = resource.labels.get("app") or resource.labels.get("application")
                if app_name:
                    if app_name not in app_groups:
                        app_groups[app_name] = []
                    app_groups[app_name].append(resource)
        
        # Create stacks
        for app_name, resources in app_groups.items():
            total_cost = sum(
                r.cost_estimate.monthly if r.cost_estimate else 0.0
                for r in resources
            )
            
            # Determine health status
            health_statuses = [r.health_status for r in resources]
            if "critical" in health_statuses:
                health = "critical"
            elif "warning" in health_statuses:
                health = "warning"
            else:
                health = "healthy"
            
            stack = GCPApplicationStack(
                id=f"stack-{app_name}",
                name=f"{app_name.title()} Stack",
                description=f"Resources for {app_name} application",
                services=[r.id for r in resources],
                labels={"app": app_name},
                totalCost=total_cost,
                healthStatus=health
            )
            stacks.append(stack)
        
        return stacks

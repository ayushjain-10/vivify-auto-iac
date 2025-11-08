"""
Pydantic models for GCP resources
Matches the TypeScript interfaces in the frontend
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Literal
from datetime import datetime


class GCPServiceMetrics(BaseModel):
    """Resource metrics"""
    cpu: Optional[float] = None
    memory: Optional[float] = None
    network_in: Optional[int] = Field(None, alias="networkIn")
    network_out: Optional[int] = Field(None, alias="networkOut")
    disk_read_ops: Optional[int] = Field(None, alias="diskReadOps")
    disk_write_ops: Optional[int] = Field(None, alias="diskWriteOps")
    requests: Optional[int] = None
    errors: Optional[int] = None
    latency: Optional[float] = None
    error_rate: Optional[float] = Field(None, alias="errorRate")

    class Config:
        populate_by_name = True


class GCPCostEstimate(BaseModel):
    """Cost estimate for a resource"""
    monthly: float
    breakdown: str
    currency: str = "USD"


class GCPService(BaseModel):
    """GCP Resource/Service"""
    id: str
    name: str
    type: str
    status: Literal["running", "stopped", "deploying", "error"]
    region: str
    zone: Optional[str] = None
    project: str
    description: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[GCPCostEstimate] = Field(None, alias="costEstimate")
    metrics: Optional[GCPServiceMetrics] = None
    labels: Optional[Dict[str, str]] = None
    connections: Optional[List[str]] = None
    health_status: Literal["healthy", "warning", "critical", "unknown"] = Field(
        "unknown", alias="healthStatus"
    )
    self_link: Optional[str] = Field(None, alias="selfLink")
    created_at: Optional[str] = Field(None, alias="createdAt")
    last_updated: Optional[str] = Field(None, alias="lastUpdated")

    class Config:
        populate_by_name = True


class GCPConnection(BaseModel):
    """Connection between GCP resources"""
    id: str
    from_resource: str = Field(..., alias="from")
    to_resource: str = Field(..., alias="to")
    type: Literal["network", "storage", "data", "api", "trigger"]
    protocol: Optional[str] = None
    port: Optional[int] = None
    description: Optional[str] = None
    direction: Optional[Literal["inbound", "outbound", "bidirectional"]] = None

    class Config:
        populate_by_name = True


class GCPApplicationStack(BaseModel):
    """Application stack grouping"""
    id: str
    name: str
    description: str
    services: List[str]  # List of service IDs
    primary_service: Optional[str] = Field(None, alias="primaryService")
    labels: Dict[str, str]
    total_cost: float = Field(..., alias="totalCost")
    health_status: Literal["healthy", "warning", "critical"] = Field(..., alias="healthStatus")
    vpc: Optional[str] = None
    subnets: Optional[List[str]] = None
    region: Optional[str] = None
    zones: Optional[List[str]] = None

    class Config:
        populate_by_name = True


class GCPArchitecture(BaseModel):
    """Complete GCP architecture"""
    project: str
    last_refresh: str = Field(..., alias="lastRefresh")
    regions: List[str]
    zones: Dict[str, List[GCPService]]
    resources: List[GCPService]
    connections: List[GCPConnection]
    total_cost: float = Field(..., alias="totalCost")
    cost_breakdown: Dict[str, float] = Field(..., alias="costBreakdown")
    application_stacks: List[GCPApplicationStack] = Field(..., alias="applicationStacks")
    has_gcp_access: bool = Field(True, alias="hasGCPAccess")

    class Config:
        populate_by_name = True


class DiscoveryRequest(BaseModel):
    """Request to discover GCP resources"""
    credentials: Dict[str, Any]
    project: Optional[str] = None  # If not provided, extract from credentials
    regions: Optional[List[str]] = None  # If not provided, discover all


class CredentialsValidationRequest(BaseModel):
    """Request to validate GCP credentials"""
    credentials: Dict[str, Any]


class CredentialsValidationResponse(BaseModel):
    """Response from credentials validation"""
    valid: bool
    project_id: Optional[str] = Field(None, alias="projectId")
    error: Optional[str] = None

    class Config:
        populate_by_name = True

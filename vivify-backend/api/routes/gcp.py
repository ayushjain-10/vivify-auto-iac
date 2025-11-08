"""
GCP API Routes
Endpoints for GCP resource discovery and management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models.gcp import (
    DiscoveryRequest,
    CredentialsValidationRequest,
    CredentialsValidationResponse,
    GCPArchitecture
)
from utils.auth import validate_service_account_credentials, get_credentials_object
from services.gcp_discovery import GCPDiscoveryService
from typing import Dict
import traceback

router = APIRouter()

# In-memory cache for discovered architectures
# In production, use Redis or a database
architecture_cache: Dict[str, GCPArchitecture] = {}


@router.post("/validate-credentials", response_model=CredentialsValidationResponse)
async def validate_credentials(request: CredentialsValidationRequest):
    """
    Validate GCP service account credentials
    """
    try:
        is_valid, project_id, error = validate_service_account_credentials(
            request.credentials
        )
        
        if is_valid:
            return CredentialsValidationResponse(
                valid=True,
                projectId=project_id
            )
        else:
            return CredentialsValidationResponse(
                valid=False,
                error=error
            )
    
    except Exception as e:
        return CredentialsValidationResponse(
            valid=False,
            error=f"Validation error: {str(e)}"
        )


@router.post("/discover", response_model=GCPArchitecture)
async def discover_resources(request: DiscoveryRequest):
    """
    Discover GCP resources for a project
    
    This endpoint:
    1. Validates credentials
    2. Discovers all resources in the project
    3. Enriches with metrics and costs
    4. Detects relationships
    5. Returns complete architecture
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 Starting GCP Discovery")
        print(f"{'='*60}")
        
        # Validate credentials first
        is_valid, project_id, error = validate_service_account_credentials(
            request.credentials
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid credentials: {error}"
            )
        
        # Use project from request or from credentials
        project = request.project or project_id
        
        print(f"📋 Project: {project}")
        print(f"🌍 Regions: {request.regions or 'All'}")
        
        # Get credentials object
        creds = get_credentials_object(request.credentials)
        
        # Create discovery service
        discovery_service = GCPDiscoveryService(creds, project)
        
        # Discover resources
        architecture = discovery_service.discover_all(request.regions)
        
        # Cache the result
        architecture_cache[project] = architecture
        
        print(f"\n{'='*60}")
        print(f"✅ Discovery Complete!")
        print(f"   Resources: {len(architecture.resources)}")
        print(f"   Connections: {len(architecture.connections)}")
        print(f"   Total Cost: ${architecture.total_cost:.2f}/month")
        print(f"{'='*60}\n")
        
        return architecture
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Discovery Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Discovery failed: {str(e)}"
        )


@router.get("/architecture/{project}", response_model=GCPArchitecture)
async def get_architecture(project: str):
    """
    Get cached architecture for a project
    """
    if project not in architecture_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No architecture found for project: {project}. Run discovery first."
        )
    
    return architecture_cache[project]


@router.delete("/architecture/{project}")
async def clear_architecture_cache(project: str):
    """
    Clear cached architecture for a project
    """
    if project in architecture_cache:
        del architecture_cache[project]
        return {"message": f"Cache cleared for project: {project}"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No cache found for project: {project}"
        )

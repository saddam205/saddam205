"""
response.py
Part of the app/schemas module.
Common response schemas for API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')


class StatusEnum(str, Enum):
    """Response status"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""
    status: StatusEnum = Field(StatusEnum.SUCCESS, description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    @classmethod
    def success(cls, data: T = None, message: str = "Success") -> "APIResponse":
        """Create success response"""
        return cls(status=StatusEnum.SUCCESS, message=message, data=data)
    
    @classmethod
    def error(cls, message: str, data: T = None) -> "APIResponse":
        """Create error response"""
        return cls(status=StatusEnum.ERROR, message=message, data=data)
    
    @classmethod
    def warning(cls, message: str, data: T = None) -> "APIResponse":
        """Create warning response"""
        return cls(status=StatusEnum.WARNING, message=message, data=data)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(100, description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        if 'total' in values and 'page_size' in values:
            return (values['total'] + values['page_size'] - 1) // values['page_size']
        return v
    
    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        if 'page' in values and 'total_pages' in values:
            return values['page'] < values['total_pages']
        return v
    
    @validator('has_previous', always=True)
    def calculate_has_previous(cls, v, values):
        if 'page' in values:
            return values['page'] > 1
        return v


class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response format"""
    status: StatusEnum = Field(StatusEnum.ERROR)
    message: str = Field(..., description="Error message")
    errors: List[ErrorDetail] = Field(default_factory=list, description="Detailed errors")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request ID")
    path: Optional[str] = Field(None, description="Request path")
    method: Optional[str] = Field(None, description="HTTP method")


class HealthCheck(BaseModel):
    """Individual health check result"""
    name: str
    status: str = Field(..., description="healthy, degraded, unhealthy")
    message: str = Field("", description="Status message")
    last_check: datetime
    response_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """System health response"""
    status: str = Field(..., description="overall system status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime")
    checks: List[HealthCheck] = Field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


class MetricsPoint(BaseModel):
    """Single metrics data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Metrics response"""
    metric_name: str
    data_points: List[MetricsPoint]
    unit: str = Field("", description="Unit of measurement")
    description: str = Field("", description="Metric description")
    statistics: Optional[Dict[str, float]] = Field(None, description="Statistical summary")


class StatusResponse(BaseModel):
    """Simple status response"""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class VersionResponse(BaseModel):
    """Version information response"""
    version: str
    build_date: Optional[str] = None
    commit_hash: Optional[str] = None
    environment: str = Field("production", description="Environment (development, staging, production)")


class SystemInfoResponse(BaseModel):
    """System information response"""
    cpu_cores: int
    memory_total_gb: float
    memory_available_gb: float
    disk_total_gb: float
    disk_free_gb: float
    python_version: str
    platform: str
    hostname: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RateLimitResponse(BaseModel):
    """Rate limit information"""
    limit: int = Field(..., description="Requests per minute limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_at: datetime = Field(..., description="When the limit resets")
    retry_after_seconds: Optional[int] = Field(None, description="Seconds to wait before retry")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response (for 422 errors)"""
    pass


class AuthenticationResponse(BaseModel):
    """Authentication response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiry in seconds")
    refresh_token: Optional[str] = None


class BatchOperationResponse(BaseModel):
    """Batch operation response"""
    total: int = Field(..., description="Total operations")
    successful: int = Field(..., description="Successful operations")
    failed: int = Field(..., description="Failed operations")
    errors: List[ErrorDetail] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
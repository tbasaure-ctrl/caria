"""CORS test endpoint for debugging CORS issues."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/cors-test")
@router.options("/api/cors-test")
async def cors_test(request: Request):
    """
    Test endpoint to diagnose CORS issues.
    Returns information about the request origin and CORS headers.
    """
    origin = request.headers.get("origin", "No Origin header")
    method = request.method
    
    # Get all CORS-related headers from request
    cors_headers = {
        "origin": origin,
        "access-control-request-method": request.headers.get("access-control-request-method"),
        "access-control-request-headers": request.headers.get("access-control-request-headers"),
    }
    
    # Response data
    response_data = {
        "status": "ok",
        "method": method,
        "request_headers": dict(cors_headers),
        "message": "CORS test endpoint - check response headers for CORS configuration",
    }
    
    # Create response
    response = JSONResponse(content=response_data)
    
    # Add CORS headers manually to ensure they're present
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
    
    # For OPTIONS requests, return 200 immediately
    if method == "OPTIONS":
        response.status_code = 200
    
    return response


# Caddy Routing and Path Rewriting

This document explains how Caddy handles routing and path rewriting in the AgentKS stack.

## Overview

Caddy serves as the reverse proxy for all external traffic. It provides:
- TLS termination (HTTPS)
- Authentication via Authentik forward_auth
- Path-based routing to backend services
- **Path prefix stripping** for clean backend APIs

## Path Rewriting with `handle_path`

### The Problem

Backend services often expose their APIs at the root path (`/`):
- RAG Injector exposes: `/health`, `/upload`, `/documents`
- Admin UI exposes: `/admin/api/health`, `/admin/dashboard`

But when multiple services are behind one domain, you need prefixes:
- Public: `https://domain.com/rag/health`
- Public: `https://domain.com/admin/api/health`

Without path rewriting, backends would need to handle these prefixes, which:
- Complicates backend code
- Breaks when service is accessed directly (e.g., for testing)
- Requires environment-specific configuration

### The Solution: `handle_path`

Caddy's `handle_path` directive automatically strips the matched prefix:

```caddyfile
handle_path /rag* {
  reverse_proxy http://backend:4002
}
```

**How it works:**
1. Client sends: `GET https://domain.com/rag/health`
2. Caddy matches: `/rag*` pattern
3. Caddy strips: `/rag` prefix
4. Backend receives: `GET /health`
5. Backend responds: `200 OK {"status": "healthy"}`
6. Caddy forwards response to client

## Current Routing Configuration

### Main Domain Routes

From `Caddyfile`:

```caddyfile
{$DOMAIN} {

  # Authentik outpost endpoints (required by forward_auth)
  reverse_proxy /outpost.goauthentik.io/* http://authentik_proxy:9000

  # /webui protected by forward_auth
  handle_path /webui* {
    forward_auth http://authentik_proxy:9000 {
      uri /outpost.goauthentik.io/auth/caddy
      copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
    }
    reverse_proxy http://openwebui:8080
  }

  # /admin protected by forward_auth (includes /admin/api/*)
  handle_path /admin* {
    forward_auth http://authentik_proxy:9000 {
      uri /outpost.goauthentik.io/auth/caddy
      copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
    }
    reverse_proxy http://backend:4000
  }

  # /web protected by forward_auth (includes /web*)
  handle_path /web* {
    forward_auth http://authentik_proxy:9000 {
      uri /outpost.goauthentik.io/auth/caddy
      copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
    }
    reverse_proxy http://backend:8000
  }

  # /rag protected by forward_auth, strips /rag prefix before forwarding
  handle_path /rag* {
    forward_auth http://authentik_proxy:9000 {
      uri /outpost.goauthentik.io/auth/caddy
      copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
    }
    reverse_proxy http://backend:4002
  }

  respond 404
}
```

### Path Mapping Table

| Client Request | Matched Route | Backend Receives | Backend Service |
|----------------|---------------|------------------|-----------------|
| `GET /webui` | `/webui*` | `GET /` | OpenWebUI:8080 |
| `GET /webui/chat` | `/webui*` | `GET /chat` | OpenWebUI:8080 |
| `POST /admin/api/health` | `/admin*` | `POST /api/health` | Backend:4000 |
| `GET /admin/dashboard` | `/admin*` | `GET /dashboard` | Backend:4000 |
| `GET /web/api/status` | `/web*` | `GET /api/status` | Backend:8000 |
| `POST /rag/upload` | `/rag*` | `POST /upload` | Backend:4002 (RAG Injector) |
| `GET /rag/health` | `/rag*` | `GET /health` | Backend:4002 (RAG Injector) |
| `GET /rag/documents` | `/rag*` | `GET /documents` | Backend:4002 (RAG Injector) |

## Authentication Flow with Path Rewriting

Every route uses `forward_auth` to check authentication before proxying:

```
┌─────────┐
│ Client  │
└────┬────┘
     │ GET /rag/health
     ↓
┌─────────┐
│ Caddy   │ Matches /rag*
└────┬────┘
     │ forward_auth
     ↓
┌────────────┐
│ Authentik  │ Validates session
│ Proxy      │
└────┬───────┘
     │ 200 OK + headers
     ↓
┌─────────┐
│ Caddy   │ Strips /rag prefix
└────┬────┘ Adds X-Authentik-* headers
     │ GET /health (with headers)
     ↓
┌────────────┐
│ Backend    │ RAG Injector (port 4002)
│ 4002       │
└────┬───────┘
     │ 200 OK {"status": "healthy"}
     ↓
┌─────────┐
│ Caddy   │ Forwards response
└────┬────┘
     │ 200 OK
     ↓
┌─────────┐
│ Client  │
└─────────┘
```

## Benefits of Path Rewriting

### 1. Clean Backend APIs

Backends don't need to know about routing prefixes:

**Without path rewriting:**
```python
# Backend must handle prefix
@app.get("/rag/health")
async def health():
    return {"status": "healthy"}
```

**With path rewriting:**
```python
# Backend uses clean root paths
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 2. Service Portability

Services can be:
- Tested locally without prefix: `curl http://localhost:4002/health`
- Used in different contexts with different prefixes
- Moved between routes without code changes

### 3. Environment Independence

No environment-specific configuration needed:

```python
# ❌ Bad: Environment-specific
BASE_PATH = os.getenv("BASE_PATH", "/rag")
@app.get(f"{BASE_PATH}/health")

# ✅ Good: Environment-agnostic
@app.get("/health")
```

### 4. Simplified Deployment

Docker compose can expose services on any port internally:
```yaml
backend:
  ports:
    - "4002:4002"  # RAG Injector
```

Caddy handles the external routing:
```caddyfile
handle_path /rag* {
  reverse_proxy http://backend:4002
}
```

## Alternative Approaches

### Using `handle` Without Path Stripping

If you want to keep the prefix in the backend request:

```caddyfile
handle /rag/* {
  reverse_proxy http://backend:4002
}
```

**Path mapping:**
- Client: `GET /rag/health`
- Backend receives: `GET /rag/health` (prefix preserved)

**When to use:**
- Backend needs to know its public path
- Multiple routes go to same backend with different prefixes
- Backend handles routing internally

### Using `route` with `rewrite`

For more complex transformations:

```caddyfile
route /api/v2/* {
  uri strip_prefix /api/v2
  reverse_proxy http://backend:4000
}
```

**Path mapping:**
- Client: `GET /api/v2/users/123`
- After `strip_prefix`: `GET /users/123`
- Backend receives: `GET /users/123`

### Using `redir` for Path Changes

For permanent redirects:

```caddyfile
redir /old-path /new-path permanent
```

## Common Patterns

### Pattern 1: Multiple Services, Same Prefix

Route different paths to different services:

```caddyfile
handle /api/rag/* {
  reverse_proxy http://backend:4002
}

handle /api/tools/* {
  reverse_proxy http://backend:5000
}

handle /api/agent/* {
  reverse_proxy http://backend:4000
}
```

### Pattern 2: Versioned APIs

Support multiple API versions:

```caddyfile
handle_path /v1/* {
  reverse_proxy http://backend-v1:4000
}

handle_path /v2/* {
  reverse_proxy http://backend-v2:4000
}
```

### Pattern 3: Microservices with Auth

Each service gets its own route with auth:

```caddyfile
handle_path /users* {
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://user-service:8001
}

handle_path /orders* {
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://order-service:8002
}
```

## Testing Path Rewriting

### Verify Configuration

```bash
# Validate Caddyfile syntax
docker compose exec caddy caddy validate --config /etc/caddy/Caddyfile

# Check running config
docker compose exec caddy caddy list-certificates
```

### Test Path Mapping

```bash
# Test with authentication (requires valid session)
curl -v https://your-domain.com/rag/health

# Test locally (bypasses Caddy)
curl http://localhost:4002/health

# Compare responses - should be identical
```

### Debug Path Issues

```bash
# Enable Caddy debug logging
docker compose exec caddy caddy adapt --config /etc/caddy/Caddyfile

# View Caddy logs
docker compose logs caddy -f

# Check what backend receives
docker compose logs backend -f | grep "GET /health"
```

## Troubleshooting

### Issue: 404 Not Found

**Symptoms:** Request returns 404 even though backend is running

**Possible causes:**
1. Path pattern doesn't match request
2. Trailing slashes mismatch
3. Backend expects different path

**Solutions:**
```bash
# Check Caddy routes
docker compose exec caddy caddy list-config

# Test backend directly
curl http://localhost:4002/health

# Check Caddy logs for route matching
docker compose logs caddy | grep -i "404"
```

### Issue: Path Not Stripped

**Symptoms:** Backend receives full path with prefix

**Possible causes:**
1. Using `handle` instead of `handle_path`
2. Wildcard pattern incorrect

**Solutions:**
```caddyfile
# ❌ Wrong: Keeps prefix
handle /rag/* {
  reverse_proxy http://backend:4002
}

# ✅ Correct: Strips prefix
handle_path /rag* {
  reverse_proxy http://backend:4002
}
```

### Issue: Auth Headers Not Passed

**Symptoms:** Backend doesn't receive identity headers

**Possible causes:**
1. `copy_headers` directive missing
2. Header names incorrect

**Solutions:**
```caddyfile
# Ensure copy_headers is inside forward_auth block
forward_auth http://authentik_proxy:9000 {
  uri /outpost.goauthentik.io/auth/caddy
  copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
}
```

### Issue: CORS Errors

**Symptoms:** Browser shows CORS errors for API requests

**Possible causes:**
1. Backend not handling CORS
2. Preflight requests not reaching backend

**Solutions:**
```caddyfile
# Add CORS headers in Caddy
handle_path /rag* {
  @cors {
    method OPTIONS
  }
  header @cors {
    Access-Control-Allow-Origin "*"
    Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Access-Control-Allow-Headers "Content-Type, Authorization"
  }
  respond @cors 204
  
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://backend:4002
}
```

## Adding New Routes

To add a new service with path rewriting:

### Step 1: Add Route to Caddyfile

```caddyfile
# /myservice protected by forward_auth, strips /myservice prefix
handle_path /myservice* {
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://myservice:8080
}
```

### Step 2: Add Service to docker-compose.yml

```yaml
myservice:
  build: ./myservice
  restart: unless-stopped
  expose:
    - "8080"
  environment:
    - DATABASE_URL=...
```

### Step 3: Restart Caddy

```bash
docker compose restart caddy
```

### Step 4: Test

```bash
# Should strip /myservice prefix
curl https://your-domain.com/myservice/health

# Backend should receive
# GET /health (without /myservice)
```

## Best Practices

### 1. Always Use Path Stripping

Default to `handle_path` for clean backend APIs:

```caddyfile
# ✅ Preferred
handle_path /service* {
  reverse_proxy http://service:8080
}

# ❌ Avoid unless necessary
handle /service/* {
  reverse_proxy http://service:8080
}
```

### 2. Consistent Wildcard Patterns

Use `*` for trailing paths:

```caddyfile
# Matches /rag, /rag/, /rag/health, /rag/upload
handle_path /rag* {
  reverse_proxy http://backend:4002
}
```

### 3. Order Routes Specifically

More specific routes first:

```caddyfile
# ✅ Correct order: specific before general
handle_path /api/admin* {
  reverse_proxy http://admin:8000
}

handle_path /api* {
  reverse_proxy http://api:4000
}

# ❌ Wrong order: general first would catch everything
```

### 4. Always Include Auth

Protect all routes with forward_auth:

```caddyfile
handle_path /sensitive* {
  # ✅ Auth included
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://backend:4002
}
```

### 5. Document Path Mappings

Keep path mappings documented:

```caddyfile
# Client: GET /rag/health -> Backend: GET /health
# Client: POST /rag/upload -> Backend: POST /upload
handle_path /rag* {
  reverse_proxy http://backend:4002
}
```

## Additional Resources

- **Caddy Documentation**: https://caddyserver.com/docs/
- **handle_path Directive**: https://caddyserver.com/docs/caddyfile/directives/handle_path
- **forward_auth**: https://caddyserver.com/docs/caddyfile/directives/forward_auth
- **Project Structure**: [`docs/STRUCTURE.md`](STRUCTURE.md)
- **Authentication**: [`docs/OPENWEBUI_AUTHENTIK.md`](OPENWEBUI_AUTHENTIK.md)

## Summary

✅ **`handle_path` strips matched prefix** before proxying  
✅ **Backends use clean root paths** (e.g., `/health`)  
✅ **Services are portable** - work in any context  
✅ **No environment-specific configuration** needed  
✅ **All routes protected** by Authentik forward_auth  
✅ **Path mappings documented** for clarity  

This approach provides clean, maintainable routing while preserving security and flexibility! 🚀

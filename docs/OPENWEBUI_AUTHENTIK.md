# OpenWebUI Authentik Integration

This document explains how OpenWebUI integrates with Authentik for authentication in the AgentKS stack.

## Architecture Overview

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ HTTP Request to /webui
     ↓
┌──────────┐
│  Caddy   │ Reverse Proxy
└────┬─────┘
     │ forward_auth to Authentik Proxy
     ↓
┌──────────────┐
│ Authentik    │ Validates session
│ Proxy        │
└────┬─────────┘
     │ Returns auth headers
     ↓
┌──────────┐
│  Caddy   │ Copies headers:
└────┬─────┘   - X-Authentik-Email
     │          - X-Authentik-Name
     │          - X-Authentik-Groups
     ↓
┌──────────────┐
│  OpenWebUI   │ Trusts headers for identity
│  (Port 8080) │ Uses PostgreSQL for data
└──────────────┘
```

## Key Principles

✅ **OpenWebUI does NOT have its own authentication system** in this setup  
✅ **ALL authentication is handled by Authentik** through Caddy forward_auth  
✅ **OpenWebUI trusts the identity headers** injected by Caddy  
✅ **Users must log in through Authentik first**  
✅ **Identity is passed via HTTP headers** after successful authentication  
✅ **OpenWebUI automatically creates/logs in users** based on email header  
✅ **Group membership can be used** for role-based access control  

## Configuration

### Docker Compose Configuration

In `docker-compose.yml`, OpenWebUI is configured with:

```yaml
openwebui:
  image: ghcr.io/open-webui/open-webui:latest
  restart: unless-stopped
  environment:
    WEBUI_AUTH: true  # Enable trusted header authentication
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER: X-Authentik-Email
    WEBUI_AUTH_TRUSTED_NAME_HEADER: X-Authentik-Name
    WEBUI_AUTH_TRUSTED_GROUPS_HEADER: X-Authentik-Groups
    DATABASE_URL: postgresql://${AK_POSTGRES_USER}:${AK_POSTGRES_PASSWORD}@postgres:5432/${OPENWEBUI_POSTGRES_DB:-openwebui}
  volumes:
    - openwebui_data:/app/backend/data
  depends_on:
    postgres:
      condition: service_healthy
  expose:
    - "8080"
```

**Key Environment Variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `WEBUI_AUTH` | `true` | Enable trusted header authentication |
| `WEBUI_AUTH_TRUSTED_EMAIL_HEADER` | `X-Authentik-Email` | Header containing user email (canonical ID) |
| `WEBUI_AUTH_TRUSTED_NAME_HEADER` | `X-Authentik-Name` | Header containing user display name |
| `WEBUI_AUTH_TRUSTED_GROUPS_HEADER` | `X-Authentik-Groups` | Header containing user groups (comma-separated) |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |

### Caddyfile Configuration

In `Caddyfile`, the `/webui` route is protected:

```caddyfile
handle_path /webui* {
  forward_auth http://authentik_proxy:9000 {
    uri /outpost.goauthentik.io/auth/caddy
    copy_headers X-Authentik-Email X-Authentik-Name X-Authentik-Groups
  }
  reverse_proxy http://openwebui:8080
}
```

**What this does:**
1. All requests to `/webui/*` are intercepted by Caddy
2. Caddy sends an auth check to `authentik_proxy:9000`
3. If authenticated, Authentik returns success with user info
4. Caddy copies the identity headers and forwards to OpenWebUI
5. OpenWebUI receives authenticated request with identity headers

### Environment Variables (.env)

Required in your `.env` file:

```bash
# OpenWebUI configuration
WEBUI_AUTH=true  # Enable Authentik-based authentication via trusted headers
OPENWEBUI_POSTGRES_DB=openwebui  # Optional, defaults to 'openwebui'

# Database credentials (shared with Authentik and Backend)
AK_POSTGRES_USER=authentik
AK_POSTGRES_PASSWORD=<secure-password>

# Authentik configuration
AUTHENTIK_OUTPOST_TOKEN=<token-from-authentik-admin>
```

## Authentication Flow

### Step-by-Step Process

1. **User navigates to `/webui`**
   - Browser sends HTTP request to `http://domain.com/webui`

2. **Caddy intercepts request**
   - Caddy receives the request
   - Invokes `forward_auth` to check authentication

3. **Authentik validates session**
   - Caddy forwards auth check to Authentik Proxy
   - Authentik checks for valid session cookie
   - If no valid session → returns 401/302 redirect to login

4. **User logs in (if needed)**
   - Redirected to Authentik login page (`https://auth.domain.com`)
   - User enters credentials
   - Authentik validates and creates session

5. **Redirected back to OpenWebUI**
   - After successful login, redirected to `/webui`
   - Authentik now returns auth success with user metadata

6. **Caddy injects identity headers**
   - `X-Authentik-Email: user@example.com`
   - `X-Authentik-Name: John Doe`
   - `X-Authentik-Groups: users,admin`

7. **OpenWebUI receives authenticated request**
   - OpenWebUI trusts the headers (can't be spoofed externally)
   - Automatically creates user account if first login
   - Loads user session and chat history from PostgreSQL

### Sequence Diagram

```
┌────────┐         ┌───────┐         ┌────────────┐         ┌──────────┐
│ User   │         │ Caddy │         │ Authentik  │         │ OpenWebUI│
└───┬────┘         └───┬───┘         └─────┬──────┘         └────┬─────┘
    │                  │                   │                      │
    │ GET /webui       │                   │                      │
    ├─────────────────>│                   │                      │
    │                  │                   │                      │
    │                  │ forward_auth      │                      │
    │                  ├──────────────────>│                      │
    │                  │                   │                      │
    │                  │ 302 Redirect      │                      │
    │                  │<──────────────────┤                      │
    │                  │                   │                      │
    │ 302 Redirect to login                │                      │
    │<─────────────────┤                   │                      │
    │                  │                   │                      │
    │ POST /login (credentials)            │                      │
    ├──────────────────┼──────────────────>│                      │
    │                  │                   │                      │
    │ 302 Redirect back to /webui          │                      │
    │<─────────────────┼───────────────────┤                      │
    │                  │                   │                      │
    │ GET /webui       │                   │                      │
    ├─────────────────>│                   │                      │
    │                  │                   │                      │
    │                  │ forward_auth      │                      │
    │                  ├──────────────────>│                      │
    │                  │                   │                      │
    │                  │ 200 OK + headers  │                      │
    │                  │<──────────────────┤                      │
    │                  │                   │                      │
    │                  │ Request + X-Authentik-* headers          │
    │                  ├──────────────────────────────────────────>│
    │                  │                   │                      │
    │                  │           Response (chat UI)             │
    │                  │<──────────────────────────────────────────┤
    │                  │                   │                      │
    │ Response         │                   │                      │
    │<─────────────────┤                   │                      │
    │                  │                   │                      │
```

## Security Model

### Trusted Headers Architecture

**Why it's secure:**

1. **OpenWebUI is NOT exposed directly**
   - Only accessible through Caddy reverse proxy
   - No direct access to `openwebui:8080` from outside Docker network
   - Port 8080 is `expose`d not `ports`, so not accessible from host

2. **Headers can only be set by Caddy**
   - External clients cannot inject headers (stripped by Caddy)
   - Caddy is the only trusted source for `X-Authentik-*` headers
   - Headers are added AFTER successful authentication

3. **Authentik validates all requests**
   - Every request to `/webui` goes through forward_auth
   - Authentik checks session validity before allowing access
   - Session cookies are HTTP-only and secure

4. **Docker network isolation**
   - OpenWebUI, Caddy, and Authentik on same Docker network
   - No external access to internal services
   - All communication over internal Docker DNS

### What Could Go Wrong?

❌ **If OpenWebUI was exposed directly** (e.g., `ports: - "8080:8080"`):
   - Attacker could send requests with fake `X-Authentik-*` headers
   - Would bypass authentication completely
   - **This is why we use `expose` not `ports`**

❌ **If Caddy was misconfigured** (e.g., no forward_auth):
   - Requests would go directly to OpenWebUI without auth
   - Users could access without logging in
   - **This is why forward_auth is critical**

✅ **With proper configuration**:
   - All requests MUST go through Caddy → Authentik → OpenWebUI
   - Headers are trusted because they can only be added by Caddy
   - Attackers cannot bypass the authentication flow

## Database Integration

### PostgreSQL Setup

OpenWebUI uses PostgreSQL instead of the default SQLite:

**Benefits:**
- Better performance for concurrent users
- Proper database scaling and backup
- Shared infrastructure with Authentik and Backend
- Support for advanced queries and indexes

**Database Structure:**
```
PostgreSQL Server (postgres:5432)
├── authentik (Authentik's database)
│   ├── Authentik tables (users, sessions, etc.)
│   ├── tool_catalog (Backend's tool registry)
│   └── kb_docs (Backend's knowledge base)
└── openwebui (OpenWebUI's database)
    ├── user (User accounts)
    ├── chat (Chat conversations)
    ├── message (Chat messages)
    ├── document (Uploaded documents)
    └── ... (other OpenWebUI tables)
```

**Connection String:**
```bash
DATABASE_URL=postgresql://${AK_POSTGRES_USER}:${AK_POSTGRES_PASSWORD}@postgres:5432/${OPENWEBUI_POSTGRES_DB:-openwebui}
```

### Migration from SQLite

If you're migrating from SQLite to PostgreSQL:

```bash
# 1. Stop services
docker compose down

# 2. Ensure OPENWEBUI_POSTGRES_DB is set in .env
echo "OPENWEBUI_POSTGRES_DB=openwebui" >> .env

# 3. Start with new configuration
docker compose up -d

# OpenWebUI will automatically create the PostgreSQL database on first run
# Note: Existing SQLite data in openwebui_data volume is NOT automatically migrated
```

**To preserve existing data:**
1. Export data from SQLite (using OpenWebUI's export feature)
2. Start with PostgreSQL configuration
3. Import data (using OpenWebUI's import feature)

## User Management

### Automatic User Creation

When a user first accesses OpenWebUI through Authentik:

1. Caddy forwards request with `X-Authentik-Email: user@example.com`
2. OpenWebUI checks if user exists in database
3. If not exists → creates new user account automatically
4. User record includes:
   - Email (from `X-Authentik-Email`)
   - Name (from `X-Authentik-Name`)
   - Groups (from `X-Authentik-Groups`)

### User Profile Sync

Every request includes fresh identity headers:
- User's name is synced from Authentik on each request
- Group membership is updated automatically
- Profile changes in Authentik propagate immediately

### Role-Based Access Control

Groups from Authentik can be used for access control:

```python
# Example: Check if user is admin
def is_admin(request):
    groups = request.headers.get("X-Authentik-Groups", "")
    return "admin" in groups.lower().split(",")
```

OpenWebUI can use these groups for:
- Admin panel access
- Feature flags
- Resource quotas
- Custom UI elements

## Troubleshooting

### User Can't Access OpenWebUI

**Symptoms:** Redirected to login, but login doesn't work

**Check:**
1. Verify Authentik is running: `docker compose ps authentik_server`
2. Check Authentik logs: `docker compose logs authentik_server`
3. Verify outpost token is set: `echo $AUTHENTIK_OUTPOST_TOKEN`
4. Check Caddy forward_auth config in Caddyfile

**Solution:**
```bash
# Restart Authentik services
docker compose restart authentik_server authentik_proxy

# Check forward_auth endpoint
curl http://localhost:9000/outpost.goauthentik.io/auth/caddy
```

### Headers Not Being Passed

**Symptoms:** User logged in to Authentik, but OpenWebUI shows anonymous

**Check:**
1. Verify Caddy is copying headers:
   ```bash
   docker compose logs caddy | grep -i authentik
   ```
2. Check OpenWebUI environment variables:
   ```bash
   docker compose exec openwebui env | grep WEBUI_AUTH
   ```

**Solution:**
```bash
# Restart Caddy to reload config
docker compose restart caddy

# Verify headers are configured
grep -A5 "handle_path /webui" Caddyfile
```

### Database Connection Issues

**Symptoms:** OpenWebUI starts but can't save data

**Check:**
1. Verify PostgreSQL is running: `docker compose ps postgres`
2. Check database exists:
   ```bash
   docker compose exec postgres psql -U authentik -c "\l" | grep openwebui
   ```
3. Test connection:
   ```bash
   docker compose exec openwebui env | grep DATABASE_URL
   ```

**Solution:**
```bash
# Create database manually if needed
docker compose exec postgres psql -U authentik -c "CREATE DATABASE openwebui;"

# Restart OpenWebUI to reconnect
docker compose restart openwebui
```

### Session Expires Too Quickly

**Symptoms:** Users logged out frequently

**Solution:**
Configure session timeout in Authentik admin panel:
1. Navigate to Authentik admin: `https://auth.your-domain.com/if/admin/`
2. Go to System → Tenants
3. Adjust "Session duration"
4. Save changes

### Can't Access OpenWebUI from Outside

**Symptoms:** Works locally but not from internet

**Check:**
1. Verify domain DNS points to server
2. Check firewall allows ports 80/443
3. Verify Caddy TLS certificate:
   ```bash
   docker compose exec caddy caddy list-certificates
   ```

**Solution:**
```bash
# Check Caddy logs for TLS errors
docker compose logs caddy | grep -i tls

# Verify domain resolves
dig +short your-domain.com

# Test HTTPS
curl -I https://your-domain.com/webui
```

## Testing

### Verify Configuration

```bash
# 1. Check all services are running
docker compose ps

# 2. Verify OpenWebUI environment
docker compose exec openwebui env | grep -E "WEBUI_AUTH|DATABASE_URL"

# 3. Test database connection
docker compose exec postgres psql -U authentik -c "\c openwebui; \dt"

# 4. Check Caddy config
docker compose exec caddy caddy validate --config /etc/caddy/Caddyfile

# 5. Test authentication flow (should redirect to login)
curl -I http://localhost/webui

# 6. Check OpenWebUI logs
docker compose logs openwebui | tail -n 50
```

### Manual Testing

1. **Access OpenWebUI:**
   - Navigate to `http://your-domain.com/webui`
   - Should redirect to Authentik login

2. **Login:**
   - Enter valid Authentik credentials
   - Should redirect back to OpenWebUI

3. **Verify Identity:**
   - Check user profile in OpenWebUI
   - Email should match Authentik account

4. **Test Persistence:**
   - Create a chat conversation
   - Logout and login again
   - Chat history should persist

## Configuration Reference

### Required Environment Variables

```bash
# Enable Authentik authentication
WEBUI_AUTH=true

# Database configuration
OPENWEBUI_POSTGRES_DB=openwebui
AK_POSTGRES_USER=authentik
AK_POSTGRES_PASSWORD=<secure-password>

# Authentik configuration
AUTHENTIK_OUTPOST_TOKEN=<from-authentik-admin>
```

### Authentik Admin Setup

1. **Create Outpost:**
   - Go to Applications → Outposts
   - Create new Outpost of type "Proxy Provider"
   - Note the token

2. **Configure Provider:**
   - External host: `https://your-domain.com`
   - Forward auth mode: `Caddy (forward auth)`
   - Trusted networks: Leave default

3. **Set Environment Variable:**
   ```bash
   echo "AUTHENTIK_OUTPOST_TOKEN=<token>" >> .env
   ```

4. **Restart Services:**
   ```bash
   docker compose down
   docker compose up -d
   ```

## Additional Resources

- **Authentik Documentation:** https://goauthentik.io/docs/
- **OpenWebUI Documentation:** https://docs.openwebui.com/
- **Caddy Documentation:** https://caddyserver.com/docs/
- **Project Structure:** [`docs/STRUCTURE.md`](STRUCTURE.md)
- **Authentication Flow:** [`README.md`](../README.md)

## Summary

✅ **OpenWebUI uses Authentik for authentication** via trusted headers  
✅ **No separate OpenWebUI login** - all auth through Authentik SSO  
✅ **Secure by design** - headers can't be spoofed externally  
✅ **PostgreSQL backend** for better performance and scaling  
✅ **Automatic user provisioning** based on Authentik identity  
✅ **Group-based access control** via Authentik groups  
✅ **Single Sign-On experience** across all services  

This architecture provides enterprise-grade authentication while maintaining a seamless user experience! 🔐

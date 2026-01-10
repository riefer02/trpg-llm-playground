# Lancer Web Application

Full-stack web application for Lancer TTRPG tactical combat.

## Architecture

```
app/
├── backend/           # FastAPI REST API
│   ├── api/           # Route handlers
│   ├── db/            # SQLModel + Alembic
│   └── tests/         # pytest test suite
│
└── frontend/          # TanStack Start + React
    ├── src/
    │   ├── routes/    # File-based routing
    │   ├── components/# React components
    │   └── lib/       # API client, hooks, types
    └── schemas/       # Generated JSON Schema
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for PostgreSQL)

### Setup

```bash
# From repository root

# 1. Install dependencies
make install-app

# 2. Copy environment file
cp .env.example .env

# 3. Start PostgreSQL
make db-up

# 4. Run database migrations
make db-migrate

# 5. Start development servers
make dev
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000/api |
| API Documentation | http://localhost:8000/api/docs |
| Health Check | http://localhost:8000/api/health |

## Development

### Running Tests

```bash
# All app tests
make test-app

# Backend only
cd app/backend && pytest -v

# Frontend only
cd app/frontend && npm test
```

### Database Migrations

```bash
# Apply migrations
make db-migrate

# Create new migration
make db-revision MSG="add campaigns table"
```

### Type Generation

TypeScript types are auto-generated from Python Pydantic models:

```bash
make generate-types
```

This runs:
1. `python -m core.export` → JSON Schema
2. `json-schema-to-typescript` → TypeScript definitions

## Backend Patterns

### Adding a New Endpoint

1. Create route file in `app/backend/api/`:

```python
# app/backend/api/campaigns.py
from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.dependencies import get_current_user

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

@router.get("")
async def list_campaigns(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    # Implementation
    pass
```

2. Register in `app/backend/api/router.py`:

```python
from app.backend.api.campaigns import router as campaigns_router
api_router.include_router(campaigns_router)
```

### Error Handling

Use custom exceptions from `app/backend/exceptions.py`:

```python
from app.backend.exceptions import NotFoundError, ValidationError

# Raises 404 with consistent JSON format
raise NotFoundError("Campaign", campaign_id)

# Raises 422 with validation errors
raise ValidationError("Invalid data", errors=[...])
```

## Frontend Patterns

### Adding API Hooks

1. Create hooks in `app/frontend/src/lib/api/`:

```typescript
// app/frontend/src/lib/api/campaigns.ts
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from './client'

export function useCampaigns() {
  return useQuery({
    queryKey: ['campaigns'],
    queryFn: () => api.get<Campaign[]>('/campaigns'),
  })
}
```

2. Export from `app/frontend/src/lib/api/index.ts`

### Adding Routes

Create file in `app/frontend/src/routes/`:

```typescript
// app/frontend/src/routes/campaigns/index.tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/campaigns/')({
  component: CampaignsPage,
})

function CampaignsPage() {
  return <div>Campaigns</div>
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://lancer:lancer@localhost:5433/lancer` |
| `DEBUG` | Enable debug mode | `true` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173` |
| `VITE_API_URL` | Frontend API base URL | `http://localhost:8000/api` |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend Framework | FastAPI |
| Database | PostgreSQL + SQLModel |
| Migrations | Alembic |
| Frontend Framework | TanStack Start |
| Data Fetching | React Query |
| Styling | Tailwind CSS |
| UI Components | shadcn/ui patterns |

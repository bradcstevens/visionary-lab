# Azure Deployment with Azure Developer CLI (azd)

This guide shows how to deploy the Visionary Lab to Azure using the Azure Developer CLI for one-click deployments.

## Prerequisites

- [Azure Developer CLI (azd)](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd) installed
- Azure subscription with access to:
  - Azure AI Foundry (AIServices)
  - Azure Container Apps
  - Azure Storage Account
  - Azure Cosmos DB
  - Azure Log Analytics

## Quick Start (One-Click Deployment)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd visionary-lab
   ```

2. **Authenticate and deploy**:
   ```bash
   azd auth login
   azd up
   ```

3. **Configure during deployment**:
   When prompted by `azd up`, provide:

   - **AI_FOUNDRY_NAME**: Name for your AI Foundry resource (must be globally unique)
   - **AI_FOUNDRY_LOCATION**: Azure region for AI Foundry (default: `swedencentral`)
   - **LLM_DEPLOYMENT**: LLM deployment name (default: `gpt-5-4`)
   - **IMAGEGEN_DEPLOYMENT**: Image generation deployment name (default: `gpt-image-1-5`)
   - **SORA_DEPLOYMENT**: Video generation deployment name (default: `sora`)

   > **No API keys required.** All services use Azure Managed Identity for authentication.

That's it! The `azd up` command will:
- Create a new environment
- Provision the AI Foundry resource with all model deployments
- Provision Storage, Cosmos DB, Container Registry, Container Apps
- Assign RBAC roles (Cognitive Services OpenAI User, Storage Blob Data Contributor, etc.)
- Build and deploy Docker images for frontend and backend
- Configure networking and environment variables
- Provide you with the application URLs

## Manual Steps

If you prefer manual control over the deployment process:

### 1. Initialize Environment
```bash
azd env new <environment-name>
```

### 2. Configure Environment Variables
```bash
# AI Foundry
azd env set AI_FOUNDRY_NAME "your-foundry-name"
azd env set AI_FOUNDRY_LOCATION "swedencentral"

# Model deployments (names must match what gets deployed)
azd env set LLM_DEPLOYMENT "gpt-5-4"
azd env set IMAGEGEN_DEPLOYMENT "gpt-image-1-5"
azd env set IMAGEGEN_15_DEPLOYMENT "gpt-image-1-5"
azd env set IMAGEGEN_1_MINI_DEPLOYMENT "gpt-image-1-mini"
azd env set SORA_DEPLOYMENT "sora"
```

### 3. Deploy Infrastructure
```bash
azd provision
```

### 4. Deploy Application
```bash
azd deploy
```

## Architecture

The deployment creates:

- **Azure AI Foundry** (AIServices): Unified AI resource with all model deployments
- **AI Foundry Project**: Scoped workspace for the application
- **Azure Container Apps Environment**: Serverless container hosting
- **Backend Container App**: FastAPI application (Python) with SystemAssigned managed identity
- **Frontend Container App**: Next.js application (Node.js)
- **Azure Container Registry**: Private registry for storing Docker images
- **Azure Storage Account**: For storing generated images and videos
- **Azure Cosmos DB**: For metadata storage
- **Log Analytics Workspace**: For monitoring and logging

### RBAC Role Assignments (auto-provisioned)

| Principal | Role | Scope |
|-----------|------|-------|
| Backend Container App | Cognitive Services OpenAI User | AI Foundry |
| Backend Container App | Storage Blob Data Contributor | Storage Account |
| Backend Container App | Storage Blob Delegator | Storage Account |
| Backend Container App | Cosmos DB Data Contributor | Cosmos DB Account |

## Environment Variables

The following environment variables are automatically configured by the infrastructure:

### Backend
- `AI_FOUNDRY_ENDPOINT`: AI Foundry endpoint URL
- `LLM_DEPLOYMENT`: LLM deployment name
- `IMAGEGEN_DEPLOYMENT`: Image generation deployment name
- `IMAGEGEN_15_DEPLOYMENT`: GPT-Image-1.5 deployment name
- `IMAGEGEN_1_MINI_DEPLOYMENT`: GPT-Image-1-mini deployment name
- `SORA_DEPLOYMENT`: Sora deployment name
- `AZURE_BLOB_SERVICE_URL`: Storage endpoint URL
- `AZURE_STORAGE_ACCOUNT_NAME`: Storage account name
- `AZURE_BLOB_IMAGE_CONTAINER`: Container for images (default: "images")
- `AZURE_COSMOS_DB_ENDPOINT`: Cosmos DB endpoint
- `AZURE_COSMOS_DB_ID`: Database name
- `AZURE_COSMOS_CONTAINER_ID`: Container name

## Local Development

For local development, the app uses `DefaultAzureCredential` which picks up your Azure CLI credentials:

```bash
# Login to Azure (required for local development)
az login

# Set environment variables in .env (see .env.example)
cp .env.example .env
# Edit .env with your AI Foundry endpoint and deployment names

# Run the backend
cd backend && uvicorn main:app --reload
```

## Monitoring

Access your deployment logs and metrics:

```bash
# View application logs
azd logs

# Monitor resources in Azure Portal
azd show --output table
```

## Cleanup

To remove all Azure resources:

```bash
azd down
```

## Persistent Image-Job Queue

The deployment provisions a persistent image-job queue (Azure Storage
Queue + KEDA-scaled worker Container App) so image regeneration jobs
survive worker restarts and rolling deploys. See
`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md` for the
full design.

### What gets deployed

- Two queues on the existing storage account: `imagejobs` (work) and
  `imagejobs-poison` (max-dequeue overflow, poison TTL 7 days).
- A second Container App `ca-worker-<env>` running the same image as
  the backend with `ROLE=worker`. It has **no ingress** so the KEDA
  `azure-queue` scale rule can legitimately scale it to **0 replicas**
  at idle. Trigger: 1 replica per 5 pending messages, capped at 10.
- `Storage Queue Data Contributor` granted to both the backend and
  worker managed identities (no connection strings; managed identity
  only).

### Feature-flag toggle

The pipeline path is gated by `FEATURE_ASYNC_QUEUE`:

- `true` (default in dev/staging) — `staging_pipeline.py` enqueues
  jobs onto `imagejobs`; the worker dispatches.
- `false` — falls back to the in-process path that the backend used
  before this feature shipped. Useful for one-release rollback.

To flip it on a deployed environment without redeploying images:

```bash
# Inspect current value
az containerapp show -n ca-backend-<env> -g <rg> \
  --query "properties.template.containers[0].env[?name=='FEATURE_ASYNC_QUEUE']"

# Flip OFF (forces in-process fallback)
az containerapp update -n ca-backend-<env> -g <rg> \
  --set-env-vars FEATURE_ASYNC_QUEUE=false
az containerapp update -n ca-worker-<env> -g <rg> \
  --set-env-vars FEATURE_ASYNC_QUEUE=false

# Flip ON
az containerapp update -n ca-backend-<env> -g <rg> \
  --set-env-vars FEATURE_ASYNC_QUEUE=true
az containerapp update -n ca-worker-<env> -g <rg> \
  --set-env-vars FEATURE_ASYNC_QUEUE=true
```

The flag flips a single revision in place; the API replica is single-
instance (see backend `maxReplicas: 1` rationale in `containerApp.bicep`)
so there is no split-state window. Worker replicas pick up the new
revision on next scale-out tick (≤ 30s).

### Rolling-deploy drain window

Worker replicas hold a Storage Queue message lease for **90 seconds**
(JobQueue visibility timeout). On rolling deploy, Container Apps
gracefully drains an old revision by:

1. Stopping new dequeues on the old replica (revision marked Inactive).
2. Letting the old replica finish any message currently being processed.
3. If the worker process does not exit within the **default 30s
   termination grace period**, the platform sends SIGKILL. The
   abandoned message becomes visible again 90s after its dequeue
   timestamp and any healthy replica re-leases it. The job is
   idempotent — `JobStore` writes are partition-keyed by `project_id`
   and the deterministic job id `{project_id}:{room_id}:{variation_id}:
   {revision}` ensures the re-run is the same logical job.

Recommended rolling-deploy procedure:

```bash
# 1. Drain proactively (optional but reduces re-runs):
#    set worker minReplicas/maxReplicas to (current, current+0) so
#    no new replicas spin up while the queue drains.
az containerapp update -n ca-worker-<env> -g <rg> \
  --min-replicas 0 --max-replicas 0
# Wait until the queue depth reads 0:
az storage queue stats --account-name <storage> --queue-name imagejobs \
  --auth-mode login

# 2. Deploy the new image:
azd deploy --service worker

# 3. Restore the scale envelope:
az containerapp update -n ca-worker-<env> -g <rg> \
  --min-replicas 0 --max-replicas 10
```

If the queue does not drain within ~10 minutes, abandon-and-redeploy
is safe — every job message that was in flight will be re-leased by
the new revision after its 90s visibility window. The user-visible
effect is one duplicate progress event per in-flight job, never a
lost or partially-applied generation.

To verify the worker reached zero at idle after a fresh deploy:

```bash
az containerapp replica list -n ca-worker-<env> -g <rg> -o table
# Empty result == scaled to zero (expected when imagejobs is empty).
```

### Worker observability (KQL)

The worker emits structured log events at every state transition.
Events appear in the `ContainerAppConsoleLogs_CL` table in the
Log Analytics workspace. Useful KQL queries:

```kql
// Job throughput by terminal state, last 24h.
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "ca-worker-<env>"
| where Log_s has_any ("job.succeeded", "job.failed", "job.cancelled")
| extend kind = case(
    Log_s has "job.succeeded", "succeeded",
    Log_s has "job.failed",    "failed",
    Log_s has "job.cancelled", "cancelled",
    "other")
| summarize count() by kind, bin(TimeGenerated, 1h)
| render columnchart
```

```kql
// Jobs that reached the final attempt and went to poison.
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "ca-worker-<env>"
| where Log_s has "job.failed" and Log_s has "terminal=True"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```

```kql
// Per-job lifecycle trace by job_id (substitute the id you care about).
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "ca-worker-<env>"
| where Log_s has "<project_id>:<room_id>:<variation_id>:<revision>"
| project TimeGenerated, Log_s
| order by TimeGenerated asc
```

The structured event names emitted by `JobWorker` are: `job.started`,
`job.succeeded`, `job.failed`, `job.cancelled`, `job.missing` (stale
queue pointer to a deleted doc). `JobQueue` adds `job.enqueued`,
`job.abandoned`, and `job.poisoned`. `ProgressEstimator` (issue 008)
will add `job.progress`.

## Troubleshooting

### Common Issues

1. **Credential errors locally**: Run `az login` to authenticate. `DefaultAzureCredential` requires an active Azure CLI session.
2. **RBAC propagation delay**: After initial deployment, role assignments may take 1-5 minutes to propagate. If the app shows 403 errors on first start, wait and restart.
3. **Region availability**: Some models (Sora, GPT-Image) may not be available in all regions. Default is `swedencentral`.
4. **Permission Issues**: You need Owner role on the resource group to create RBAC assignments.

### Getting Help

```bash
# Check azd status
azd env list

# View detailed logs
azd logs --follow

# Get environment info
azd env get-values
```

For more information, see the [Azure Developer CLI documentation](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/).

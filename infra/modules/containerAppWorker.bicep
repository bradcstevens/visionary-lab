// Worker variant of the backend Container App used to consume the
// persistent image-job queue described in
// `prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`
// (Infrastructure → Storage Queue + KEDA worker; user stories 19, 38–40).
//
// Differences from `containerApp.bicep`:
//
//   - No ingress. Workers receive work via the Storage Queue, not HTTP,
//     so disabling ingress is what allows the KEDA azure-queue scale
//     rule to legitimately scale this app down to zero replicas at
//     idle (Container Apps will not idle a replica below 1 while
//     ingress is enabled).
//   - `ROLE=worker` env var so the same image (built once for both
//     api + worker) can branch its bootstrap into the JobWorker
//     consumer loop.
//   - KEDA `azure-queue` scale rule (min=0, max=10, queueLength=5)
//     authenticated via the system-assigned managed identity — no
//     connection strings.
//   - SecurityControl=Ignore tag on the Container App resource so the
//     tenant MCAPS local-auth governance policies do not interact
//     with the worker (parity with the storage account tag added in
//     `storageAccount.bicep`).

param location string
param containerAppName string
param containerAppEnvId string
param DOCKER_IMAGE string
param deployNew bool = true
param azdServiceName string = ''

// AI Foundry endpoint + model deployment names — workers run the same
// `ImagePipelineService` as the API replica and therefore need the
// identical Foundry wiring.
param AI_FOUNDRY_ENDPOINT string = ''
param LLM_DEPLOYMENT string = 'gpt-5-4'
param IMAGEGEN_DEPLOYMENT string = 'gpt-image-2'
param IMAGEGEN_15_DEPLOYMENT string = ''
param IMAGEGEN_1_MINI_DEPLOYMENT string = ''
param SORA_DEPLOYMENT string = 'sora-2'
param FLUX_KONTEXT_DEPLOYMENT string = ''

// Azure Blob Storage (managed identity — no keys)
param AZURE_BLOB_SERVICE_URL string
param AZURE_STORAGE_ACCOUNT_NAME string
param AZURE_BLOB_IMAGE_CONTAINER string = 'images'
param CDN_BLOB_URL string = ''

// Azure Storage Queue (managed identity — no keys). The worker reads
// from `imagejobs` and the JobQueue routes max-dequeue overflow to
// `imagejobs-poison` — see `storageAccount.bicep` for the queue-resource
// definitions.
param AZURE_STORAGE_QUEUE_URL string
param JOB_QUEUE_NAME string = 'imagejobs'
param JOB_QUEUE_POISON_NAME string = 'imagejobs-poison'

// Cosmos DB (managed identity — no keys)
param COSMOS_ENDPOINT string = ''
param COSMOS_DATABASE_NAME string = ''
param COSMOS_CONTAINER_NAME string = ''

// Azure Container Registry (image pull)
param AZURE_CONTAINER_REGISTRY_ENDPOINT string = ''
@secure()
param AZURE_CONTAINER_REGISTRY_USERNAME string = ''
@secure()
param AZURE_CONTAINER_REGISTRY_PASSWORD string = ''

// KEDA azure-queue scale rule defaults — match PRD § "KEDA azure-queue
// scale rule with min=0, max=10, queueLength=5".
@description('Minimum replica count. 0 enables true scale-to-zero (PRD requirement).')
param minReplicas int = 0
@description('Maximum replica count. PRD default is 10.')
param maxReplicas int = 10
@description('KEDA queueLength trigger — replicas added per N pending messages.')
param queueLength int = 5

@description('Resource tags merged onto the worker container app. SecurityControl=Ignore exempts the resource from the tenant MCAPS local-auth governance policies.')
param tags object = {
  SecurityControl: 'Ignore'
}

resource workerApp 'Microsoft.App/containerApps@2024-03-01' = if (deployNew) {
  name: containerAppName
  location: location
  tags: union(tags, azdServiceName != '' ? {
    'azd-service-name': azdServiceName
  } : {})
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnvId
    configuration: {
      // No ingress — workers consume from the queue, not HTTP. This is
      // what lets the KEDA scale rule legitimately scale to 0.
      activeRevisionsMode: 'Single'
      registries: AZURE_CONTAINER_REGISTRY_ENDPOINT != '' ? [
        {
          server: AZURE_CONTAINER_REGISTRY_ENDPOINT
          username: AZURE_CONTAINER_REGISTRY_USERNAME
          passwordSecretRef: 'acr-password'
        }
      ] : []
      secrets: AZURE_CONTAINER_REGISTRY_ENDPOINT != '' ? [
        {
          name: 'acr-password'
          value: AZURE_CONTAINER_REGISTRY_PASSWORD
        }
      ] : []
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: DOCKER_IMAGE
          // Override the Dockerfile's API CMD so this container runs
          // the JobWorker bootstrap (issue 007 of the
          // project-generation-async-queue-cutover PRD). The API
          // container keeps its `fastapi run backend/main.py`
          // entrypoint untouched — same image, different processes.
          command: [
            'python'
          ]
          args: [
            '-m'
            'backend.worker_main'
          ]
          resources: {
            cpu: 1
            memory: '2Gi'
          }
          env: [
            {
              name: 'ROLE'
              value: 'worker'
            }
            {
              name: 'AI_FOUNDRY_ENDPOINT'
              value: AI_FOUNDRY_ENDPOINT
            }
            {
              name: 'LLM_DEPLOYMENT'
              value: LLM_DEPLOYMENT
            }
            {
              name: 'IMAGEGEN_DEPLOYMENT'
              value: IMAGEGEN_DEPLOYMENT
            }
            {
              name: 'IMAGEGEN_15_DEPLOYMENT'
              value: IMAGEGEN_15_DEPLOYMENT
            }
            {
              name: 'IMAGEGEN_1_MINI_DEPLOYMENT'
              value: IMAGEGEN_1_MINI_DEPLOYMENT
            }
            {
              name: 'SORA_DEPLOYMENT'
              value: SORA_DEPLOYMENT
            }
            {
              name: 'FLUX_KONTEXT_DEPLOYMENT'
              value: FLUX_KONTEXT_DEPLOYMENT
            }
            {
              name: 'AZURE_BLOB_SERVICE_URL'
              value: AZURE_BLOB_SERVICE_URL
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_NAME'
              value: AZURE_STORAGE_ACCOUNT_NAME
            }
            {
              name: 'AZURE_BLOB_IMAGE_CONTAINER'
              value: AZURE_BLOB_IMAGE_CONTAINER
            }
            {
              name: 'AZURE_STORAGE_QUEUE_URL'
              value: AZURE_STORAGE_QUEUE_URL
            }
            {
              name: 'JOB_QUEUE_NAME'
              value: JOB_QUEUE_NAME
            }
            {
              name: 'JOB_QUEUE_POISON_NAME'
              value: JOB_QUEUE_POISON_NAME
            }
            {
              name: 'CDN_BLOB_URL'
              value: CDN_BLOB_URL
            }
            {
              name: 'AZURE_COSMOS_DB_ENDPOINT'
              value: COSMOS_ENDPOINT
            }
            {
              name: 'AZURE_COSMOS_DB_ID'
              value: COSMOS_DATABASE_NAME
            }
            {
              name: 'AZURE_COSMOS_CONTAINER_ID'
              value: COSMOS_CONTAINER_NAME
            }
            {
              name: 'AZURE_CONTAINER_REGISTRY_ENDPOINT'
              value: AZURE_CONTAINER_REGISTRY_ENDPOINT
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            // KEDA azure-queue trigger authenticated via the system-
            // assigned managed identity. The trigger reads the message
            // count of `JOB_QUEUE_NAME` and adds one replica per
            // `queueLength` pending messages, capped at `maxReplicas`.
            //
            // The Bicep type definitions for `CustomScaleRule` predate
            // the `identity` property that the Container Apps API now
            // supports for managed-identity-authenticated KEDA triggers.
            // Disable the stale-type BCP037 warning rather than fall
            // back to a connection-string `auth` block — see PRD §
            // "Auth is managed-identity only — no connection strings".
            name: 'azure-queue-imagejobs'
            custom: {
              type: 'azure-queue'
              #disable-next-line BCP037
              identity: 'system'
              metadata: {
                accountName: AZURE_STORAGE_ACCOUNT_NAME
                queueName: JOB_QUEUE_NAME
                queueLength: string(queueLength)
                cloud: 'AzurePublicCloud'
              }
            }
          }
        ]
      }
    }
  }
}

output containerAppId string = workerApp.id
output containerAppPrincipalId string = deployNew ? workerApp.identity.principalId : ''
output containerAppName string = containerAppName

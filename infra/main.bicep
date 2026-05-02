// File: infra/main.bicep

// This Bicep template deploys the Visionary Lab infrastructure:
// - Azure AI Foundry (AIServices) with all model deployments
// - Azure Container App Environment with backend and frontend Container Apps
// - Azure Storage Account, Cosmos DB, Container Registry, Log Analytics
// - RBAC role assignments for managed identity access (no API keys)

@description('Location for all resources')
param location string = resourceGroup().location

// Parameters for the Container App Environment and Container Apps
@description('Name of the Container App Environment')
param containerAppEnvName string = 'cae-${environmentName}'
@description('Name of the Container App')
param containerAppNameBackend string = 'ca-backend-${environmentName}'
param containerAppNameFrontend string = 'ca-frontend-${environmentName}'
@description('Name of the worker Container App that consumes the persistent image-job queue (PRD § Infrastructure → KEDA worker).')
param containerAppNameWorker string = 'ca-worker-${environmentName}'
param logAnalyticsWorkspaceName string = 'log-${environmentName}'

@description('Unique name for the Storage Account (3-24 lowercase letters and numbers)')
param storageAccountName string = 'st${toLower(uniqueString(resourceGroup().id, environmentName))}'

@description('Unique name for the Container Registry (5-50 lowercase letters and numbers)')
param containerRegistryName string = 'cr${toLower(uniqueString(resourceGroup().id, environmentName))}'

// AI Foundry parameters
@description('Name of the AI Foundry resource')
param aiFoundryName string
@description('Name of the AI Foundry project')
param aiProjectName string = '${aiFoundryName}-proj'
@description('Location for AI Foundry (some models are region-specific)')
param aiFoundryLocation string = 'swedencentral'

// Model deployment names
@description('Name of the LLM deployment (Azure deployment names disallow ".", so gpt-5.4 maps to gpt-5-4)')
param LLM_DEPLOYMENT string = 'gpt-5-4'
@description('Name of the primary image generation deployment (gpt-image-2)')
param IMAGEGEN_DEPLOYMENT string = 'gpt-image-2'
@description('Name of the optional legacy image generation deployment (gpt-image-1.5 era)')
param IMAGEGEN_15_DEPLOYMENT string = ''
@description('Name of the gpt-image-1-mini deployment')
param IMAGEGEN_1_MINI_DEPLOYMENT string = ''
@description('Name of the Sora deployment')
param SORA_DEPLOYMENT string = 'sora-2'
@description('Name of the FLUX Kontext Pro deployment')
param FLUX_KONTEXT_DEPLOYMENT string = ''

// Model types and versions (for Bicep-managed deployments)
param llmModelType string = 'gpt-5.4'
param llmModelVersion string = '2026-03-05'
// Image/video models may be deployed via CLI when Bicep doesn't support the format
@description('Set to true to deploy image gen models via Bicep (requires OpenAI-format models). Defaults to true so that the primary gpt-image-2 deployment is created when IMAGEGEN_DEPLOYMENT is set.')
param deployImageGenModels bool = true
param imageGenModelType string = 'gpt-image-2'
param imageGenModelVersion string = '2026-04-21'
param imageGen15ModelVersion string = '2025-12-16'
param imageGen1MiniModelVersion string = '2025-10-06'
@description('Set to true to deploy Sora via Bicep')
param deploySoraModel bool = true
param soraModelVersion string = '2025-10-06'

// Docker images for the backend and frontend container apps
param DOCKER_IMAGE_BACKEND string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param DOCKER_IMAGE_FRONTEND string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
@description('Docker image for the queue-worker Container App. Defaults to the backend image — worker and backend share the same image and branch on the ROLE env var. azd populates this from SERVICE_WORKER_IMAGE_NAME when present.')
param DOCKER_IMAGE_WORKER string = ''
param API_PROTOCOL string = ''
param API_HOSTNAME string = ''
param API_PORT string = ''

// Easy Auth for frontend (Microsoft tenant restriction)
@secure()
param AUTH_CLIENT_ID string = ''
@secure()
param AUTH_CLIENT_SECRET string = ''
param AUTH_ISSUER string = ''

// Environment name for azd
param environmentName string = ''

// VNet parameters
@description('Name of the Virtual Network')
param vnetName string = 'vnet-${environmentName}'

// Front Door parameters
@description('Name of the Azure Front Door profile')
param frontDoorName string = 'afd-${environmentName}'

// Custom domain for frontend
@description('Custom domain name for the frontend (e.g., visionary.go-agentic.com). Leave empty to skip.')
param frontendCustomDomain string = ''
@description('Managed certificate resource ID for the custom domain. Leave empty for first-time setup.')
param frontendCertificateId string = ''

// Parameters for Cosmos DB
param cosmosAccountName string = 'cosmos-${environmentName}'
param cosmosDatabaseName string = 'VisionaryLabDB'
param cosmosContainerName string = 'visionarylab'

// ─── Local-dev IP allowlist ───
@description('Comma-separated list of client IP addresses or CIDR ranges to allow through the Cosmos DB and Storage Account firewalls (e.g., for running the backend locally against Azure resources). Leave empty to keep services reachable only via private endpoint. Example: "203.0.113.7,198.51.100.0/24".')
param allowedClientIpAddresses string = ''

var allowedClientIpAddressList = empty(allowedClientIpAddresses) ? [] : split(replace(allowedClientIpAddresses, ' ', ''), ',')

// ─── Azure Storage Account ───
module storageAccountMod './modules/storageAccount.bicep' = {
  name: 'storageAccountMod'
  params: {
    location: location
    storageAccountName: storageAccountName
    deployNew: true
    allowedIpAddresses: allowedClientIpAddressList
  }
}

module storageContainerMod './modules/storageAccountContainer.bicep' = {
  name: 'storageContainerMod'
  params: {
    storageAccountName: storageAccountName
    containerName: 'images'
    deployNew: true
  }
  dependsOn: [
    storageAccountMod
  ]
}

// ─── Azure Container Registry ───
module containerRegistryMod './modules/containerRegistry.bicep' = {
  name: 'containerRegistryMod'
  params: {
    location: location
    containerRegistryName: containerRegistryName
    deployNew: true
  }
}

// ─── Azure Cosmos DB ───
var cosmosPrefix = toLower(substring(uniqueString(resourceGroup().id, environmentName), 0, 5))
var cosmosAccountNamePrefixed = '${cosmosPrefix}-${cosmosAccountName}'
module cosmosDbMod './modules/cosmosDb.bicep' = {
  name: 'cosmosDbMod'
  params: {
    location: location
    cosmosAccountName: cosmosAccountNamePrefixed
    databaseName: cosmosDatabaseName
    containerName: cosmosContainerName
    subnetId: ''
    deployNew: true
    publicNetworkAccess: 'Disabled'
    allowedIpAddresses: allowedClientIpAddressList
  }
}

// ─── Virtual Network ───
module vnetMod './modules/virtualNetwork.bicep' = {
  name: 'vnetMod'
  params: {
    location: location
    vnetName: vnetName
  }
}

// ─── Private DNS Zones ───
module blobDnsZoneMod './modules/privateDnsZone.bicep' = {
  name: 'blobDnsZoneMod'
  params: {
    zoneName: 'privatelink.blob.core.windows.net'
    vnetId: vnetMod.outputs.vnetId
  }
}

module cosmosDnsZoneMod './modules/privateDnsZone.bicep' = {
  name: 'cosmosDnsZoneMod'
  params: {
    zoneName: 'privatelink.documents.azure.com'
    vnetId: vnetMod.outputs.vnetId
  }
}

// ─── Private Endpoints ───
module storagePrivateEndpointMod './modules/privateEndpoint.bicep' = {
  name: 'storagePrivateEndpointMod'
  params: {
    location: location
    privateEndpointName: 'pe-storage-${environmentName}'
    subnetId: vnetMod.outputs.privateEndpointsSubnetId
    privateLinkServiceId: storageAccountMod.outputs.storageAccountId
    groupIds: ['blob']
    privateDnsZoneId: blobDnsZoneMod.outputs.privateDnsZoneId
  }
}

module cosmosPrivateEndpointMod './modules/privateEndpoint.bicep' = {
  name: 'cosmosPrivateEndpointMod'
  params: {
    location: location
    privateEndpointName: 'pe-cosmos-${environmentName}'
    subnetId: vnetMod.outputs.privateEndpointsSubnetId
    privateLinkServiceId: cosmosDbMod.outputs.cosmosAccountId
    groupIds: ['Sql']
    privateDnsZoneId: cosmosDnsZoneMod.outputs.privateDnsZoneId
  }
}

// ─── Azure Front Door (CDN with private link to storage) ───
var storageHostName = replace(replace(storageAccountMod.outputs.storageAccountPrimaryEndpoint, 'https://', ''), '/', '')
module frontDoorMod './modules/frontDoor.bicep' = {
  name: 'frontDoorMod'
  params: {
    frontDoorName: frontDoorName
    storageAccountHostName: storageHostName
    storageAccountId: storageAccountMod.outputs.storageAccountId
    storageAccountLocation: location
  }
}

// ─── AI Foundry (replaces separate Azure OpenAI resources) ───
module aiFoundryMod './modules/aiFoundry.bicep' = {
  name: 'aiFoundryMod'
  params: {
    aiFoundryName: aiFoundryName
    location: aiFoundryLocation
    deployNew: true
  }
}

module aiFoundryProjectMod './modules/aiFoundryProject.bicep' = {
  name: 'aiFoundryProjectMod'
  params: {
    aiProjectName: aiProjectName
    aiFoundryName: aiFoundryName
    location: aiFoundryLocation
  }
  dependsOn: [
    aiFoundryMod
  ]
}

// ─── Model Deployments (under AI Foundry) ───
// Chained sequentially — Azure doesn't support parallel deployments on the same account
module llmDeployment './modules/aiFoundryModelDeployment.bicep' = {
  name: 'llmDeployment'
  params: {
    aiFoundryName: aiFoundryName
    deploymentName: LLM_DEPLOYMENT
    modelName: llmModelType
    modelVersion: llmModelVersion
    skuCapacity: 30
  }
  dependsOn: [
    aiFoundryMod
  ]
}

module imageGenDeployment './modules/aiFoundryModelDeployment.bicep' = if (deployImageGenModels && IMAGEGEN_DEPLOYMENT != '') {
  name: 'imageGenDeployment'
  params: {
    aiFoundryName: aiFoundryName
    deploymentName: IMAGEGEN_DEPLOYMENT
    modelName: imageGenModelType
    modelVersion: imageGenModelVersion
    skuCapacity: 1
  }
  dependsOn: [
    llmDeployment
  ]
}

module imageGen15Deployment './modules/aiFoundryModelDeployment.bicep' = if (deployImageGenModels && IMAGEGEN_15_DEPLOYMENT != '') {
  name: 'imageGen15Deployment'
  params: {
    aiFoundryName: aiFoundryName
    deploymentName: IMAGEGEN_15_DEPLOYMENT
    modelName: 'gpt-image-1.5'
    modelVersion: imageGen15ModelVersion
    skuCapacity: 1
  }
  dependsOn: [
    imageGenDeployment
  ]
}

module imageGen1MiniDeployment './modules/aiFoundryModelDeployment.bicep' = if (deployImageGenModels && IMAGEGEN_1_MINI_DEPLOYMENT != '') {
  name: 'imageGen1MiniDeployment'
  params: {
    aiFoundryName: aiFoundryName
    deploymentName: IMAGEGEN_1_MINI_DEPLOYMENT
    modelName: 'gpt-image-1-mini'
    modelVersion: imageGen1MiniModelVersion
    skuCapacity: 1
  }
  dependsOn: [
    imageGen15Deployment
  ]
}

module soraDeployment './modules/aiFoundryModelDeployment.bicep' = if (deploySoraModel && SORA_DEPLOYMENT != '') {
  name: 'soraDeployment'
  params: {
    aiFoundryName: aiFoundryName
    deploymentName: SORA_DEPLOYMENT
    modelName: 'sora-2'
    modelVersion: soraModelVersion
    skuCapacity: 1
    skuName: 'GlobalStandard'
  }
  dependsOn: [
    imageGen1MiniDeployment
  ]
}

// ─── Container App Environment ───
module containerAppEnvMod './modules/containerAppEnv.bicep' = {
  name: 'containerAppEnvMod'
  params: {
    location: location
    containerAppEnvName: containerAppEnvName
    logAnalyticsWorkspaceName: logAnalyticsWorkspaceName
    subnetId: vnetMod.outputs.containerAppsSubnetId
    deployNew: true
  }
}

// ─── Container App: Backend ───
module containerAppBackend './modules/containerApp.bicep' = {
  name: 'containerAppBackend'
  params: {
    location: location
    containerAppName: containerAppNameBackend
    containerAppEnvId: containerAppEnvMod.outputs.containerAppEnvId
    targetPort: 80
    deployNew: true
    AZURE_BLOB_SERVICE_URL: storageAccountMod.outputs.storageAccountPrimaryEndpoint
    AZURE_STORAGE_ACCOUNT_NAME: storageAccountName
    AZURE_BLOB_IMAGE_CONTAINER: 'images'
    CDN_BLOB_URL: 'https://${frontDoorMod.outputs.frontDoorEndpointHostName}'
    DOCKER_IMAGE: DOCKER_IMAGE_BACKEND
    AZURE_CONTAINER_REGISTRY_ENDPOINT: containerRegistryMod.outputs.containerRegistryLoginServer
    AZURE_CONTAINER_REGISTRY_USERNAME: containerRegistryMod.outputs.containerRegistryUsername
    AZURE_CONTAINER_REGISTRY_PASSWORD: containerRegistryMod.outputs.containerRegistryPassword
    AI_FOUNDRY_ENDPOINT: aiFoundryMod.outputs.aiFoundryEndpoint
    LLM_DEPLOYMENT: LLM_DEPLOYMENT
    IMAGEGEN_DEPLOYMENT: IMAGEGEN_DEPLOYMENT
    IMAGEGEN_15_DEPLOYMENT: IMAGEGEN_15_DEPLOYMENT
    IMAGEGEN_1_MINI_DEPLOYMENT: IMAGEGEN_1_MINI_DEPLOYMENT
    SORA_DEPLOYMENT: SORA_DEPLOYMENT
    FLUX_KONTEXT_DEPLOYMENT: FLUX_KONTEXT_DEPLOYMENT
    COSMOS_ENDPOINT: cosmosDbMod.outputs.cosmosAccountEndpoint
    COSMOS_DATABASE_NAME: cosmosDbMod.outputs.databaseName
    COSMOS_CONTAINER_NAME: cosmosDbMod.outputs.containerName
    azdServiceName: 'backend'
    // Pin the backend to a single replica. The in-process semaphore in
    // `ImagePipelineService` and the per-project asyncio lock in
    // `StagingPipeline` rely on a single replica for correctness. See
    // `prds/2026-04-29-parallel-processing-prd.md` (Single-replica
    // deployment constraint) before raising this.
    maxReplicas: 1
  }
}

// ─── Container App: Worker (queue consumer) ───
//
// Second Container App configured as the persistent image-job queue
// consumer described in
// `prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`
// (Infrastructure → KEDA worker; user stories 19, 38–40). Same image
// as the backend; differentiated at runtime via `ROLE=worker`. Scales
// 0..10 on the `imagejobs` queue depth via KEDA, with no ingress so
// scale-to-zero is honored at idle.
module containerAppWorker './modules/containerAppWorker.bicep' = {
  name: 'containerAppWorker'
  params: {
    location: location
    containerAppName: containerAppNameWorker
    containerAppEnvId: containerAppEnvMod.outputs.containerAppEnvId
    deployNew: true
    AZURE_BLOB_SERVICE_URL: storageAccountMod.outputs.storageAccountPrimaryEndpoint
    AZURE_STORAGE_ACCOUNT_NAME: storageAccountName
    AZURE_BLOB_IMAGE_CONTAINER: 'images'
    AZURE_STORAGE_QUEUE_URL: storageAccountMod.outputs.storageAccountPrimaryQueueEndpoint
    JOB_QUEUE_NAME: storageAccountMod.outputs.imageJobsQueueName
    JOB_QUEUE_POISON_NAME: storageAccountMod.outputs.imageJobsPoisonQueueName
    CDN_BLOB_URL: 'https://${frontDoorMod.outputs.frontDoorEndpointHostName}'
    DOCKER_IMAGE: DOCKER_IMAGE_WORKER == '' ? DOCKER_IMAGE_BACKEND : DOCKER_IMAGE_WORKER
    AZURE_CONTAINER_REGISTRY_ENDPOINT: containerRegistryMod.outputs.containerRegistryLoginServer
    AZURE_CONTAINER_REGISTRY_USERNAME: containerRegistryMod.outputs.containerRegistryUsername
    AZURE_CONTAINER_REGISTRY_PASSWORD: containerRegistryMod.outputs.containerRegistryPassword
    AI_FOUNDRY_ENDPOINT: aiFoundryMod.outputs.aiFoundryEndpoint
    LLM_DEPLOYMENT: LLM_DEPLOYMENT
    IMAGEGEN_DEPLOYMENT: IMAGEGEN_DEPLOYMENT
    IMAGEGEN_15_DEPLOYMENT: IMAGEGEN_15_DEPLOYMENT
    IMAGEGEN_1_MINI_DEPLOYMENT: IMAGEGEN_1_MINI_DEPLOYMENT
    SORA_DEPLOYMENT: SORA_DEPLOYMENT
    FLUX_KONTEXT_DEPLOYMENT: FLUX_KONTEXT_DEPLOYMENT
    COSMOS_ENDPOINT: cosmosDbMod.outputs.cosmosAccountEndpoint
    COSMOS_DATABASE_NAME: cosmosDbMod.outputs.databaseName
    COSMOS_CONTAINER_NAME: cosmosDbMod.outputs.containerName
    azdServiceName: 'worker'
  }
}

// ─── Container App: Frontend ───
module containerAppFrontend './modules/containerApp.bicep' = {
  name: 'containerAppFrontend'
  params: {
    location: location
    containerAppName: containerAppNameFrontend
    containerAppEnvId: containerAppEnvMod.outputs.containerAppEnvId
    targetPort: 3000
    deployNew: true
    AZURE_BLOB_SERVICE_URL: storageAccountMod.outputs.storageAccountPrimaryEndpoint
    AZURE_STORAGE_ACCOUNT_NAME: storageAccountName
    AZURE_BLOB_IMAGE_CONTAINER: 'images'
    CDN_BLOB_URL: 'https://${frontDoorMod.outputs.frontDoorEndpointHostName}'
    DOCKER_IMAGE: DOCKER_IMAGE_FRONTEND
    AZURE_CONTAINER_REGISTRY_ENDPOINT: containerRegistryMod.outputs.containerRegistryLoginServer
    AZURE_CONTAINER_REGISTRY_USERNAME: containerRegistryMod.outputs.containerRegistryUsername
    AZURE_CONTAINER_REGISTRY_PASSWORD: containerRegistryMod.outputs.containerRegistryPassword
    AI_FOUNDRY_ENDPOINT: aiFoundryMod.outputs.aiFoundryEndpoint
    LLM_DEPLOYMENT: LLM_DEPLOYMENT
    IMAGEGEN_DEPLOYMENT: IMAGEGEN_DEPLOYMENT
    IMAGEGEN_15_DEPLOYMENT: IMAGEGEN_15_DEPLOYMENT
    IMAGEGEN_1_MINI_DEPLOYMENT: IMAGEGEN_1_MINI_DEPLOYMENT
    SORA_DEPLOYMENT: SORA_DEPLOYMENT
    FLUX_KONTEXT_DEPLOYMENT: FLUX_KONTEXT_DEPLOYMENT
    API_PROTOCOL: API_PROTOCOL == '' ? 'https' : API_PROTOCOL
    API_PORT: API_PORT == '' ? '443' : API_PORT
    API_HOSTNAME: API_HOSTNAME == '' ? '${containerAppNameBackend}.${containerAppEnvMod.outputs.containerAppDefaultDomain}' : API_HOSTNAME
    enableAuth: AUTH_CLIENT_ID != ''
    authClientId: AUTH_CLIENT_ID
    authClientSecret: AUTH_CLIENT_SECRET
    authIssuer: AUTH_ISSUER
    customDomainName: frontendCustomDomain
    certificateId: frontendCertificateId
    azdServiceName: 'frontend'
  }
}

// ─── RBAC Role Assignments ───

module cosmosRoleAssignmentMod './modules/cosmosRoleAssignment.bicep' = {
  name: 'cosmosRoleAssignmentMod'
  params: {
    cosmosAccountName: cosmosAccountNamePrefixed
    containerAppPrincipalId: containerAppBackend.outputs.containerAppPrincipalId
    dataContributorRoleId: cosmosDbMod.outputs.dataContributorRoleId
  }
}

module aiFoundryRoleAssignmentMod './modules/aiFoundryRoleAssignment.bicep' = {
  name: 'aiFoundryRoleAssignmentMod'
  params: {
    aiFoundryId: aiFoundryMod.outputs.aiFoundryId
    aiFoundryName: aiFoundryName
    containerAppPrincipalId: containerAppBackend.outputs.containerAppPrincipalId
  }
}

module storageRoleAssignmentMod './modules/storageRoleAssignment.bicep' = {
  name: 'storageRoleAssignmentMod'
  params: {
    storageAccountName: storageAccountName
    containerAppPrincipalId: containerAppBackend.outputs.containerAppPrincipalId
  }
}

// Worker Container App needs the same Cosmos / AI Foundry / Storage
// (Blob + Queue) plane access as the backend so it can run the same
// `ImagePipelineService` against the persistent JobQueue.
module workerCosmosRoleAssignmentMod './modules/cosmosRoleAssignment.bicep' = {
  name: 'workerCosmosRoleAssignmentMod'
  params: {
    cosmosAccountName: cosmosAccountNamePrefixed
    containerAppPrincipalId: containerAppWorker.outputs.containerAppPrincipalId
    dataContributorRoleId: cosmosDbMod.outputs.dataContributorRoleId
  }
}

module workerAiFoundryRoleAssignmentMod './modules/aiFoundryRoleAssignment.bicep' = {
  name: 'workerAiFoundryRoleAssignmentMod'
  params: {
    aiFoundryId: aiFoundryMod.outputs.aiFoundryId
    aiFoundryName: aiFoundryName
    containerAppPrincipalId: containerAppWorker.outputs.containerAppPrincipalId
  }
}

module workerStorageRoleAssignmentMod './modules/storageRoleAssignment.bicep' = {
  name: 'workerStorageRoleAssignmentMod'
  params: {
    storageAccountName: storageAccountName
    containerAppPrincipalId: containerAppWorker.outputs.containerAppPrincipalId
  }
}

// ─── Outputs ───
output AZURE_LOCATION string = location
output AZURE_CONTAINER_ENVIRONMENT_NAME string = containerAppEnvMod.outputs.containerAppEnvId
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistryMod.outputs.containerRegistryLoginServer
output BACKEND_URI string = 'https://${containerAppBackend.outputs.containerAppFqdn}'
output BACKEND_INTERNAL_URI string = 'https://${containerAppBackend.outputs.containerAppFqdn}'
output FRONTEND_URI string = 'https://${containerAppFrontend.outputs.containerAppFqdn}'
output AZURE_STORAGE_ACCOUNT_NAME string = storageAccountName
output AZURE_BLOB_SERVICE_URL string = storageAccountMod.outputs.storageAccountPrimaryEndpoint
output AZURE_STORAGE_QUEUE_URL string = storageAccountMod.outputs.storageAccountPrimaryQueueEndpoint
output JOB_QUEUE_NAME string = storageAccountMod.outputs.imageJobsQueueName
output JOB_QUEUE_POISON_NAME string = storageAccountMod.outputs.imageJobsPoisonQueueName
output WORKER_CONTAINER_APP_NAME string = containerAppWorker.outputs.containerAppName
output AI_FOUNDRY_ENDPOINT string = aiFoundryMod.outputs.aiFoundryEndpoint
output AI_FOUNDRY_NAME string = aiFoundryName
output COSMOS_DB_ENDPOINT string = cosmosDbMod.outputs.cosmosAccountEndpoint
output COSMOS_DB_DATABASE_NAME string = cosmosDbMod.outputs.databaseName
output COSMOS_DB_CONTAINER_NAME string = cosmosDbMod.outputs.containerName
output CDN_BLOB_URL string = 'https://${frontDoorMod.outputs.frontDoorEndpointHostName}'

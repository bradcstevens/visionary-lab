// Role assignments for Storage Account: grants Container App managed identity
// access to Blob Storage (read/write/delete + SAS delegation) and to the
// Storage Queue Data plane used by the persistent JobQueue.

@description('Name of the Storage Account')
param storageAccountName string

@description('Principal ID of the Container App managed identity')
param containerAppPrincipalId string

// Built-in role definition IDs
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var storageBlobDelegatorRoleId = 'db58b8e5-c6ad-4a2a-8342-4190687cbf4a'
// Storage Queue Data Contributor — peek/get/add/update/delete queue
// messages. Required so the JobQueue (and the JobWorker dispatching from
// it) can authenticate via managed identity instead of a shared key /
// connection string. See PRD § "Auth is managed-identity only — no
// connection strings".
var storageQueueDataContributorRoleId = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

// Storage Blob Data Contributor — read/write/delete blobs
resource blobContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, containerAppPrincipalId, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributorRoleId)
    principalId: containerAppPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// Storage Blob Delegator — required for generating User Delegation SAS tokens
resource blobDelegatorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, containerAppPrincipalId, storageBlobDelegatorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDelegatorRoleId)
    principalId: containerAppPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// Storage Queue Data Contributor — required for the JobQueue producer
// (REST endpoints) and the JobWorker consumer to peek/dequeue/complete/
// abandon messages on `imagejobs` and `imagejobs-poison` using managed
// identity.
resource queueContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, containerAppPrincipalId, storageQueueDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageQueueDataContributorRoleId)
    principalId: containerAppPrincipalId
    principalType: 'ServicePrincipal'
  }
}

output blobContributorRoleAssignmentId string = blobContributorRole.id
output blobDelegatorRoleAssignmentId string = blobDelegatorRole.id
output queueContributorRoleAssignmentId string = queueContributorRole.id

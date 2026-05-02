param location string
param storageAccountName string = 'a${toLower(uniqueString(resourceGroup().id, 'storage'))}'
// param keyVaultName string
param deployNew bool = true

@description('Optional list of client IP addresses or CIDR ranges allowed through the storage firewall (e.g., for local development). When empty, the account stays fully private and is reachable only via private endpoint.')
param allowedIpAddresses array = []

@description('Resource tags merged onto the storage account. The SecurityControl=Ignore tag exempts the account from the tenant-level MCAPS local-auth governance policies.')
param tags object = {
  SecurityControl: 'Ignore'
}

var hasIpAllowList = !empty(allowedIpAddresses)
var ipRules = [for ip in allowedIpAddresses: {
  value: ip
  action: 'Allow'
}]

resource storageAccount 'Microsoft.Storage/storageAccounts@2024-01-01' = if(deployNew) {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    publicNetworkAccess: hasIpAllowList ? 'Enabled' : 'Disabled'
    allowBlobPublicAccess: false
    networkAcls: {
      defaultAction: 'Deny'
      bypass: hasIpAllowList ? 'AzureServices' : 'None'
      ipRules: ipRules
    }
  }
}

// ─── Storage Queue service + persistent job queue + poison queue ───
//
// Backs the persistent image-job queue described in
// `prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`
// (Infrastructure → Storage Queue + KEDA worker). The two queues are:
//
//   - `imagejobs`         — primary work queue for the JobWorker.
//   - `imagejobs-poison`  — receives messages whose dequeue count exceeds
//                            the JobQueue's max-dequeue policy (3) so a
//                            poisoned payload can never wedge the worker.
//
// Queue sub-resources are NOT taggable in ARM; the SecurityControl=Ignore
// tag lives on the parent storage account above and is inherited by the
// MCAPS exemption scope.
resource queueServices 'Microsoft.Storage/storageAccounts/queueServices@2024-01-01' = if (deployNew) {
  parent: storageAccount
  name: 'default'
  properties: {}
}

resource imageJobsQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2024-01-01' = if (deployNew) {
  parent: queueServices
  name: 'imagejobs'
  properties: {
    metadata: {}
  }
}

resource imageJobsPoisonQueue 'Microsoft.Storage/storageAccounts/queueServices/queues@2024-01-01' = if (deployNew) {
  parent: queueServices
  name: 'imagejobs-poison'
  properties: {
    metadata: {}
  }
}

output storageAccountPrimaryEndpoint string = storageAccount.properties.primaryEndpoints.blob
output storageAccountPrimaryQueueEndpoint string = storageAccount.properties.primaryEndpoints.queue
output storageAccountId string = storageAccount.id
output storageAccountName string = storageAccount.name
output imageJobsQueueName string = imageJobsQueue.name
output imageJobsPoisonQueueName string = imageJobsPoisonQueue.name


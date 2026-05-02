## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

Infrastructure foundation for the persistent job queue. Add Storage Queue
service to the existing storage account with `imagejobs` and
`imagejobs-poison` queues, a second Container App template variant
configured as the queue worker (`ROLE=worker` env var, KEDA `azure-queue`
scale rule with min=0, max=10, queueLength=5), and a role assignment
granting the workload identity `Storage Queue Data Contributor`. All
new resources tagged `SecurityControl=Ignore`. Foundry resources keep
`disableLocalAuth: false`. Auth is managed-identity only — no connection
strings.

See PRD sections "Infrastructure (Bicep)" and "Cross-cutting" user
stories 38–40.

## Acceptance criteria

- [ ] `infra/modules/storageAccount.bicep` provisions `queueServices` plus `imagejobs` and `imagejobs-poison` queues
- [ ] `infra/modules/containerApp.bicep` (or a sibling worker variant) deploys a second app with `ROLE=worker` and a KEDA `azure-queue` scale rule (min=0, max=10, queueLength=5)
- [ ] `infra/modules/storageRoleAssignment.bicep` assigns `Storage Queue Data Contributor` to the workload identity
- [ ] All new resources carry `SecurityControl=Ignore` tag
- [ ] `azd up` deploys cleanly to a fresh environment and the worker app reaches a Running revision with zero replicas at idle
- [ ] `DEPLOYMENT.md` documents the queue feature-flag toggle procedure and rolling-deploy drain window

## Blocked by

None - can start immediately.

## User stories addressed

- User story 19
- User story 38
- User story 39
- User story 40

# Agents

## Skills & Documentation

- Check available skills before any action. Announce which skill you are using.
- Use `microsoft-docs` and Azure-related skills for latest Microsoft best practices.
- Use Context7 MCP for library/repo documentation before falling back to web search.

## Local Testing (Required Before Every Commit)

Run the full test suite locally after every feature or component change:

- **Backend:** `uv run pytest tests/ --ignore=tests/integration -v`
- **Frontend:** `cd frontend && npx playwright test`
- **Build:** `cd frontend && npm run build`
- **Lint:** `cd frontend && npx next lint`

Playwright reports must include screenshots, multi-browser coverage, and pass/fail results. Save reports to `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.

Only commit and push when all local tests pass.

## CI/CD (Triggered on Push)

GitHub Actions runs on every push and must:

1. Deploy via `azd up`
2. Run full Playwright E2E suite against the deployed environment (using `tests/projects/` scenario data)
3. Tear down all resources if any step fails
4. Tear down after all tests pass on success
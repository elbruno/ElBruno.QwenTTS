# Issue and PR Validation Runbook (Release Cycle)

Last validated: 2026-08-02

## Current GitHub state (this cycle)

- Open issues: `#62` — *feat: Add ElBruno.QwenTTS.BlazorComponents — Blazor Razor Class Library for TTS and voice cloning UI*
- Open PRs: none

Commands used:

```bash
gh issue list --state open --limit 20
gh pr list --state open --limit 20
gh issue view 62 --json number,title,state,labels,assignees,url,updatedAt,author,body
```

## 1) Validate open issues are still valid and in scope

For each open issue:

1. Confirm status is still `OPEN` and that no newer issue supersedes it.
2. Confirm scope aligns with current release goals and repository architecture.
3. Confirm acceptance criteria are concrete and testable.
4. Confirm deliverables map to real repo artifacts (project, docs, workflow updates).
5. Add/update a checklist comment so progress is trackable.

`#62` scope check (current cycle): **in scope** for UI/package expansion and release planning.

## 2) Validate PR merge readiness

When a PR exists, validate all of the following:

### Scope and quality gates

- PR is linked to a valid open issue (for this cycle, issue `#62`).
- Changes are limited to requested scope (no unrelated refactors).
- Documentation updated where feature/API behavior changes.

### .NET build/test/package gates (repository-aligned)

- `dotnet build` passes for solution projects.
- `dotnet test` passes (`ElBruno.QwenTTS.Core.Tests`).
- If packageable changes are included, `dotnet pack src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj -c Release -o artifacts` succeeds.

### Release workflow gates

- `.github/workflows/publish.yml` remains compatible with release flow:
  - release tag/manual dispatch versioning is preserved
  - NuGet OIDC login step remains valid
  - package metadata/version inputs are correct

## 3) No open PRs yet (current case)

Until the first PR opens:

1. Keep issue `#62` as the active work anchor.
2. Track progress through an issue checklist comment (deliverable-based).
3. Require first PR to explicitly include:
   - linked issue reference (`#62`)
   - evidence for build/test/package gates
   - docs updates for new Blazor components/package usage

## 4) Merge readiness decision for this cycle

Current decision: **Not yet merge-ready** (no PR submitted).  
Readiness status: **Issue validation complete; PR validation pending first PR.**

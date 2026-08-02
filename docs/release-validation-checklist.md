# Release Validation Checklist

Use this checklist for issue closure, PR readiness, and release gates.

## 1) Issue validation

- Link code/docs changes to an issue (or explicitly document why no issue is needed).
- Confirm acceptance criteria are met.
- Ensure impacted docs are updated (`README.md`, `docs/*`, `CHANGELOG.md` as needed).
- Run targeted validation for changed areas (`dotnet build`, targeted `dotnet test`).

## 2) PR validation

- PR description includes scope, risk, and validation evidence.
- Required CI checks pass.
- Reviewer confirms no unrelated files were changed.
- If release-facing, reviewer verifies `README.md` and `CHANGELOG.md` alignment.

### Current-state behavior: no open PRs

When there are no open PRs, treat the repository as release-candidate-from-`main`:

- Validate the latest `main` commit directly (build/tests/docs sanity).
- Record "No open PR validation items" in release notes/checklist tracking.
- Continue only if `main` is green and release docs are current.

## 3) Merge-to-release gates

- Confirm merge commit is on `main`.
- Confirm version source for publish:
  - **GitHub Release trigger**: tag-driven (`vX.Y.Z` → workflow strips `v` and publishes `X.Y.Z`).
  - **Manual dispatch**: optional `version` input (also normalized by stripping leading `v`/`.`).
  - **Fallback**: if no input, version from `src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj`.
- Confirm semantic version format is valid (workflow enforces SemVer regex).
- Ensure publish workflow gates pass: restore, release build, tests, pack core + voice cloning, OIDC NuGet login, push.
- Confirm NuGet artifacts expected for the release are present.

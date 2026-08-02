# What's New Policy

This repository keeps the `README.md` **What's New** section concise and release-focused.

## Rules

1. Keep only the **latest 5 important features** in `README.md` → `## What's New`.
2. Entries should be short, user-visible, and non-duplicative.
3. Remove or roll up older items when adding a new important feature.
4. On every NuGet release, validate whether a new "What's New" entry is needed.
5. If no new important feature is shipped, keep the existing list unchanged.

## Release check

During release validation, explicitly answer:

- "Does this release add an important user-facing feature?"
  - **Yes**: add/update an entry and keep the list at 5 items max.
  - **No**: document "No README What's New change required."

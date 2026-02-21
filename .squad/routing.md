# Work Routing

How to decide who handles what.

## Routing Table

| Work Type | Route To | Examples |
|-----------|----------|----------|
| Architecture, scope, decisions | Morpheus | Model selection, pipeline design, ONNX export strategy |
| Python, ML, ONNX export | Trinity | PyTorch model analysis, ONNX conversion, tokenizer export |
| C#, .NET, ONNX Runtime | Neo | Console app, ONNX inference, BPE tokenizer in C# |
| Code review | Morpheus | Review PRs, check quality, architecture alignment |
| Testing, validation | Tank | Parity tests, edge cases, cross-language validation |
| Scope & priorities | Morpheus | What to build next, trade-offs, decisions |
| Session logging | Scribe | Automatic — never needs routing |

## Rules

1. **Eager by default** — spawn all agents who could usefully start work, including anticipatory downstream work.
2. **Scribe always runs** after substantial work, always as `mode: "background"`. Never blocks.
3. **Quick facts → coordinator answers directly.** Don't spawn an agent for simple questions.
4. **When two agents could handle it**, pick the one whose domain is the primary concern.
5. **"Team, ..." → fan-out.** Spawn all relevant agents in parallel as `mode: "background"`.
6. **Anticipate downstream work.** If Trinity exports ONNX, spawn Neo to scaffold the C# project simultaneously.

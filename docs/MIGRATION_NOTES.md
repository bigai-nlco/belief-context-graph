# Migration notes

## Design direction

The documentation no longer tries to make every page look like Fern's default
site. Instead, Fern is used for navigation, search, MDX rendering, and the
generated HTTP API reference.

The BCG brand remains responsible for the visual language.

## Navigation before / after

### First Fern revision

```text
Documentation
  Start here
  Key concepts
  Build with BCG
  Inspect and evaluate
  Construction modes
  Operations
  Reference

SDK Reference
  Graph
    many individual method pages
  Memory
    ...
  Runner
    ...
```

Most sections were expanded, so the sidebar appeared much longer than it was
conceptually.

### Current revision

```text
Documentation
  Start here                    [expand/collapse]
  Understand BCG                [expand/collapse]
  Build and integrate           [expand/collapse]
  Inspect and operate           [expand/collapse]
  Reference                     [expand/collapse]

SDK Reference
  SDK Reference                 [expand/collapse]
  Graph                         [expand/collapse]
  Memory                        [expand/collapse]
  Runner                        [expand/collapse]
  Model client                  [expand/collapse]
  Configuration and types       [expand/collapse]
```

Related Python methods are consolidated into task pages rather than appearing
as dozens of independent sidebar entries.

## SDK Reference is intentionally not the HTTP API

BCG currently has two distinct application surfaces:

1. **Python SDK**
   - in-process classes such as `BCG`, `BCGMemory`, and `BCGRunner`;
   - Python method signatures and Python return types;
   - no HTTP transport is required.

2. **HTTP construction API**
   - a running server;
   - routes such as `/turn`, `/graph`, and `/finalize`;
   - JSON request/response contracts;
   - language-neutral HTTP access.

The navigation can look similar, but the contract documented on each page is
different.

If BCG later ships a generated client SDK that is only a wrapper around those
HTTP routes, the HTTP endpoint pages can also display generated SDK snippets
in the same style as products such as Zep.

## Theme mapping

The current CSS ports the original BCG palette and interaction states:

- brick red = links, headings, primary action;
- teal = current/selected navigation;
- blue = user/info semantics;
- pink = assistant semantics;
- green = tool/support semantics;
- purple = decision semantics;
- orange = warning semantics.

This preserves the multi-color visual grammar from the original static docs.

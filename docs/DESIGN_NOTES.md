# Design notes

## Why Fern is still useful here

Fern provides the parts that were expensive to maintain manually:

- OpenAPI-driven HTTP reference pages;
- request/response schemas;
- code snippets and API Explorer;
- search;
- responsive navigation;
- collapsible sidebar sections;
- MDX content components.

The custom CSS deliberately overrides the generic Fern look so those
capabilities sit inside the BCG brand rather than replacing it.

## Why the SDK is curated manually

Fern can generate Python library documentation, but BCG's public docs need more
than symbol extraction:

- internal names should be translated into generalized user-facing wording;
- related methods should be grouped by user intent;
- compatibility behaviors need explicit warnings;
- current implementation limitations such as substring search and compatibility
  parameters need explanation.

For that reason, the public SDK Reference is hand-curated and compact.

## CSS ownership

The public visual contract lives in:

```text
fern/styles.css
```

Prefer changing the `--bcg-*` variables first. Use Fern selectors only for
component-level adjustments.

The site-level Fern color settings in `docs.yml` are intentionally aligned with
the same palette so generated API components inherit compatible surfaces.

---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for xradar, with links to the rest
      of the site.
html_theme.sidebar_secondary.remove: true
---

**Release:** {{release}}\
**Date:** {{today}}

```{include} ../README.md
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
contributing
authors
history
reference
```

Indices and tables
==================
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

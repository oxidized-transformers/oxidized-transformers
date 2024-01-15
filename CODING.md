# Coding guidelines

## Module `forward` signature

In modules that do not implement the `Module` or ModuleT` traits, we use the
following arguments order:

- The primary input argument (e.g. hidden representations or piece
  identifiers).
- Other input arguments in alphabetical order.
- Option arguments in alphabetical order.

If there is no primary input argument, all input arguments are ordered
alphabetically.

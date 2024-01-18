# How to write documentations in MASE

## The sphinx flow

Documentations are automatically generated from the [docstring](https://peps.python.org/pep-0257/) from each function using [sphinx](https://www.sphinx-doc.org/en/master/).

You can easily generate a local copy of the documentation website for testing whether your doc has been written correctly.

```bash
machop/sphinx_docs
make html
```
This would then build a doc directory under `sphinx_docs/build`, and you can open `sphinx_docs/build/html` and open the `index.html` page to have a local version of the doc.


## Automatic documentation generation through docstrings


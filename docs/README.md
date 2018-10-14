# How to build api documents in local environment

## Prerequisites
Make sure you have already installed sphinx and (rtd_theme)

```python
pip install sphinx sphinx_rtd_theme sphinx-autobuild
```

## Building full api documents

```python
sphinx-apidoc -f -o ./_build/docs ../researchutils
```

## Checking api documents locally with your browser

```python
sphinx-autobuild . _build/html
```

After typing above, if build is successful, below message will be displayed in your terminal

```sh
| The HTML pages are in _build/html.
+--------------------------------------------------------------------------------

[I 181014 15:57:56 server:292] Serving on http://127.0.0.1:8000
```

Check http://127.0.0.1:8000 with your browser
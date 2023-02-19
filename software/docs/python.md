# Python Coding Styles

## Why?

Having a good coding style is somehow important, since it makes your code less error-prone, and also makes collaboration a lot easier.
The general guidance is the [yapf standard](https://github.com/google/yapf).
However, the PEP8 is a very long document, this readme serves as a quick 'rule of thumbs' for python coding styles.

## Naming

```python
# In this project, we follow the habit of CapitalTheFirstLetter for class names
class DummyClass(...):
	def _my_priviate_method():
		# _ is a good sign for people to know that this method is only used inside this class
		...
	def my_public_method():
```

```python
# In this project, we follow the habit of break_words_by_underscore for function and variable names names
def my_function_name:
	...

# however, we generally would prefer shorter names!
my_old_school_variable = 13
```

## Line breaks

```bash
# long lines should generally be avoided
# bad
xs = ['my', 'line', 'is', 'very', 'very', 'very', 'long']
# good
xs = [
	'my', 'line', 'is', 
	'very', 'very', 
	'very', 'long']

# same thing applies to dicts, strings ....
```


## VSCode settings

The following settings in VScode can ensure auto-linting for Python.
You can find how to create a `settings.json` for your workspace in [here](https://code.visualstudio.com/docs/getstarted/settings).

The following setup would trigger a automatic formatting when you save your `.py` files.

```json
 "python.linting.enabled": true,
 "python.linting.pylintPath": "pylint",
 "editor.formatOnSave": true,
 "python.formatting.provider": "yapf",
 "python.linting.pylintEnabled": true,
```
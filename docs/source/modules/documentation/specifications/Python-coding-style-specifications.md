# Python Coding Style Specifications

## Why?

Having a good coding style is somehow important, since it makes your code less error-prone, and also makes collaboration a lot easier.
The general guidance is the [PEP8 standard](https://peps.python.org/pep-0008/).
However, the PEP8 is a very long document, this readme serves as a quick 'rule of thumbs' for python coding styles.

## Naming

```python
# In this project, we follow the habit of CapitalTheFirstLetter for class names
class DummyClass(...):
    def _my_priviate_method():
        # _ is a good sign for people to know that this method is only used inside this class
        ...
    def my_public_method():
        ...
```

```python
# In this project, we follow the habit of break_words_by_underscore for function and variable names names
def my_function_name:
    ...

# However, we generally would prefer shorter names!
my_old_school_variable = 13
```

## Line breaks

Line breaks are usually automatically handled by the pre-installed tool `black`.
To format a specific python file:
```shell
black xxx.py
```

Here is an example: 
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

## Using Black in MASE

[Black](https://github.com/psf/black) is the uncompromising Python code formatter, and has been used in many open-source projects. 
If you installed the packages for MASE, black is then included. It is crucial to format your code using `black` before you make a pull request in MASE. Our CIs would check for formatting, and your PR will not pass this check if code is not formatted.

Example on using black for format the whole directory:
```bash
# all .py files should be in chop
cd machop/chop
black *
```
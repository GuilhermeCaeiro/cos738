# Execution

In order to run this code, everything can be run at once by running:

```console
python run_all.py
```

Alternatively, each module can be executed individually. To achive that, you must first import the proper class, like in the following example:

```python
from src.inverted_list_generator import InvertedListGenerator
```

And then instantiate it and call its "run()" method:

```python
InvertedListGenerator().run()
```

Examples of that can be found in the file run_all.py.

# Dependencies

Apart from the libraries included in Python's default library, this code also relies on "numpy", "nltk" and "unidecode" for a series of tasks. In order to install those libraries you can run:

```console
pip install numpy nltk unidecode
```


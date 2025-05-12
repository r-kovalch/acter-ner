from setuptools import setup, find_packages

setup(
    name="acter_gliner",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.6.0",
        "spacy-transformers",
        "gliner>=0.4.4",
        "torch",
        "thinc",
    ],
    entry_points={
        "spacy_factories": [
            "gliner = spacy_trainable_gliner.gliner_component:make_gliner"
        ],
    },
)

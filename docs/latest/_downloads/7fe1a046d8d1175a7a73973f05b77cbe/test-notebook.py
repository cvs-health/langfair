"""
.. _adversarial_Example_1:

===============================================================
Basics & Model Specification of `AdversarialFairnessClassifier`
===============================================================
"""

import langfair


def main():
    print("Hello, world!")
    print(f"langfair version: {langfair.__version__}")
    print(f"langfair path: {langfair.__path__}")
    print(f"langfair file: {langfair.__file__}")
    print(f"langfair doc: {langfair.__doc__}")
    print(f"langfair author: {langfair.__author__}")
    print(f"langfair license: {langfair.__license__}")


# %%
# As with other machine learning methods, it is wise to take a train-test split
# of the data in order to validate the model on unseen data:

if __name__ == "__main__":
    main()

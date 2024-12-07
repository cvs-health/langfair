"""
.. _adversarial_Example_1:

===============================================================
Test Notebook
===============================================================
"""

import langfair


def main():
    print("Hello, world!")
    print(f"langfair version: {langfair.__version__}")
    print(f"langfair path: {langfair.__path__}")


# %%
# As with other machine learning methods, it is wise to take a train-test split
# of the data in order to validate the model on unseen data:
# :fa:`check`


if __name__ == "__main__":
    main()

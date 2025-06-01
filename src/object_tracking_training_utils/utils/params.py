import json

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        """
        Initializes the Params instance by loading parameters from the specified JSON file.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Saves current parameters to a JSON file.

        Args:
            json_path (str): Path to the file where parameters will be saved.
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Updates current parameters from another JSON file.

        Args:
            json_path (str): Path to the JSON file containing new parameters.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update_with_dict(self, dictio):
        """
        Updates current parameters using a Python dictionary.

        Args:
            dictio (dict): Dictionary containing key-value pairs of parameters to update.
        """
        self.__dict__.update(dictio)

    @property
    def dict(self):
        """
        Provides dictionary-style access to parameters.

        Returns:
            dict: The current parameters stored in the instance.
        """
        return self.__dict__

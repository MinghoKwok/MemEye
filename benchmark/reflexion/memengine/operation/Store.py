from abc import ABC, abstractmethod


def __store_convert_str_to_observation__(method):
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {"text": observation})
        return method(self, observation)

    return wrapper


class BaseStore(ABC):
    def __init__(self, config):
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, observation):
        pass


class FUMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.storage = kwargs["storage"]

    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        self.storage.add(observation)

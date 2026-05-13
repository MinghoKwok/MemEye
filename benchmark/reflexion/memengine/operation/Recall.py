from abc import ABC, abstractmethod

from ..function.Truncation import LMTruncation
from ..function.Utilization import ConcateUtilization


def __recall_convert_str_to_observation__(method):
    """
    If the input is a string, convert it to the dict form.
    """

    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {"text": observation})
        return method(self, observation)

    return wrapper


class BaseRecall(ABC):
    def __init__(self, config):
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, query):
        pass


class FUMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.storage = kwargs["storage"]
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        memory_context = self.utilization(self.storage.get_all_memory_in_order())
        return self.truncation(memory_context)


class RFMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.storage = kwargs["storage"]
        self.insight = kwargs["insight"]
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            if self.insight["global_insight"]:
                memory_context = self.utilization({"Insight": self.insight["global_insight"]})
            else:
                return self.config.empty_memory
        else:
            if self.insight["global_insight"]:
                memory_context = self.utilization(
                    {
                        "Insight": self.insight["global_insight"],
                        "Memory": self.storage.get_all_memory_in_order(),
                    }
                )
            else:
                memory_context = self.utilization(self.storage.get_all_memory_in_order())
        return self.truncation(memory_context)

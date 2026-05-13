from abc import ABC, abstractmethod

from ..function.LLM import APILLM


def _format_prompt(config, input_dict):
    return str(config.template).format(**input_dict)

class BaseReflector(ABC):
    """
    Draw new insights in higher level from existing information, commonly for reflection and optimization operations.
    """
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def generate_insight(self, *args, **kwargs):
        pass

class TrialReflector(BaseReflector):
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)
    
    def generate_insight(self, input_dict):
        prompt = _format_prompt(self.config.prompt, input_dict)
        res = self.llm.fast_run(prompt)

        return res

import os
import random
from abc import ABC, abstractmethod

import os

base_dir = os.path.dirname(__file__)  # 当前 .py 文件所在目录

decision_map = {
    1: open(os.path.join(base_dir, "prompt_template/decision/decision_making_1.md")).read(),
    2: open(os.path.join(base_dir, "prompt_template/decision/decision_making_2.md")).read(),
    3: open(os.path.join(base_dir, "prompt_template/decision/decision_making_3.md")).read(),
}



class PromptTemplate(ABC):
    def __init__(self) -> None:
        self.prompt_dict = self.load_prompt_dict()

    @abstractmethod
    def load_prompt_dict(self):
        raise NotImplementedError("Subclasses must implement load_prompt_dict()")

    def get_random_prompt_template(self) -> str:
        return self.prompt_dict[random.choice(list(self.prompt_dict.keys()))]

    @abstractmethod
    def reformat_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement reformat_prompt()")

class DecisionPrompt(PromptTemplate):
    def __init__(self) -> None:
        super().__init__()

    def load_prompt_dict(self):
        return decision_map

    def reformat_prompt(self, **kwargs) -> str:
        """
        :param kwargs: expected to contain current_analysis, current_rate, meeting_date
        :return: formatted prompt string
        """
        prompt_template = self.get_random_prompt_template()
        return prompt_template.format(
            current_analysis=kwargs.get("current_analysis", ""),
            current_rate=kwargs.get("current_rate", ""),
            meeting_date=kwargs.get("meeting_date", "")
        )

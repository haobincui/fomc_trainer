
import os
import random
from abc import ABC, abstractmethod

decision_map = {
    1: open("../prompt_template/decision_making_with_choice_1.md").read(),
    2: open("../prompt_template/decision_making_with_choice_2.md").read(),
    3: open("../prompt_template/decision_making_with_choice_3.md").read(),
}


class PromptTemplate(ABC):
    def __init__(self) -> None:
        self.prompt_dict = self.load_prompt_dict()

    @abstractmethod
    def load_prompt_dict(self):
        raise NotImplemented
        

    def get_random_prompt_template(self) -> str:
        return self.prompt_dict[random.choice(list(self.prompt_dict.keys()))]

    
    @abstractmethod
    def reformat_prompt(self, **kwargs):
        raise NotImplemented
    

class DecisionPrompt(PromptTemplate):
    pass

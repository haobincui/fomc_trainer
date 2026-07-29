import os
import random
from abc import ABC, abstractmethod

base_dir = os.path.dirname(__file__)  # 当前 .py 文件所在目录

decision_map = {
    1: open(os.path.join(base_dir, "prompt_template/decision/decision_making_1.md")).read(),
    2: open(os.path.join(base_dir, "prompt_template/decision/decision_making_2.md")).read(),
    3: open(os.path.join(base_dir, "prompt_template/decision/decision_making_3.md")).read(),
}



class PromptTemplate(ABC):
    def __init__(self, seed: int | None = None, template_id: int | None = None) -> None:
        self.prompt_dict = self.load_prompt_dict()
        self._rng = random.Random(seed)
        self.default_template_id = template_id
        self.last_template_id: int | None = None

    @abstractmethod
    def load_prompt_dict(self):
        raise NotImplementedError("Subclasses must implement load_prompt_dict()")

    def get_random_prompt_template(self) -> str:
        template_id = self._rng.choice(list(self.prompt_dict.keys()))
        self.last_template_id = template_id
        return self.prompt_dict[template_id]

    def get_prompt_template(self, template_id: int | None = None) -> str:
        selected_template_id = template_id if template_id is not None else self.default_template_id
        if selected_template_id is None:
            return self.get_random_prompt_template()
        if selected_template_id not in self.prompt_dict:
            raise ValueError(
                f"Template id '{selected_template_id}' not found. Available ids: {sorted(self.prompt_dict)}"
            )
        self.last_template_id = selected_template_id
        return self.prompt_dict[selected_template_id]

    @abstractmethod
    def reformat_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement reformat_prompt()")

class DecisionPrompt(PromptTemplate):
    def __init__(self, seed: int | None = None, template_id: int | None = None) -> None:
        super().__init__(seed=seed, template_id=template_id)

    def load_prompt_dict(self):
        return decision_map

    def reformat_prompt(self, **kwargs) -> str:
        """
        :param kwargs: expected to contain current_analysis, current_rate, meeting_date
        :return: formatted prompt string
        """
        prompt_template = self.get_prompt_template(kwargs.get("template_id"))
        return prompt_template.format(
            current_analysis=kwargs.get("current_analysis", ""),
            current_rate=kwargs.get("current_rate", ""),
            meeting_date=kwargs.get("meeting_date", "")
        )

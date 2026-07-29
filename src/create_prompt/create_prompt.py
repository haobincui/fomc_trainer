


from .prompt_template import DecisionPrompt


def create_decision_prompt(
    *,
    current_analysis: str,
    current_rate: str,
    meeting_date: str,
    template_id: int | None = None,
    seed: int | None = None,
) -> dict:
    prompt_builder = DecisionPrompt(seed=seed, template_id=template_id)
    prompt = prompt_builder.reformat_prompt(
        current_analysis=current_analysis,
        current_rate=current_rate,
        meeting_date=meeting_date,
        template_id=template_id,
    )
    return {
        "prompt": prompt,
        "prompt_template_id": prompt_builder.last_template_id,
    }

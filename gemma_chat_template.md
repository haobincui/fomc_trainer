2. Thinking Mode Configuration
Compared to Gemma 3, the models use standard system, assistant, and user roles. To properly manage the thinking process, use the following control tokens:

Trigger Thinking: Thinking is enabled by including the <|think|> token at the start of the system prompt. To disable thinking, remove the token.
Standard Generation: When thinking is enabled, the model will output its internal reasoning followed by the final answer using this structure:

<|channel>thought\n[Internal reasoning]<channel|>
Disabled Thinking Behavior: For all models except for the E2B and E4B variants, if thinking is disabled, the model will still generate the tags but with an empty thought block:

<|channel>thought\n<channel|>[Final answer]
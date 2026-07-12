"""Tool to switch which Claude model answers the conversation.

The choice lives on the ToolContext (not in config), so Claude.converse reads
it fresh each call and the switch takes effect from the very next reply — the
same tool loop that handles "switch to Opus" then generates its acknowledgement
with the new model. Resets to the default (Haiku, for latency) on restart.
"""

import config as cfg
from tools import tool


@tool({
    "name": "set_conversation_model",
    "description": (
        "Switch which Claude model powers this conversation. Call this when the "
        "user asks to change models — e.g. 'switch to Opus', 'use the smart "
        "model', 'this is hard, think harder', or 'go back to the fast one'. "
        "Options: 'haiku' is fastest and lowest-latency (the default); 'sonnet' "
        "reasons more strongly with a little more delay; 'opus' is the most "
        "capable but the slowest and most expensive. Map the user's words to the "
        "closest option (e.g. 'the smart/best model' -> opus, 'faster' -> haiku). "
        "The change takes effect immediately, for this reply onward."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["haiku", "sonnet", "opus"],
                "description": "Which model to switch to.",
            }
        },
        "required": ["model"],
    },
})
def set_conversation_model(ctx, args):
    choice = (args.get("model") or "").strip().lower()
    model_id = cfg.CONVO_MODELS.get(choice)
    if not model_id:
        options = ", ".join(cfg.CONVO_MODELS)
        return f"Unknown model '{choice}'. Choose one of: {options}."
    if ctx.convo_model == model_id:
        return f"Already using {cfg.convo_model_label(model_id)}."
    ctx.convo_model = model_id
    return f"Switched the conversation model to {cfg.convo_model_label(model_id)}."

"""Tools to read and switch which model answers the conversation.

The choice lives on the ToolContext (not in config), so Claude.converse reads
it fresh each call and the switch takes effect from the very next reply — the
same tool loop that handles "switch to Opus" then generates its acknowledgement
with the new model. Resets to the default (Haiku, for latency) on restart.

DeepSeek options are served through their Anthropic-compatible endpoint and
need DEEPSEEK_API_KEY in the environment; the switch is refused out loud when
the key is missing, because the alternative is a turn that dies mid-reply.

get_current_model exists because a model asked "which model are you?" answers
from conversation history, and that history is SHARED by every persona: it
carries switches other hats made and choices that have since changed. The
system prompt already states the truth every call, and the model still
overrode it with the history three times in one day — claiming Opus while on
Haiku, and DeepSeek while on Opus (session_2026-07-31.log 11:32, 11:48,
12:23). A tool turns the answer into a live read of the same expression
converse() routes the API call with, lands it at the END of the context where
it outweighs old turns, and leaves a tool_use line in the log so the answer is
auditable instead of merely plausible.
"""

import os

from brain import agents
import config as cfg
from tools import tool


@tool({
    "name": "get_current_model",
    "description": (
        "Read which model is answering right now, and which assistant it is "
        "running as. Call this EVERY time the user asks anything at all about "
        "the model, assistant, or provider in use — 'what model are you?', "
        "'which one is this?', and yes/no forms like 'are you still on "
        "DeepSeek?' or 'aren't you Opus?'. It reads the live setting, so it "
        "is the ONLY correct source: the conversation history is shared by "
        "every assistant and records past choices and other assistants' "
        "switches that no longer apply. Takes no arguments."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def get_current_model(ctx, args):
    # The live override if one is set, else the shared resolver converse()
    # routes with — one function, so this can never disagree with the model
    # that actually answers.
    model_id = ctx.convo_model or agents.registry_model(ctx.active_agent)
    hat = agents.AGENTS.get(ctx.active_agent or "")
    who = hat["name"] if hat else "This assistant"
    return (f"{who} is answering on {cfg.convo_model_label(model_id)}, served "
            f"by {cfg.provider_label(model_id)}. This is the live setting, "
            "read just now. State it as-is, and ignore anything earlier in "
            "the conversation that says otherwise — that was a different "
            "moment or a different assistant.")


@tool({
    "name": "set_conversation_model",
    "description": (
        "Switch which model powers this conversation. Call this when the "
        "user asks to change models — e.g. 'switch to Opus', 'use the smart "
        "model', 'this is hard, think harder', or 'go back to the fast one'. "
        "Options: 'haiku' is fastest and lowest-latency (the default); 'sonnet' "
        "reasons more strongly with a little more delay; 'opus' is the most "
        "capable but the slowest and most expensive. 'deepseek' is an external "
        "budget model (DeepSeek V4 Flash, by far the cheapest); 'deepseek pro' "
        "is DeepSeek's strongest. Map the user's words to the closest option "
        "(e.g. 'the smart/best model' -> opus, 'faster' -> haiku, 'the cheap "
        "one' -> deepseek). The change takes effect immediately, for this "
        "reply onward."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["haiku", "sonnet", "opus", "deepseek", "deepseek pro"],
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
    if (cfg.model_provider(model_id) == "deepseek"
            and not os.environ.get("DEEPSEEK_API_KEY")):
        # Refuse here, while we can still answer with the current model — once
        # ctx.convo_model is set, the very next API call would fail instead.
        return ("I can't switch to DeepSeek: no DEEPSEEK_API_KEY is "
                "configured. Add it to the .env file and restart, or pick a "
                "Claude model.")
    if ctx.convo_model == model_id:
        return f"Already using {cfg.convo_model_label(model_id)}."
    ctx.convo_model = model_id
    return f"Switched the conversation model to {cfg.convo_model_label(model_id)}."

"""Tools over the user's saved notes and folders (NoteStore)."""

from tools import tool


@tool({
    "name": "search_notes",
    "description": (
        "Semantic search across saved notes. Use for questions like 'what did "
        "I say about X' or to find notes on a topic. Pass folder to limit the "
        "search to one folder (e.g. 'what did I note about spreads in my "
        "Trading folder'); omit it to search everywhere."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "folder": {
                "type": "string",
                "description": "Optional: only search this folder, e.g. 'Trading'.",
            },
        },
        "required": ["query"],
    },
})
def search_notes(ctx, args):
    return ctx.store.search_notes(args["query"], folder=args.get("folder"))


@tool({
    "name": "list_recent_notes",
    "description": (
        "List the most recent notes, newest first. Use for 'what's my latest "
        "note' or 'what have I recorded recently'. Pass folder to scope it "
        "(e.g. 'what's the latest note in my General folder'); omit it for "
        "all folders."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "How many to list (default 5)"},
            "folder": {
                "type": "string",
                "description": "Optional: only list notes from this folder, e.g. 'General'.",
            },
        },
    },
})
def list_recent_notes(ctx, args):
    return ctx.store.list_recent_notes(int(args.get("n", 5)), folder=args.get("folder"))


@tool({
    "name": "read_note",
    "description": "Read the full saved summary for a note id (e.g. note_2026-06-22_141500).",
    "input_schema": {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "The note id to read"}
        },
        "required": ["note_id"],
    },
})
def read_note(ctx, args):
    return ctx.store.read_note(args["note_id"])


@tool({
    "name": "list_folders",
    "description": (
        "List the note folders/categories available to file notes into, with a "
        "short description of what belongs in each. Use for 'what folders do I "
        "have', 'where can I put my notes', or 'what categories are there'."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def list_folders(ctx, args):
    return ctx.store.list_folders()


@tool({
    "name": "create_folder",
    "description": (
        "Create a new note folder/category. Use when the user asks to make, "
        "add, or create a folder (e.g. 'create a folder called Recipes'). Pass "
        "the spoken name; optionally a short description of what belongs in it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name for the new folder, e.g. 'Recipes'"},
            "description": {
                "type": "string",
                "description": "Optional: what kind of notes belong in this folder.",
            },
        },
        "required": ["name"],
    },
})
def create_folder(ctx, args):
    return ctx.store.create_folder(args["name"], args.get("description"))


@tool({
    "name": "rename_folder",
    "description": (
        "Rename an existing note folder/category. Use when the user asks to "
        "rename or change a folder's name (e.g. 'rename Ideas to Brainstorms'). "
        "Notes already filed there are kept. Pass the current name and the new name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "current": {"type": "string", "description": "The current folder name, e.g. 'Ideas'"},
            "new_name": {"type": "string", "description": "The new name, e.g. 'Brainstorms'"},
        },
        "required": ["current", "new_name"],
    },
})
def rename_folder(ctx, args):
    return ctx.store.rename_folder(args["current"], args["new_name"])


@tool({
    "name": "delete_folder",
    "description": (
        "Delete a note folder/category. Use when the user asks to delete or "
        "remove a folder. Notes in it are never lost — they're moved to General "
        "by default, or to 'move_notes_to' if the user names a destination. The "
        "General folder itself can't be deleted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The folder to delete, e.g. 'Recipes'"},
            "move_notes_to": {
                "type": "string",
                "description": "Optional folder to move any notes into before deleting (defaults to General).",
            },
        },
        "required": ["name"],
    },
})
def delete_folder(ctx, args):
    return ctx.store.delete_folder(args["name"], args.get("move_notes_to"))


@tool({
    "name": "move_note",
    "description": (
        "Move a single saved note into a different folder. First find the note's "
        "id with search_notes or list_recent_notes (they show it in [brackets]), "
        "then call this with that id and the destination folder. Use for 'move my "
        "last note to Ideas' or 'put the grocery note in Recipes'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "The note id, e.g. note_2026-06-22_141500"},
            "to_folder": {"type": "string", "description": "Destination folder name, e.g. 'Ideas'"},
        },
        "required": ["note_id", "to_folder"],
    },
})
def move_note(ctx, args):
    return ctx.store.move_note(args["note_id"], args["to_folder"])


@tool({
    "name": "count_notes",
    "description": (
        "Count saved notes, optionally within one category folder. Use for "
        "'how many notes do I have' or 'how many notes are in my trading folder'. "
        "Omit folder for a total with a per-category breakdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": "Category/folder name, e.g. 'Trading'. Omit to count all.",
            }
        },
    },
})
def count_notes(ctx, args):
    return ctx.store.count_notes(args.get("folder"))


@tool({
    "name": "save_conversation_note",
    "description": (
        "Save something from this conversation as a note. ONLY call this when "
        "the user explicitly asks (e.g. 'save that as a note', 'make a note of "
        "that'); never call it proactively or suggest saving on your own. Write "
        "the note content yourself from the conversation — clean markdown with a "
        "Summary section and Key Points. After calling this, reply with one "
        "short acknowledgement only; the system will ask the user which "
        "folder to file it in, so never ask about folders yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short descriptive title, max ~8 words"},
            "content": {
                "type": "string",
                "description": "The full note body in markdown, drawn from the conversation.",
            },
            "spoken_summary": {
                "type": "string",
                "description": "1-2 plain sentences recapping the note, to be read aloud after saving.",
            },
            "category": {
                "type": "string",
                "description": "Optional: the folder slug that seems the best fit.",
            },
        },
        "required": ["title", "content"],
    },
})
def save_conversation_note(ctx, args):
    title = (args.get("title") or "").strip() or "Conversation note"
    content = (args.get("content") or "").strip()
    if not content:
        return "No content provided — include the note body in 'content'."
    ctx.pending_note = {
        "title": title,
        "content": content,
        "spoken": (args.get("spoken_summary") or "").strip(),
        "category": args.get("category"),
    }
    return ("Note prepared. Acknowledge briefly; the system will now ask "
            "the user which folder to file it in.")

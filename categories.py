"""Note folder/category registry — the one piece of *runtime-mutable* state
that used to live in config.py.

The built-in seed folders below are defaults: at startup ensure_loaded()
overlays any folders the user has created or renamed by voice (persisted in
data/categories.json, same format as before), and add_category() /
rename_category() / delete_category() mutate the live dict and re-save. Each
note (summary + transcript) is filed under data/<folder>/. "description"
tells the model what belongs in a category — both when auto-classifying a
finished note and when resolving a folder name the user speaks.
"""

import json
import re

import config as cfg

# Slugs never change once assigned so notes, the index, and Chroma metadata
# stay linked across renames.
NOTE_CATEGORIES = {
    "trading": {
        "display": "Trading",
        "folder": "Trading",
        "description": "Any option trading note.",
    },
    "therapy_book": {
        "display": "Therapy book",
        "folder": "TherapyBooks",
        "description": "Any physical therapy or neurology book notes.",
    },
    "to-do": {
        "display": "To-do",
        "folder": "To-do",
        "description": "My to-do notes."
    },
    "ideas": {
        "display": "Ideas",
        "folder": "Ideas",
        "description": "Thoughts I wanna get back to"
    },
    "reminders": {
        "display": "Reminders",
        "folder": "Reminders",
        "description": "Things that I wanna remember later."
    },
    "general": {
        "display": "General",
        "folder": "General",
        "description": "Anything that doesn't fit another category.",
    },
}
DEFAULT_CATEGORY = "general"


def category_dir(slug):
    """Absolute path to a category's folder; unknown slugs fall back to default."""
    slug = slug if slug in NOTE_CATEGORIES else DEFAULT_CATEGORY
    return cfg.DATA_DIR / NOTE_CATEGORIES[slug]["folder"]


def _slugify(name: str) -> str:
    """Stable dict key derived from a display name: lowercase, non-alphanumeric
    runs collapsed to single hyphens."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "folder"


def _folder_name(name: str) -> str:
    """Filesystem-friendly directory name from a display name (e.g. 'Meeting
    notes' -> 'MeetingNotes')."""
    folder = re.sub(r"[^0-9A-Za-z]+", "", (name or "").strip().title())
    return folder or "Folder"


def _unique(value: str, existing, sep: str = "") -> str:
    """Return `value`, or value-2/value2/... if it clashes with `existing`."""
    if value not in existing:
        return value
    i = 2
    while f"{value}{sep}{i}" in existing:
        i += 1
    return f"{value}{sep}{i}"


def load_categories():
    """Overlay user-persisted folders onto the built-in defaults. Idempotent, so
    it's safe to call on every ensure_dirs()."""
    if not cfg.CATEGORIES_PATH.exists():
        return
    try:
        data = json.loads(cfg.CATEGORIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        for slug, meta in data.items():
            if isinstance(meta, dict) and {"display", "folder"} <= meta.keys():
                NOTE_CATEGORIES[slug] = meta


def save_categories():
    cfg.CATEGORIES_PATH.write_text(
        json.dumps(NOTE_CATEGORIES, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_category(display: str, description: str = "") -> str:
    """Create a new folder from a spoken name. Returns its (new, unique) slug and
    creates the directory on disk."""
    display = (display or "").strip()
    slug = _unique(_slugify(display), set(NOTE_CATEGORIES), sep="-")
    folder = _unique(_folder_name(display),
                     {m["folder"] for m in NOTE_CATEGORIES.values()})
    NOTE_CATEGORIES[slug] = {
        "display": display,
        "folder": folder,
        "description": (description or "").strip() or f"Notes about {display}.",
    }
    category_dir(slug).mkdir(parents=True, exist_ok=True)
    save_categories()
    return slug


def rename_category(slug: str, new_display: str):
    """Rename an existing folder's display name (and its on-disk directory) while
    keeping the slug stable so saved notes stay linked."""
    meta = NOTE_CATEGORIES[slug]
    old_dir = category_dir(slug)
    meta["display"] = (new_display or "").strip()
    meta["folder"] = _unique(
        _folder_name(new_display),
        {m["folder"] for s, m in NOTE_CATEGORIES.items() if s != slug},
    )
    new_dir = category_dir(slug)
    if new_dir != old_dir:
        if old_dir.exists():
            old_dir.rename(new_dir)
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
    save_categories()


def delete_category(slug: str):
    """Remove a folder from the registry and drop its (expected-empty) directory.
    Callers must relocate any notes first — this only removes the directory when
    it's empty, so stray files are left in place rather than destroyed."""
    old_dir = category_dir(slug)  # resolve before popping (category_dir needs the entry)
    NOTE_CATEGORIES.pop(slug, None)
    save_categories()
    try:
        if old_dir.exists() and not any(old_dir.iterdir()):
            old_dir.rmdir()
    except OSError:
        pass

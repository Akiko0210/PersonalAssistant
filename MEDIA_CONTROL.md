# How Headset Media-Button Control Works

This document explains how `media_control.py` lets you drive the voice agent
entirely from your Bluetooth headset (AirPods) — no keyboard, no touching the
PC — and the long road of failed approaches that came before it.

## TL;DR

| Press        | AirPods sends | Agent action       |
| ------------ | ------------- | ------------------ |
| Single press | Play / Pause  | Toggle notetaking  |
| Double press | Next track    | Toggle mute        |
| Triple press | Previous track| Quit               |

The working solution registers **our own Windows media session (SMTC)** so the
operating system routes hardware media-button presses to us as *distinct,
already-decoded events*. The broken solutions all tried to read raw keyboard-style
media keys, which AirPods never send.

---

## The core problem

We need to tell **single press** from **double press** from **triple press** on
an AirPods stem, from across the room, with no keyboard.

The trap is assuming the headset sends three different "key" events that you can
count yourself. **It does not.** Understanding *who* decodes the multi-press is
the whole game.

---

## Why the old approaches failed

### Attempt 1 — `pynput` keyboard listener (`test_pynput.py`)

The original design used `pynput` to listen for media keys like
`Key.media_play_pause`, `Key.media_next`, `Key.media_previous` — treating the
headset like a keyboard with media keys.

This works on a **wired** headset because a wired headset is essentially a dumb
button: each physical click emits one HID media-key event, and *our code*
counted the clicks within a timing window to decide single vs. double vs. triple.

**On AirPods it collapses.** We widened debounce windows (80ms → 200ms → 300ms)
and multi-click windows (450ms → 650ms → 1000ms) trying to catch the second and
third clicks. None of it helped, and the reason is fundamental:

> **AirPods do the multi-press decoding themselves, in firmware, before sending
> anything to the computer.** A double squeeze does **not** arrive as two
> `play_pause` events 150ms apart. The AirPods firmware recognizes "that was a
> double press" and sends a **single** `Next` command. A triple press becomes a
> single `Previous` command.

You verified this directly: in `test_pynput.py`, pressing the stem 1×, 2×, or 3×
*all* printed only `media_play_pause` — or nothing. `pynput` literally cannot see
the `Next`/`Previous` intents, because at the OS layer those aren't delivered as
keyboard media keys at all. They're delivered through a different channel:
**SMTC (System Media Transport Controls).**

So the timing-window tuning was doomed from the start. There was no second event
to wait for. We were trying to count clicks that the AirPods had already counted
for us and thrown away the raw form of.

### Why "just observe the current media session" also fails

A tempting middle ground is to *watch* whatever media session is currently active
(e.g. via `GlobalSystemMediaTransportControlsSessionManager`) and react to track
changes. But an observer only sees **side effects** — "the track changed" — not
the **button intent**, and only if some *other* app (Spotify, a browser) happens
to own the session and actually changes tracks. If nothing is playing, there's no
session, and the buttons vanish into the void. Not reliable, not PC-free.

---

## How the working version works

`media_control.py` flips the relationship: instead of *observing* a media session
owned by someone else, **we register our own media session and become the active
one.** Windows then routes the headset's decoded button intents *to us* as
explicit events.

### Step 1 — Create a media session via `MediaPlayer` + SMTC

```python
self._player = MediaPlayer()
self._player.command_manager.is_enabled = False   # so the legacy button_pressed event fires
smtc = self._player.system_media_transport_controls
```

`SystemMediaTransportControls` (SMTC) is the Windows subsystem behind the media
overlay you see when you press play on a keyboard or headset. Any app can publish
a session into it.

### Step 2 — Enable the session and each button

```python
smtc.is_enabled = True            # REQUIRED — without this, no button events fire at all
smtc.is_play_enabled = True
smtc.is_pause_enabled = True
smtc.is_next_enabled = True
smtc.is_previous_enabled = True
```

`smtc.is_enabled = True` was the **single most important fix.** Without it, the
session exists but Windows never dispatches button presses to it — you get total
silence on every press, which is exactly the dead-end we hit first. Enabling each
individual button tells Windows "yes, route Next/Previous to me too," which is
what makes the double/triple-press gestures reachable.

### Step 3 — Give the session real metadata

```python
updater = smtc.display_updater
updater.type = MediaPlaybackType.MUSIC
updater.music_properties.title = "Voice Agent"
updater.update()
smtc.playback_status = MediaPlaybackStatus.PLAYING
```

Windows only treats us as a genuine, button-eligible media session if we look
like one: a media type, a title, and a "Playing" status. Claiming `PLAYING` is
what makes Windows consider us the **active** session worth routing buttons to.

### Step 4 — Subscribe to button presses

```python
smtc.add_button_pressed(self._on_button)
```

Note the API quirk: it's `add_button_pressed(handler)`, **not** the `+=` event
syntax you'd use in C#. (Same story with `is_looping_enabled`, not
`is_loop_enabled`.) These are pywinrt naming conventions that differ from the
.NET docs, and tripped us up until corrected.

In the handler, `args.button` is already one of `PLAY` / `PAUSE` / `NEXT` /
`PREVIOUS` — **the multi-press has already been decoded for us.** We just map each
to a callback, with a short debounce to swallow duplicate events:

```python
def _on_button(self, sender, args):
    cb = self._cb.get(args.button)
    if cb is None:
        return
    now = time.monotonic()
    if now - self._last.get(args.button, 0) < self._debounce_s:
        return        # ignore Bluetooth duplicate/phantom repeats
    self._last[args.button] = now
    cb()
```

### Step 5 — The silent keepalive (winning the session)

Registering a session isn't enough if *another* app (Spotify, a YouTube tab) is
already the active session — the headset buttons go to **them**, not us. This was
the "still nothing at all" symptom even after `is_enabled = True`.

The fix: actually **play audio**, so Windows promotes us to the active session.
We loop a 1-second silent WAV forever:

```python
# generate 1 second of silence to a temp .wav
path = tempfile.mktemp(suffix=".wav")
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
    w.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))

self._player.source = MediaSource.create_from_uri(
    Uri("file:///" + path.replace("\\", "/")))
self._player.is_looping_enabled = True
self._player.play()
```

It's inaudible, but to Windows we are a media app that is actively playing —
so we own the session and the headset buttons land on us. This is what made the
buttons work reliably "from across the room."

---

## Why this is PC-free and robust

- **No keyboard bindings.** Nothing depends on focus, foreground windows, or HID
  media keys. The agent works while your laptop is in your bag.
- **The headset's own firmware does the click-counting**, and SMTC hands us the
  result pre-decoded — so single/double/triple are unambiguous and instant, with
  no timing windows to tune.
- **We own the session via the silent keepalive**, so we don't lose the buttons
  to whatever else might be playing.

---

## The packaging gotcha

The standard `winsdk` package wouldn't install — it tries to build from source and
demands Visual Studio (a non-starter on Python 3.13 without a full toolchain).
The fix was the modern **`winrt-*` namespace packages**, which ship prebuilt
wheels:

```
winrt-runtime
winrt-Windows.Media
winrt-Windows.Media.Playback
winrt-Windows.Media.Core      # needed for the silent keepalive (MediaSource)
winrt-Windows.Foundation
```

`media_control.py` imports `winrt.*` first and falls back to `winsdk.*` (identical
API) if only the legacy package is present.

---

## Summary of the key insight

The whole saga reduces to one realization:

> **You can't count AirPods clicks yourself — the AirPods already counted them.**
> A double press is delivered as `Next`, a triple as `Previous`, and the only
> channel that carries those intents on Windows is SMTC. So instead of listening
> for keyboard media keys (which never arrive), we publish our own SMTC session,
> enable it, claim the active-playing state with a silent keepalive, and receive
> the already-decoded button events directly.

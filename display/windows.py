"""Windows variant of the display window's OS-specific half.

Windows won't let a background process take the foreground: it only flashes
the taskbar button. Setting the window topmost and clearing it again raises
it above every other window, and SWP_NOACTIVATE keeps the keyboard where the
user left it. pywebview's own on_top goes through WinForms' TopMost, which
activates as it raises, and from a thread that isn't the UI's.
"""

import subprocess

# Off the agent's console, so one rule decides when the window goes — its
# agent's process is gone (agent_alive) — not which console signals reach it.
POPEN_KWARGS = {"creationflags": subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP}

# Win32 values pywin32 doesn't name in every release.
_QUERY_LIMITED = 0x1000   # PROCESS_QUERY_LIMITED_INFORMATION
_STILL_ACTIVE = 259


def agent_alive(pid):
    # Not os.kill(pid, 0): on Windows that is TerminateProcess.
    import pywintypes
    import win32api
    import win32process
    try:
        handle = win32api.OpenProcess(_QUERY_LIMITED, False, pid)
    except pywintypes.error:
        return False  # no such process
    try:
        return win32process.GetExitCodeProcess(handle) == _STILL_ACTIVE
    finally:
        win32api.CloseHandle(handle)


def bring_to_front(window):
    import win32con
    import win32gui
    hwnd = win32gui.FindWindow(None, window.title)
    if not hwnd:
        return
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
    flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)

"""Raw Input probe, phase 2: register EVERY HID collection and keyboard input.

Phase 1 (test_rawinput.py) registered four likely collections and, with sound
playing, saw only HALF the play/pause presses (4 of 8, all on Consumer Control
as `01 08 00`). The missing presses produced nothing there — so they either
arrive on a collection phase 1 didn't register (e.g. a vendor-defined UC/Teams
page), arrive as keyboard-type raw input (phase 1 discarded those), or the
dongle genuinely doesn't transmit them.

This probe removes the guesswork:
  1. It enumerates every HID top-level collection present on the system and
     registers Raw Input for ALL of them (plus the keyboard type), so nothing
     can slip past on an unregistered page.
  2. It prints the full device list at startup — look for the Yealink entries
     to see what collections the dongle exposes (vendor pages show as
     page=0xFFxx).
  3. Every event is labelled with the device it came from.

Run:  python test_rawinput2.py            (sound ON  — the failing case)
      python test_rawinput2.py --silent   (no sound  — control)
      python test_rawinput2.py --obey     (sound toggles on each press)

--obey tests the dongle-state-machine theory: the phase-2 run showed EXACTLY
every other press vanishing (nothing on any channel), which fits a dongle that
tracks play/pause state itself — it sends "pause", and if the host's audio
keeps playing anyway, the next press is swallowed while it flips its state
back. --obey behaves like an obedient media player (each received press
toggles the sound), keeping the dongle's state in sync. If all 8 presses
arrive in this mode, the theory is confirmed and the agent can stay reliable
by always silencing audio when a press lands.

Click play/pause 8 times, ~2 s apart, then Ctrl+C. If the missing presses now
show up (on a vendor page or as keyboard events), that channel gets wired into
the agent. If they STILL don't appear anywhere, the dongle is not sending them
and the fix lives in the headset's own configuration software.
"""

import ctypes as ct
from ctypes import wintypes as wt
import sys
import time
from pathlib import Path

user32 = ct.WinDLL("user32", use_last_error=True)
kernel32 = ct.WinDLL("kernel32", use_last_error=True)

# --- Win32 constants ---
HWND_MESSAGE = -3
RIDEV_INPUTSINK = 0x00000100
RIDEV_PAGEONLY = 0x00000020
RID_INPUT = 0x10000003
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002
PM_REMOVE = 0x0001

RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RIM_TYPEHID = 2

RIDI_DEVICENAME = 0x20000007
RIDI_DEVICEINFO = 0x2000000B

VK_NAMES = {
    0xB3: "MEDIA_PLAY_PAUSE", 0xB2: "MEDIA_STOP",
    0xB0: "MEDIA_NEXT_TRACK", 0xB1: "MEDIA_PREV_TRACK",
    0xAD: "VOLUME_MUTE", 0xAE: "VOLUME_DOWN", 0xAF: "VOLUME_UP",
}

LRESULT = ct.c_ssize_t
WNDPROC = ct.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class RAWINPUTDEVICE(ct.Structure):
    _fields_ = [("usUsagePage", wt.USHORT),
                ("usUsage", wt.USHORT),
                ("dwFlags", wt.DWORD),
                ("hwndTarget", wt.HWND)]


class RAWINPUTDEVICELIST(ct.Structure):
    _fields_ = [("hDevice", wt.HANDLE),
                ("dwType", wt.DWORD)]


class RAWINPUTHEADER(ct.Structure):
    _fields_ = [("dwType", wt.DWORD),
                ("dwSize", wt.DWORD),
                ("hDevice", wt.HANDLE),
                ("wParam", wt.WPARAM)]


class RAWHID(ct.Structure):
    _fields_ = [("dwSizeHid", wt.DWORD),
                ("dwCount", wt.DWORD),
                ("bRawData", ct.c_ubyte * 1)]


class RAWKEYBOARD(ct.Structure):
    _fields_ = [("MakeCode", wt.USHORT),
                ("Flags", wt.USHORT),
                ("Reserved", wt.USHORT),
                ("VKey", wt.USHORT),
                ("Message", wt.UINT),
                ("ExtraInformation", wt.ULONG)]


class RID_DEVICE_INFO_HID(ct.Structure):
    _fields_ = [("dwVendorId", wt.DWORD),
                ("dwProductId", wt.DWORD),
                ("dwVersionNumber", wt.DWORD),
                ("usUsagePage", wt.USHORT),
                ("usUsage", wt.USHORT)]


class RID_DEVICE_INFO(ct.Structure):
    # Only the HID arm of the union matters here; pad to the full union size.
    _fields_ = [("cbSize", wt.DWORD),
                ("dwType", wt.DWORD),
                ("hid", RID_DEVICE_INFO_HID),
                ("_pad", ct.c_ubyte * 8)]


class WNDCLASS(ct.Structure):
    _fields_ = [("style", wt.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ct.c_int),
                ("cbWndExtra", ct.c_int),
                ("hInstance", wt.HINSTANCE),
                ("hIcon", wt.HANDLE),
                ("hCursor", wt.HANDLE),
                ("hbrBackground", wt.HANDLE),
                ("lpszMenuName", wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR)]


# --- function prototypes ---
user32.GetRawInputDeviceList.argtypes = [ct.c_void_p, ct.POINTER(wt.UINT), wt.UINT]
user32.GetRawInputDeviceList.restype = wt.UINT
user32.GetRawInputDeviceInfoW.argtypes = [wt.HANDLE, wt.UINT, ct.c_void_p,
                                          ct.POINTER(wt.UINT)]
user32.GetRawInputDeviceInfoW.restype = wt.UINT
user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, ct.c_void_p,
                                   ct.POINTER(wt.UINT), wt.UINT]
user32.GetRawInputData.restype = wt.UINT
user32.RegisterRawInputDevices.argtypes = [ct.POINTER(RAWINPUTDEVICE), wt.UINT, wt.UINT]
user32.RegisterRawInputDevices.restype = wt.BOOL
user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassW.argtypes = [ct.POINTER(WNDCLASS)]
user32.RegisterClassW.restype = wt.ATOM
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                                   ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                   wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.CreateWindowExW.restype = wt.HWND
user32.PeekMessageW.argtypes = [ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
user32.PeekMessageW.restype = wt.BOOL
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HMODULE

_playing = False
_hid_count = 0
_key_count = 0
_dev_label = {}  # hDevice(int) -> short label, filled during enumeration
_HID_DATA_OFFSET = ct.sizeof(RAWINPUTHEADER) + RAWHID.bRawData.offset


def _ts():
    return time.strftime("%H:%M:%S")


def _device_name(hdevice):
    size = wt.UINT(0)
    user32.GetRawInputDeviceInfoW(hdevice, RIDI_DEVICENAME, None, ct.byref(size))
    if not size.value:
        return "?"
    buf = ct.create_unicode_buffer(size.value + 1)
    got = user32.GetRawInputDeviceInfoW(hdevice, RIDI_DEVICENAME, buf, ct.byref(size))
    return buf.value if got and got != 0xFFFFFFFF else "?"


def _short(name):
    r"""Compress a device path like \\?\HID#VID_6993&PID_B70B&MI_03... to the
    interesting middle (VID/PID/interface/collection)."""
    core = name.split("#")[1] if "#" in name else name
    return core[:34]


def enumerate_devices():
    """Return ([(page, usage)], startup report lines). Also fills _dev_label."""
    n = wt.UINT(0)
    user32.GetRawInputDeviceList(None, ct.byref(n), ct.sizeof(RAWINPUTDEVICELIST))
    arr = (RAWINPUTDEVICELIST * max(1, n.value))()
    got = user32.GetRawInputDeviceList(arr, ct.byref(n), ct.sizeof(RAWINPUTDEVICELIST))
    if got == 0xFFFFFFFF:
        print("GetRawInputDeviceList failed:", ct.get_last_error())
        return [], []
    pairs = set()
    lines = []
    for d in arr[:got]:
        name = _device_name(d.hDevice)
        label = _short(name)
        _dev_label[int(d.hDevice or 0)] = label
        if d.dwType == RIM_TYPEHID:
            info = RID_DEVICE_INFO()
            info.cbSize = ct.sizeof(RID_DEVICE_INFO)
            size = wt.UINT(ct.sizeof(RID_DEVICE_INFO))
            ok = user32.GetRawInputDeviceInfoW(d.hDevice, RIDI_DEVICEINFO,
                                               ct.byref(info), ct.byref(size))
            if ok and ok != 0xFFFFFFFF:
                page, usage = info.hid.usUsagePage, info.hid.usUsage
                pairs.add((page, usage))
                lines.append(f"  HID  page=0x{page:02X} usage=0x{usage:02X}  "
                             f"vid={info.hid.dwVendorId:04X} "
                             f"pid={info.hid.dwProductId:04X}  {label}")
        elif d.dwType == RIM_TYPEKEYBOARD:
            lines.append(f"  KBD  {label}")
        # mice skipped — registering them would just spam movement events
    return sorted(pairs), lines


def start_sound():
    global _playing
    path = Path(__file__).with_name("assets") / "summarizing.wav"
    if not path.is_file():
        print(f"[!] no sound at {path} — running silent")
        return
    try:
        import winsound
        winsound.PlaySound(
            str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
        )
        _playing = True
    except Exception as e:  # noqa: BLE001
        print("could not start sound:", e)


def stop_sound():
    global _playing
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:  # noqa: BLE001
        print("could not stop sound:", e)
    _playing = False


OBEY = False  # --obey: act like an obedient player — each press toggles the sound


def _obey_press():
    if not OBEY:
        return
    if _playing:
        stop_sound()
        print(f"[{_ts()}]        (obey: sound paused)")
    else:
        start_sound()
        print(f"[{_ts()}]        (obey: sound resumed)")


def _handle_raw_input(lparam):
    global _hid_count, _key_count
    size = wt.UINT(0)
    user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, None,
                           ct.byref(size), ct.sizeof(RAWINPUTHEADER))
    if not size.value:
        return
    buf = (ct.c_ubyte * size.value)()
    got = user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, buf,
                                 ct.byref(size), ct.sizeof(RAWINPUTHEADER))
    if got == 0xFFFFFFFF or got == 0:
        return
    header = ct.cast(buf, ct.POINTER(RAWINPUTHEADER)).contents
    src = _dev_label.get(int(header.hDevice or 0), "hDevice=0 (synthesized)")

    if header.dwType == RIM_TYPEHID:
        hid = ct.cast(ct.byref(buf, ct.sizeof(RAWINPUTHEADER)),
                      ct.POINTER(RAWHID)).contents
        n = hid.dwSizeHid * hid.dwCount
        data = bytes(buf[_HID_DATA_OFFSET:_HID_DATA_OFFSET + n])
        pressed = any(data[1:])
        if pressed:
            _hid_count += 1
        print(f"[{_ts()}] HID #{_hid_count:<3} ({n:>2}B) {data.hex(' '):<24} "
              f"{'<-- PRESS' if pressed else '(release)'}  "
              f"sound={'ON ' if _playing else 'OFF'}  {src}")
        if pressed:
            _obey_press()
    elif header.dwType == RIM_TYPEKEYBOARD:
        kbd = ct.cast(ct.byref(buf, ct.sizeof(RAWINPUTHEADER)),
                      ct.POINTER(RAWKEYBOARD)).contents
        up = kbd.Flags & 1
        name = VK_NAMES.get(kbd.VKey, "")
        if not up:
            _key_count += 1
        print(f"[{_ts()}] KEY #{_key_count:<3} vk=0x{kbd.VKey:02X} {name:17} "
              f"{'(release)' if up else '<-- PRESS'}  "
              f"sound={'ON ' if _playing else 'OFF'}  {src}")


def wndproc(hwnd, msg, wparam, lparam):
    if msg == WM_INPUT:
        _handle_raw_input(lparam)
        return 0
    if msg == WM_DESTROY:
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


def main():
    global OBEY
    OBEY = "--obey" in sys.argv
    if "--silent" not in sys.argv:
        start_sound()

    pairs, report = enumerate_devices()
    print("HID collections present on this system:")
    print("\n".join(report) or "  (none found?)")
    print()

    hinst = kernel32.GetModuleHandleW(None)
    wndproc_cb = WNDPROC(wndproc)  # keep a reference alive
    wc = WNDCLASS()
    wc.lpfnWndProc = wndproc_cb
    wc.hInstance = hinst
    wc.lpszClassName = "RawInputMediaProbe2"
    if not user32.RegisterClassW(ct.byref(wc)):
        print("RegisterClassW failed:", ct.get_last_error())
        return
    hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "probe2", 0, 0, 0, 0, 0,
                                  wt.HWND(HWND_MESSAGE), None, hinst, None)
    if not hwnd:
        print("CreateWindowExW failed:", ct.get_last_error())
        return

    # Register every HID collection found, plus the keyboard type. One call per
    # pair so a single rejected registration can't sink the rest.
    to_register = pairs + [(0x01, 0x06)]  # generic keyboard
    registered = []
    for page, usage in to_register:
        if (page, usage) == (0x01, 0x02):
            continue  # mouse — movement spam, can't carry the button
        # usage 0 means "whole page" — that needs the PAGEONLY flag
        flags = RIDEV_INPUTSINK | (RIDEV_PAGEONLY if usage == 0 else 0)
        dev = RAWINPUTDEVICE(page, usage, flags, hwnd)
        if user32.RegisterRawInputDevices(ct.byref(dev), 1,
                                          ct.sizeof(RAWINPUTDEVICE)):
            registered.append((page, usage))
        else:
            print(f"  [!] could not register page=0x{page:02X} "
                  f"usage=0x{usage:02X} (error {ct.get_last_error()})")
    print("Registered:", ", ".join(f"0x{p:02X}/0x{u:02X}" for p, u in registered))
    print(f"\nListening (sound={'ON' if _playing else 'OFF'}). "
          f"Click play/pause 8 times, ~2s apart; Ctrl+C to stop.\n")

    msg = wt.MSG()
    try:
        while True:
            while user32.PeekMessageW(ct.byref(msg), None, 0, 0, PM_REMOVE):
                user32.DispatchMessageW(ct.byref(msg))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\nTotals -> HID presses: {_hid_count}   key presses: {_key_count}")


if __name__ == "__main__":
    main()

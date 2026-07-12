"""Raw Input probe for the media button.

The keyboard-hook logs proved that while audio plays, ~half the Yealink dongle's
play/pause presses never reach the low-level keyboard hook — Windows routes them
around it (media-session / HID app-command). Raw Input reads the HID reports
straight from the device, below that routing, so it should see EVERY press.

This standalone probe registers for Consumer Control HID input and prints every
report. By default it loops the agent's idle sound so you test under the failing
condition; pass --silent to run without audio for comparison.

Run:  python test_rawinput.py            (sound ON  — the failing case)
      python test_rawinput.py --silent   (no sound  — control)

Click play/pause ~8 times and count the reports. If every press prints a report
even with sound ON, Raw Input is the fix and I'll wire it into the agent.
Stop with Ctrl+C.
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
RID_INPUT = 0x10000003
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002
RIM_TYPEHID = 2
PM_REMOVE = 0x0001

RIDI_DEVICEINFO = 0x2000000b

# HID usage (page, usage) collections to listen on. Consumer control is where
# play/pause normally lives; the telephony/headset pages are where a call-control
# headset like the Yealink often re-routes its button while audio is active.
LISTEN_USAGES = [
    (0x0C, 0x01),  # Consumer Control
    (0x0B, 0x05),  # Telephony - Headset
    (0x0B, 0x01),  # Telephony - Phone
    (0x0C, 0x80),  # Consumer - Selection (some headsets)
]

LRESULT = ct.c_ssize_t
WNDPROC = ct.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class RAWINPUTDEVICE(ct.Structure):
    _fields_ = [("usUsagePage", wt.USHORT),
                ("usUsage", wt.USHORT),
                ("dwFlags", wt.DWORD),
                ("hwndTarget", wt.HWND)]


class RAWINPUTHEADER(ct.Structure):
    _fields_ = [("dwType", wt.DWORD),
                ("dwSize", wt.DWORD),
                ("hDevice", wt.HANDLE),
                ("wParam", wt.WPARAM)]


class RAWHID(ct.Structure):
    _fields_ = [("dwSizeHid", wt.DWORD),
                ("dwCount", wt.DWORD),
                ("bRawData", ct.c_ubyte * 1)]


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


# --- function prototypes ---
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
user32.GetRawInputDeviceInfoW.argtypes = [wt.HANDLE, wt.UINT, ct.c_void_p,
                                          ct.POINTER(wt.UINT)]
user32.GetRawInputDeviceInfoW.restype = wt.UINT
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HMODULE

_playing = False
_count = 0
_dev_usage = {}  # hDevice -> "page/usage" label, cached
_HID_DATA_OFFSET = ct.sizeof(RAWINPUTHEADER) + RAWHID.bRawData.offset


def _device_label(hdevice):
    key = int(hdevice) if hdevice else 0
    if key in _dev_usage:
        return _dev_usage[key]
    label = "?"
    info = RID_DEVICE_INFO()
    info.cbSize = ct.sizeof(RID_DEVICE_INFO)
    size = wt.UINT(ct.sizeof(RID_DEVICE_INFO))
    got = user32.GetRawInputDeviceInfoW(wt.HANDLE(hdevice), RIDI_DEVICEINFO,
                                        ct.byref(info), ct.byref(size))
    if got and got != 0xFFFFFFFF:
        label = f"page=0x{info.hid.usUsagePage:02X} usage=0x{info.hid.usUsage:02X}"
    _dev_usage[key] = label
    return label


def _ts():
    return time.strftime("%H:%M:%S")


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


def _handle_raw_input(lparam):
    global _count
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
    if header.dwType != RIM_TYPEHID:
        return
    hid = ct.cast(ct.byref(buf, ct.sizeof(RAWINPUTHEADER)), ct.POINTER(RAWHID)).contents
    n = hid.dwSizeHid * hid.dwCount
    data = bytes(buf[_HID_DATA_OFFSET:_HID_DATA_OFFSET + n])
    # Byte 0 is the report ID; a press has a non-zero payload after it, release
    # is all-zero payload. This is what a physical click actually looks like.
    pressed = any(data[1:])
    if pressed:
        _count += 1
    src = _device_label(header.hDevice)
    print(f"[{_ts()}] #{_count:<3} {src:<24} ({n:>2}B) {data.hex(' '):<20} "
          f"{'<-- PRESS' if pressed else '(release)'}  sound={'ON ' if _playing else 'OFF'}")


def wndproc(hwnd, msg, wparam, lparam):
    if msg == WM_INPUT:
        _handle_raw_input(lparam)
        return 0
    if msg == WM_DESTROY:
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


def main():
    global _count
    if "--silent" not in sys.argv:
        start_sound()

    hinst = kernel32.GetModuleHandleW(None)
    wndproc_cb = WNDPROC(wndproc)  # keep a reference alive
    wc = WNDCLASS()
    wc.lpfnWndProc = wndproc_cb
    wc.hInstance = hinst
    wc.lpszClassName = "RawInputMediaProbe"
    if not user32.RegisterClassW(ct.byref(wc)):
        print("RegisterClassW failed:", ct.get_last_error())
        return
    hwnd = user32.CreateWindowExW(0, wc.lpszClassName, "probe", 0, 0, 0, 0, 0,
                                  wt.HWND(HWND_MESSAGE), None, hinst, None)
    if not hwnd:
        print("CreateWindowExW failed:", ct.get_last_error())
        return

    devs = (RAWINPUTDEVICE * len(LISTEN_USAGES))()
    for i, (page, usage) in enumerate(LISTEN_USAGES):
        devs[i].usUsagePage = page
        devs[i].usUsage = usage
        devs[i].dwFlags = RIDEV_INPUTSINK
        devs[i].hwndTarget = hwnd
    if not user32.RegisterRawInputDevices(devs, len(LISTEN_USAGES),
                                          ct.sizeof(RAWINPUTDEVICE)):
        print("RegisterRawInputDevices failed:", ct.get_last_error())
        return
    print("Registered usage pages:",
          ", ".join(f"0x{p:02X}/0x{u:02X}" for p, u in LISTEN_USAGES))

    print(f"Listening (sound={'ON' if _playing else 'OFF'}). "
          f"Click play/pause ~8 times; Ctrl+C to stop.\n")
    msg = wt.MSG()
    try:
        while True:
            while user32.PeekMessageW(ct.byref(msg), None, 0, 0, PM_REMOVE):
                user32.DispatchMessageW(ct.byref(msg))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\nTotal presses seen via Raw Input: {_count}")


if __name__ == "__main__":
    main()

from pynput import keyboard

def on_press(key):
    print("Pressed:", key)

    if key == keyboard.Key.media_play_pause:
        print("MATCH: media_play_pause")

    elif key == keyboard.Key.media_volume_up:
        print("MATCH: media_volume_up")

    elif key == keyboard.Key.media_volume_down:
        print("MATCH: media_volume_down")

    elif key == keyboard.Key.media_volume_mute:
        print("MATCH: media_volume_mute")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
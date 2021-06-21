import winsound as sd

def ring_alarm():
    sd.PlaySound("SystemAsterisk", sd.SND_ALIAS | sd.SND_ASYNC)
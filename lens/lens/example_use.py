from lens.lens.detect_score import DETECT

srcs = [
    # 1 - very good simplification
    "Die komplexe Maschine wurde von den Ingenieuren konzipiert, um den Energieverbrauch zu minimieren.",
    # 2 - good meaning, slightly awkward
    "Obwohl das Wetter schlecht war, entschied sich die Gruppe, die Wanderung trotzdem fortzusetzen.",
    # 3 - poor simplification (still complex)
    "Die Implementierung der strategischen Maßnahmen erforderte eine detaillierte Analyse der betrieblichen Prozesse.",
    # 4 - completely wrong meaning
    "Der Arzt verschrieb dem Patienten ein neues Medikament zur Behandlung der chronischen Schmerzen.",
    # 5 - ungrammatical / not fluent
    "Das Hund gehen schnell Straße über und springen auf Auto.",
    # 6 - simple but loses key meaning
    "Die Regierung kündigte ein umfangreiches Paket wirtschaftlicher Reformen an, um die Krise zu bewältigen.",
    # 7 - fluent but adds wrong info
    "Das Unternehmen stellte viele neue Mitarbeiter ein, um die Produktion zu erhöhen.",
    # 8 - overly literal translation (stiff, unnatural)
    "Er machte eine Entscheidung, zu gehen zum Markt.",
    # 9 - nonsensical / garbage output
    "Blume Tisch springen gestern glücklich.",
    # 10 - perfect simplification (short, clear, same meaning)
    "Das Kind aß den Apfel, den seine Mutter ihm gegeben hatte."
]

mts = [
    # 1
    "Die Maschine wurde so gebaut, dass sie weniger Energie verbraucht.",
    # 2
    "Es war schlechtes Wetter, aber die Gruppe ging trotzdem weiter.",
    # 3
    "Die Umsetzung der Maßnahmen brauchte eine detaillierte Analyse der Prozesse.",
    # 4
    "Der Arzt hat den Patienten operiert.",
    # 5
    "Der Hund gehen Straße schnell und Auto springen.",
    # 6
    "Die Regierung machte Änderungen, um das Problem zu lösen.",
    # 7
    "Das Unternehmen entließ viele Arbeiter, um Kosten zu sparen.",
    # 8
    "Er entschied sich, zum Markt zu gehen.",
    # 9
    "Glücklich springende Tische essen Blumen.",
    # 10
    "Das Kind bekam von seiner Mutter einen Apfel und aß ihn."
]

refs = [
    ["Die Ingenieure entwickelten die Maschine, damit sie weniger Energie braucht."],
    ["Trotz des schlechten Wetters wanderte die Gruppe weiter."],
    ["Für die Maßnahmen musste man die Abläufe genau untersuchen."],
    ["Der Arzt gab dem Patienten neue Medikamente gegen die Schmerzen."],
    ["Der Hund lief schnell über die Straße und sprang auf ein Auto."],
    ["Die Regierung plante Reformen, um die Krise zu bewältigen."],
    ["Das Unternehmen stellte neue Leute ein, um mehr zu produzieren."],
    ["Er beschloss, auf den Markt zu gehen."],
    ["Das ist völlig unverständlich."],
    ["Das Kind aß den Apfel, den seine Mutter ihm gegeben hatte."]
]

detect = DETECT(rescale=True)
scores = detect.score(srcs, mts, refs, batch_size=8, devices=[0])
print(scores)

#[{'simplicity': 65.86385840008762, 'meaning_preservation': 63.56789293496763, 'fluency': 61.15524475006539, 'total': 64.00374948403518},
# {'simplicity': 51.84285692406446, 'meaning_preservation': 36.74834595870999, 'fluency': 45.87459322968328, 'total': 44.61139979904644},
# {'simplicity': 62.42447154285515, 'meaning_preservation': 74.02913548475031, 'fluency': 59.96728417384312, 'total': 66.57489964581082},
# {'simplicity': 52.276910348841795, 'meaning_preservation': 19.678954784378238, 'fluency': 49.99620326667203, 'total': 19.678954784378238},
# {'simplicity': 68.23166606936574, 'meaning_preservation': 52.0437610719525, 'fluency': 58.29686362047064, 'total': 59.76954358062143},
# {'simplicity': 59.13034959083314, 'meaning_preservation': 34.96279269779944, 'fluency': 55.18327191660898, 'total': 48.67391129877483},
# {'simplicity': 51.39972211046231, 'meaning_preservation': 53.897205093852875, 'fluency': 49.990919317281126, 'total': 52.116954745182305},
# {'simplicity': 69.69536936303516, 'meaning_preservation': 70.57698751418677, 'fluency': 64.44418593061387, 'total': 68.99777993701154},
# {'simplicity': 59.494574648630916, 'meaning_preservation': 63.30572949838905, 'fluency': 53.17056531069998, 'total': 59.754234720947984},
# {'simplicity': 59.02726739462645, 'meaning_preservation': 69.82844542994316, 'fluency': 57.69896504242751, 'total': 63.08207813831335}]
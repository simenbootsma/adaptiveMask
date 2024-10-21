import PySimpleGUI as sg


layout = [
    [sg.Frame('Calibration:', [[sg.Text(), sg.Column([[sg.Text('Save as:')],
                                                      [sg.Input(key='-SAVE-NAME-', size=(19, 1), default_text='save_name.txt')],
                                                      [sg.Checkbox('Crop image view area', key='-CROP-')],
                                                      [sg.Button('Calibrate', key='-CALIBRATE-')]
                                                      ])
                                ]]
              )],
    [sg.Frame('Run:', [[sg.Text(), sg.Column([
                                                      [sg.Button('Run', key='-RUN-')]
                                                      ])
                                ]]
              )]
]


window = sg.Window('Adaptive mask', layout, font=("Helevtica", 16))

while True:
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED:
        break

window.close()


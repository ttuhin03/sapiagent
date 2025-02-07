import os
import json
import csv
from datetime import datetime

# Pfad zum Ordner mit den JSON-Dateien
json_folder_path = '/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/owndata_raw'
# Pfad zum Ordner für die Ausgabe-CSV-Dateien
csv_output_folder_path = '/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/sapimouse_ownhumandata'

# Erstellen Sie den Ausgabeordner, falls er nicht existiert
os.makedirs(csv_output_folder_path, exist_ok=True)

# Benutzerzähler initialisieren
user_counter = 1

# Alle JSON-Dateien im Ordner durchlaufen
for filename in os.listdir(json_folder_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, filename)
        
        # JSON-Datei lesen
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Den ersten Zeitstempel als Startzeitpunkt festlegen
        start_timestamp = data['mouseEvents'][0]['timestamp']
        
        # Benutzerordner erstellen
        user_folder = os.path.join(csv_output_folder_path, f'user{user_counter}')
        os.makedirs(user_folder, exist_ok=True)
        
        # Datum aus dem Dateinamen extrahieren
        date_str = filename.split('_')[1]
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        formatted_date = date_obj.strftime('%Y_%m_%d')
        
        # Pfad zur Ausgabe-CSV-Datei
        csv_file_name = f'session_{formatted_date}_3min.csv'
        csv_file_path = os.path.join(user_folder, csv_file_name)
        
        # CSV-Datei schreiben
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # CSV-Header schreiben
            csv_writer.writerow(['client timestamp', 'button', 'state', 'x', 'y'])
            
            # MouseEvents in CSV-Format konvertieren und schreiben
            for event in data['mouseEvents']:
                relative_timestamp = event['timestamp'] - start_timestamp
                button_state = 'NoButton'
                if event['button'] == 1:
                    button_state = 'LeftButton'
                elif event['button'] == 2:
                    button_state = 'RightButton'
                
                # Mausereignistypen ändern
                event_type = event['type']
                if event_type == 'mousemove':
                    event_type = 'Move'
                elif event_type == 'mousedown':
                    event_type = 'Pressed'
                elif event_type == 'mouseup':
                    event_type = 'Released'
                
                csv_writer.writerow([relative_timestamp, button_state, event_type, event['x'], event['y']])
        
        # Benutzerzähler erhöhen
        user_counter += 1

print("Alle Mouse events wurden erfolgreich konvertiert und gespeichert.")
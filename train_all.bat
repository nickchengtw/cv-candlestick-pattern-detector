
python .\extract_features.py -c peak.json
python .\train_model.py -c .\peak.json

python .\extract_features.py -c valley.json
python .\train_model.py -c .\valley.json

@REM python .\test_model.py -c .\peak.json -i ".\testing\Screenshot 2024-12-01 232716.png"
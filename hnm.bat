python .\hard_negative_mine.py -c peak.json
python .\hard_negative_mine.py -c .\valley.json
python .\train_model.py -c .\peak.json --hard-negatives 1
python .\train_model.py -c .\valley.json --hard-negatives 1
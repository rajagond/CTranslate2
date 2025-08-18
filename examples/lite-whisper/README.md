### Lite Whisper example

#### compile
```bash
bash compile.sh
```

#### convert to ct2
```bash
bash convert.sh whisper-tiny
```

#### run examples
```bash
python3 whisper.py --device cuda
python3 lite-whisper.py --device cuda
```
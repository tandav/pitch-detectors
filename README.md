# pitch-detectors
collection of pitch detection algorithms with unified interface

## list of algorithms

| algorithm  | cpu | gpu |
|------------|-----|-----|
| [PraatAC](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_ac)    |  ✓  |     |
| [PraatCC](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_cc)    |  ✓  |     |
| [PraatSHS](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_shs)   |  ✓  |     |
| [Pyin](https://librosa.org/doc/latest/generated/librosa.pyin.html)       |  ✓  |     |
| [Reaper](https://github.com/r9y9/pyreaper)     |  ✓  |     |
| [Yaapt](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html#amfm_decompy.pYAAPT.yaapt)      |  ✓  |     |
| [Rapt](https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.rapt.html)       |  ✓  |     |
| [Swipe](https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.swipe.html)      |  ✓  |  ✓  |
| [World](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)      |  ✓  |     |
| [TorchYin](https://github.com/brentspell/torch-yin)   |  ✓  |     |
| [Crepe](https://github.com/marl/crepe)      |  ✓  |  ✓  |
| [TorchCrepe](https://github.com/maxrmorrison/torchcrepe) |  ✓  |  ✓  |


## additional features
- robust (vote-based + median) averaging of pitch
- json import/export

## install
```bash
pip install pitch-detectors
```

## usage

# pitch-detectors
collection of pitch detection (f0, fundamental frequency) algorithms with unified interface

## list of algorithms
<!-- table-start -->
| algorithm                                                                                                  | cpu   | gpu   | [MIR-1K](https://www.kaggle.com/datasets/datongmuyuyi/mir1k) accuracy   |
|------------------------------------------------------------------------------------------------------------|-------|-------|-------------------------------------------------------------------------|
| [Crepe](https://github.com/marl/crepe)                                                                     | ✓     | ✓     | 0.811 ± 0.080                                                           |
| [Penn](https://github.com/interactiveaudiolab/penn)                                                        | ✓     | ✓     | 0.742 ± 0.126                                                           |
| [PipTrack](https://librosa.org/doc/latest/generated/librosa.piptrack.html)                                 | ✓     |       | 0.727 ± 0.228                                                           |
| [PraatAC](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_ac)   | ✓     |       | 0.885 ± 0.069                                                           |
| [PraatCC](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_cc)   | ✓     |       | 0.897 ± 0.073                                                           |
| [PraatSHS](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch_shs) | ✓     |       | 0.628 ± 0.201                                                           |
| [Pyin](https://librosa.org/doc/latest/generated/librosa.pyin.html)                                         | ✓     |       | 0.891 ± 0.058                                                           |
| [Rapt](https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.rapt.html)                            | ✓     |       | 0.864 ± 0.068                                                           |
| [Reaper](https://github.com/r9y9/pyreaper)                                                                 | ✓     |       | 0.833 ± 0.075                                                           |
| [Spice](https://ai.googleblog.com/2019/11/spice-self-supervised-pitch-estimation.html)                     | ✓     | ✓     | 0.913 ± 0.056                                                           |
| [Swipe](https://pysptk.readthedocs.io/en/stable/generated/pysptk.sptk.swipe.html)                          | ✓     |       | 0.882 ± 0.065                                                           |
| [TorchCrepe](https://github.com/maxrmorrison/torchcrepe)                                                   | ✓     | ✓     | 0.825 ± 0.077                                                           |
| [TorchYin](https://github.com/brentspell/torch-yin)                                                        | ✓     |       | 0.887 ± 0.075                                                           |
| [World](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)                                   | ✓     |       | 0.879 ± 0.061                                                           |
| [Yaapt](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html#amfm_decompy.pYAAPT.yaapt)                    | ✓     |       | 0.774 ± 0.103                                                           |
| [Yin](https://librosa.org/doc/latest/generated/librosa.yin.html#librosa.yin)                               | ✓     |       | 0.887 ± 0.064                                                           |<!-- table-stop -->



accuracy is mean [raw pitch accuracy](http://craffel.github.io/mir_eval/#mir_eval.melody.raw_pitch_accuracy)

## install
all agorithms tested on python3.10, this is recommended python version to use
```bash
pip install pitch-detectors
```

## usage

```python
from scipy.io import wavfile
from pitch_detectors import algorithms
import matplotlib.pyplot as plt

fs, a = wavfile.read('data/b1a5da49d564a7341e7e1327aa3f229a.wav')
pitch = algorithms.Crepe(a, fs)
plt.plot(pitch.t, pitch.f0)
plt.show()
```

![Alt text](data/b1a5da49d564a7341e7e1327aa3f229a.png)


## additional features
- [ ] robust (vote + median) ensemble algorithm using all models
- [ ] json import/export

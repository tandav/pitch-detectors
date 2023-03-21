from pathlib import Path

from pitch_detectors.algorithms import ALGORITHMS

MIR_1K_DIR = Path('MIR-1K')
WAV_DIR = MIR_1K_DIR / 'Wavfile'


def main():
    for wav_path in WAV_DIR.glob('*.wav'):
        for algorithm in ALGORITHMS:
            print(wav_path.name, algorithm)


if __name__ == '__main__':
    main()

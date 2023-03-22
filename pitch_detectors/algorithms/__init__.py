from pitch_detectors.algorithms.crepe import Crepe
from pitch_detectors.algorithms.penn import Penn
from pitch_detectors.algorithms.piptrack import PipTrack
from pitch_detectors.algorithms.praatac import PraatAC
from pitch_detectors.algorithms.praatcc import PraatCC
from pitch_detectors.algorithms.praatshs import PraatSHS
from pitch_detectors.algorithms.pyin import Pyin
from pitch_detectors.algorithms.rapt import Rapt
from pitch_detectors.algorithms.reaper import Reaper
from pitch_detectors.algorithms.spice import Spice
from pitch_detectors.algorithms.swipe import Swipe
from pitch_detectors.algorithms.torchcrepe import TorchCrepe
from pitch_detectors.algorithms.torchyin import TorchYin
from pitch_detectors.algorithms.world import World
from pitch_detectors.algorithms.yaapt import Yaapt
from pitch_detectors.algorithms.yin import Yin

ALGORITHMS = (
    PraatAC,
    PraatCC,
    PraatSHS,
    Pyin,
    Yin,
    Reaper,
    Yaapt,
    Crepe,
    TorchCrepe,
    Swipe,
    Rapt,
    World,
    TorchYin,
    Spice,
    Penn,
    PipTrack,
)

cpu_algorithms = tuple(a.name() for a in ALGORITHMS if not a.use_gpu)  # type: ignore
gpu_algorithms = tuple(a.name() for a in ALGORITHMS if a.use_gpu)  # type: ignore
algorithms = cpu_algorithms + gpu_algorithms

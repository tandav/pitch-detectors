import os

import s3fs

s3 = s3fs.S3FileSystem(
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'],
)

s3.get('pitchtrack/spice_model', os.environ['PITCH_DETECTORS_SPICE_MODEL_PATH'], recursive=True)
s3.get('pitchtrack/fcnf0++.pt', os.environ['PITCH_DETECTORS_PENN_CHECKPOINT_PATH'])

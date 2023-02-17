import os
import time

import numpy as np
import onnxruntime as ort


# os.add_dll_directory(r'D:\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')
# os.add_dll_directory(r'D:\NVIDIA GPU Computing Toolkit\cuDNN\bin')

tokens = np.load('onnx/assets/tokens.npy')
durations = np.load('onnx/assets/durations.npy')
f0 = np.load('onnx/assets/f0_denorm.npy')
speedup = np.array(50, dtype=np.int64)

print('tokens', tokens.shape)
print('durations', durations.shape)
print('f0', f0.shape)

options = ort.SessionOptions()
session = ort.InferenceSession(
    'onnx/assets/1220_zhibin_ds1000.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    sess_options=options
)

start = time.time()
mel = session.run(['mel'], {'tokens': tokens, 'durations': durations, 'f0': f0, 'speedup': speedup})[0]
end = time.time()

print('mel', mel.shape)
print('cost', end - start)

np.save('onnx/assets/mel_test.npy', mel)

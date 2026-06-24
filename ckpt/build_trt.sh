#!/usr/bin/env bash
set -e

# Resolve TensorRT root – prefer TENSORRT_ROOT env var, then auto-detect
if [ -z "$TENSORRT_ROOT" ]; then
  TENSORRT_ROOT=$(ls -d ~/Software/TensorRT-* 2>/dev/null | sort | tail -1)
  if [ -z "$TENSORRT_ROOT" ]; then
    echo "ERROR: Cannot find TensorRT under ~/Software/. Set TENSORRT_ROOT env var."
    exit 1
  fi
fi
echo ">>> Using TensorRT: $TENSORRT_ROOT"

TRT_BIN=$TENSORRT_ROOT/bin/trtexec
TRT_LIB=$TENSORRT_ROOT/lib

echo ">>> Deactivating conda env (if any)"
conda deactivate 2>/dev/null || true

echo ">>> Setting TensorRT LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB

echo ">>> Building TensorRT engine: 512x640"
$TRT_BIN \
  --onnx=spnet_512_640.onnx \
  --saveEngine=spnet_512_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x512x640,depth:1x1x512x640,mask:1x1x512x640

echo ">>> Building TensorRT engine: 480x640"
$TRT_BIN \
  --onnx=spnet_480_640.onnx \
  --saveEngine=spnet_480_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x480x640,depth:1x1x480x640,mask:1x1x480x640

echo ">>> TensorRT engine build finished."

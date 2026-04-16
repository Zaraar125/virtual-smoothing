#!/bin/bash

# =============================================================================
# ResNet-18 Variant Training Script (Virtual Smoothing Classes)
# Tradeoff: Model Size vs Accuracy on CIFAR-10
# =============================================================================

# --- Common Training Args ---
DATASET="cifar10"
METHOD="clean"
V_CLASSES=10
ALPHA=0.5
GPUID=0
EPOCHS=30
MODEL_NAME="resnet-18-custom"

# --- Variants: "base_width b1 b2 b3 b4" ---
VARIANTS=(
#   "64 2 2 2 2"   # Baseline ResNet-18
  "64 1 2 2 1"   # ResNet-14 (reduced depth)
  "64 1 1 1 1"   # ResNet-10 (min depth)
  "32 2 2 2 2"   # Half-width
  "16 2 2 2 2"   # Quarter-width
  "32 1 1 1 1"   # ResNet-10 Half-width
)

echo "=============================================="
echo " Starting ResNet-18 Variant Training"
echo " Dataset : $DATASET"
echo " Method  : $METHOD"
echo " Epochs  : $EPOCHS"
echo "=============================================="

for VARIANT in "${VARIANTS[@]}"; do

  # Parse variant parameters
  read -r W B1 B2 B3 B4 <<< "$VARIANT"

  # Build directory/log suffix
  SUFFIX="${W}_${B1}_${B2}_${B3}_${B4}"

  MODEL_DIR=".dnn/cifar_10/resnet_18_custom_${SUFFIX}"
  LOG_DIR="training_logs/cifar_10/resnet_18_custom_${SUFFIX}"

  echo ""
  echo "----------------------------------------------"
  echo " Variant : base_width=${W}, blocks=[${B1},${B2},${B3},${B4}]"
  echo " Model Dir : $MODEL_DIR"
  echo " Log Dir   : $LOG_DIR"
  echo "----------------------------------------------"

  python -W ignore clean_train_vs.py \
    --model_name        "$MODEL_NAME"    \
    --dataset           "$DATASET"       \
    --training_method   "$METHOD"        \
    --v_classes         "$V_CLASSES"     \
    --alpha             "$ALPHA"         \
    --gpuid             "$GPUID"         \
    --model_dir         "$MODEL_DIR"     \
    --training_logs     "$LOG_DIR"       \
    --epochs            "$EPOCHS"        \
    --base_width        "$W"             \
    --resnet_num_blocks "$B1" "$B2" "$B3" "$B4"

  if [ $? -eq 0 ]; then
    echo " [DONE] Variant ${SUFFIX} trained successfully."
  else
    echo " [FAILED] Variant ${SUFFIX} encountered an error. Continuing..."
  fi

done

echo ""
echo "=============================================="
echo " All variants finished."
echo "=============================================="
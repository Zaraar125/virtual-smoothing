#!/bin/bash

# =============================================================================
# ResNet-18 Variant Evaluation Script (Virtual Smoothing Classes)
# =============================================================================

# --- Common Eval Args ---
DATASET="cifar10"
GPUID=0
V_CLASSES=10
MODEL_NAME="resnet-18-custom"
TESTING_LOGS="testing_logs"
ALPHA=0.5
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
echo " Starting ResNet-18 Variant Evaluation"
echo " Dataset : $DATASET"
echo "=============================================="

for VARIANT in "${VARIANTS[@]}"; do

  # Parse variant parameters
  read -r W B1 B2 B3 B4 <<< "$VARIANT"

  SUFFIX="${W}_${B1}_${B2}_${B3}_${B4}"
  MODEL_DIR=".dnn/cifar_10/resnet_18_custom_${SUFFIX}"
  LOG_DIR="${TESTING_LOGS}/cifar_10/resnet_18_custom_${SUFFIX}"

  echo ""
  echo "----------------------------------------------"
  echo " Variant    : base_width=${W}, blocks=[${B1},${B2},${B3},${B4}]"
  echo " Model Dir  : $MODEL_DIR"

  # --- Check model directory exists ---
  if [ ! -d "$MODEL_DIR" ]; then
    echo " [SKIP] Model directory not found: $MODEL_DIR"
    continue
  fi

  # --- Find highest epoch model file ---
  MAX_EPOCH=$(ls "${MODEL_DIR}"/clean_model_epoch*.pt 2>/dev/null \
    | grep -oP 'epoch\K[0-9]+' \
    | sort -n \
    | tail -1)

  if [ -z "$MAX_EPOCH" ]; then
    echo " [SKIP] No model files found in: $MODEL_DIR"
    continue
  fi

  MODEL_FILE="${MODEL_DIR}/clean_model_epoch${MAX_EPOCH}.pt"

  echo " Best Epoch : $MAX_EPOCH"
  echo " Model File : $MODEL_FILE"
  echo " Log Dir    : $LOG_DIR"
  echo "----------------------------------------------"

  python -W ignore eval_clean.py \
    --model_name        "$MODEL_NAME"    \
    --dataset           "$DATASET"       \
    --gpuid             "$GPUID"         \
    --v_classes         "$V_CLASSES"     \
    --model_file        "$MODEL_FILE"    \
    --base_width        "$W"             \
    --resnet_num_blocks "$B1" "$B2" "$B3" "$B4" \
    --testing_logs      "$LOG_DIR" \
    --final_epoch       "$MAX_EPOCH" \
    --alpha             "$ALPHA"


  if [ $? -eq 0 ]; then
    echo " [DONE] Variant ${SUFFIX} (epoch ${MAX_EPOCH}) evaluated successfully."
  else
    echo " [FAILED] Variant ${SUFFIX} encountered an error. Continuing..."
  fi

done

echo ""
echo "=============================================="
echo " All evaluations finished."
echo "=============================================="
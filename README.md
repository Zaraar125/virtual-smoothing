# Training Deep Neural Networks with Virtual Smoothing Classes (AAAI 2025)

**Requisite**\
This code is implemented in PyTorch, and we have tested the code under the following environment settings:\
`python = 3.8.5`\
`numpy == 1.19.2`\
`torch = 1.7.1`\
`torchvision = 0.8.2`

**Train Neural Networks with Virtual Smoothing Labels**
1. Standard Training:  
`python clean_train_vs.py --model_name resnet-18 --dataset cifar10 --training_method clean --v_classes 10 --alpha 0.5 --gpuid 0 --model_dir $MODEL_DIR$`

2. Adversarial Training:  
`python rob_train_vs.py --model_name WRN-34-10 --dataset cifar10 --training_method pgd --v_classes 10 --alpha 0.9 --gpuid 0 --model_dir $MODEL_DIR$`

**Evaluation**    
1. Standard Evaluation 
`python eval_clean.py --gpuid 0 --v_classes 10 --model_file $MODEL_DIR$`
2.  Adersarial Evaluation
`python eval_rob.py --gpuid 0 --v_classes 10 --model_file $MODEL_DIR$`

**Reference Code**\    
[1] AT: https://github.com/locuslab/robust_overfitting \
[2] TRADES: https://github.com/yaodongyu/TRADES/ \
[3] AutoAttack: https://github.com/fra31/auto-attack \
[4] RST: https://github.com/yaircarmon/semisup-adv \
[5] AWP: https://github.com/csdongxian/AWP

## Resnet - 18 base variant
python -W ignore clean_train_vs.py --model_name resnet-18 --dataset cifar10 --training_method clean --v_classes 10 --alpha 0.5 --gpuid 0 --model_dir .dnn/cifar_10/resnet_18 --training_logs training_logs/cifar_10/resnet_18 --alpha 0.5 --epochs 28 --resume_epoch 2

python -W ignore eval_clean.py --model_name resnet-18 --dataset cifar10 --gpuid 0 --v_classes 10 --model_file .dnn/cifar_10/resnet_18/clean_model_epoch28.pt 

## ResNext -29 base variant
 python -W ignore clean_train_vs.py --model_name resnext-29_2x64d --dataset cifar10 --training_method clean --v_classes 10 --alpha 0.5 --gpuid 0 --model_dir .dnn/cifar_10/resnext_29_2_64d --training_logs training_logs/cifar_10/resnext_29_2_64d --epochs 30 --resume_epoch 2

python -W ignore eval_clean.py --model_name resnext-29_2x64d --dataset cifar10 --gpuid 0 --v_classes 10 --model_file .dnn\cifar_10\resnext_29_2_64d\clean_model_epoch30.pt    


### plotting : 
python -W ignore .\visualization\resnet_18_BxV_test_dislpay.py --log_root testing_logs/cifar_10 --output_dir results/resnet_18_BxV

python -W ignore .\visualization\resnet_18_BxV_train_display.py --log_root training_logs/cifar_10 --output_dir results/resnet_18_BxV

### training vit_14 something something
python -W ignore main_mnist.py --dataset mnist --model_name t2t_vit_14 --optim AdamW --lr 3e-4 --weight_decay 0.05 --epochs 15 --batch_size 64 --alpha 0.5 --v_classes 10 --model_dir .dnn/mnist/t2t_vit_14 --training_logs training_logs/mnist/t2t_vit_14

python -W ignore eval_clean.py --model_name t2t_vit_14 --dataset mnist --gpuid 0 --v_classes 10 --model_file .dnn/mnist/t2t_vit_14/clean_model_epoch15.pt --testing_logs testing_logs/mnist/t2t_vit_14 --final_epoch 15 --batch_size 64 --alpha 0.5
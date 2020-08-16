# Train
python3 main.py --config config/nturgbd-cross-subject/train_upper.yaml --phase train

# Test
python3 main.py --config ./config/nturgbd-cross-subject/test_upper.yaml --work-dir pretrain_eval/ntu60/xsub/joint-fusion --weights pretrained-models/ntu60-xsub-joint-fusion.pt


Place the trained model checkpoint here after downloading from Kaggle.

Expected filename: roadrishi_finetuned.pth

Steps:
1. Run training/kaggle_segformer_train.py on Kaggle (attach DeepGlobe dataset, enable GPU T4)
2. After training completes, go to Kaggle Output panel
3. Download roadrishi_finetuned.pth
4. Place it in this folder: training/checkpoints/roadrishi_finetuned.pth

Once the checkpoint is here, the /api/checkpoint-status endpoint will switch from
"simulation" mode to "live_model" mode automatically.

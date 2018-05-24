th evaluate_model_A.lua -max_images -1 -checkpoint checkpoint.t7 -gpu 1
ipython eval/measure_meteor.py

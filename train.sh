python train.py --maxdisp 192 \
               --model CVSMNet_Downsize \
               --datapath /media/yoko/SSD-PGU3/workspace/datasets/KITTI/data_scene_flow/training/ \
               --epochs 300 \
               --loadmodel result/weights/CVSMNet_Downsize299.tar \
               --savemodel ./result

               
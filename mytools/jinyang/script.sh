CUDA_VISIBLE_DEVICES=2,3 python run_net.py \
  --cfg configs/SSv2/SLOWFAST_16x8_R50.yaml
  DATA.PATH_TO_DATA_DIR /data/zjt/ssv2/
  NUM_GPUS 1
  TRAIN.BATCH_SIZE 16
  BN.NUM_SYNC_DEVICES 1

CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 run_net.py \
  --cfg configs/SSv2/SLOWFAST_16x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/zjt/ssv2/ \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16

CUDA_VISIBLE_DEVICES=2,3,4,5,6  python run_net.py  --cfg configs/SSv2/SLOWFAST_16x8_R50.yaml

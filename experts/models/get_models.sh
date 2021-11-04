wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e-BnicM1tifi12hgQpmu20PKZDH7zXnv' -O of_raft.pth
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Al53Y5bk5LRy8pZ8Nj0Qixoyey6bxboy' -O of_liteflownet
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=125dZCdCh5YUq2HLYf9wD89NTioXfhedz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=125dZCdCh5YUq2HLYf9wD89NTioXfhedz" -O saliency_seg_egnet.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kvh_yH8v4EZNrJE63TWUS2FK2o7Axo89' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kvh_yH8v4EZNrJE63TWUS2FK2o7Axo89" -O normals_xtc.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vm-wZs3s6UwyAK-H52KxSKOUVJkvA9Tb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vm-wZs3s6UwyAK-H52KxSKOUVJkvA9Tb" -O edges_dexined.h5 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J_AcX9T2BP5v6XO25vu3QLaW6CIXWqgR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J_AcX9T2BP5v6XO25vu3QLaW6CIXWqgR" -O depth_sgdepth.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wPvpUMbZ5neDO_OeyRhVrZ1jf-TT7mFP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wPvpUMbZ5neDO_OeyRhVrZ1jf-TT7mFP" -O depth_xtc.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PIjJ0U4uNsPcyNPySKXkJEdNEvvuvcn_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PIjJ0U4uNsPcyNPySKXkJEdNEvvuvcn_" -O hrnet_encoder_epoch_30.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FBnv2Jxq1tCiTsaMByHlNYmInHsYry0I' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FBnv2Jxq1tCiTsaMByHlNYmInHsYry0I" -O hrnet_decoder_epoch_30.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1miSQk9HxSBZdFxnPIbr2-s7CEU4rPtRO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1miSQk9HxSBZdFxnPIbr2-s7CEU4rPtRO" -O cartoon_wb.zip && rm -rf /tmp/cookies.txt
unzip cartoon_wb.zip
wget https://raw.githubusercontent.com/fuy34/superpixel_fcn/master/pretrain_ckpt/SpixelNet_bsd_ckpt.tar
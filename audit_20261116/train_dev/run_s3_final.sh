#!/usr/bin/env bash
# S3 with the final trainer code: fresh run (300 s budget) then resume to 420 s total, into s3_final/.
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
export S3_DIR=audit_20261116/train_dev/s3_final
bash audit_20261116/train_dev/run_s3.sh 300 > audit_20261116/train_dev/logs/s3_final_fresh.log 2>&1
tail -1 audit_20261116/train_dev/logs/s3_final_fresh.log
cp $S3_DIR/latest.pkl audit_20261116/train_dev/s3_final_latest_part1.pkl
bash audit_20261116/train_dev/run_s3.sh resume 420 > audit_20261116/train_dev/logs/s3_final_resume.log 2>&1
tail -1 audit_20261116/train_dev/logs/s3_final_resume.log

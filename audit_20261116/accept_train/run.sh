cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
source audit_20261116/nav4_dev/env.sh
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
A=audit_20261116/accept_train
$NAV3_PYTHON -m unittest discover -s nav4/tests -p "test_*.py" > $A/tests.log 2>&1; echo "tests rc=$?" >> $A/done.txt
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method dreamer --action mode --checkpoint checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl --repeats 8 --seed 914901 --output $A/s2.json > $A/s2.log 2>&1; echo "s2 rc=$?" >> $A/done.txt
$NAV3_PYTHON -m nav4.evaluate --manifest r9_official_runner/maps_scan_holdout --method planner --repeats 8 --seed 914901 --output $A/c2.json > $A/c2.log 2>&1; echo "c2 rc=$?" >> $A/done.txt

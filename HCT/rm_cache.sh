export TASK_NAME=snli
cd datasets/$TASK_NAME/with_conf/train
rm cache*
cd ../../../../
cd datasets/$TASK_NAME/with_conf/validation
rm cache*
cd ../../../../
cd datasets/$TASK_NAME/with_conf/test
rm cache*
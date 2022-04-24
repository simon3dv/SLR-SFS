for ((i=0; i<130; i+=10))
do
sbatch test_animating/CLAW/test_v1_sbatch_1.sh $i $((i+9))
echo $i $((i+10))
done
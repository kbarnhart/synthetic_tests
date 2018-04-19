## run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
  dakota -i dakota_ego.in -o dakota_ego.out --read_restart dakota.rst &> dakota.log
else
  dakota -i dakota_ego.in -o dakota_ego.out &> dakota.log
fi

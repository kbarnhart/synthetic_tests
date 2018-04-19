## run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
  dakota -i dakota_hierarchical.in -o dakota_hierarchical.out --read_restart dakota.rst &> dakota.log
else
  dakota -i dakota_hierarchical.in -o dakota_hierarchical.out &> dakota.log
fi

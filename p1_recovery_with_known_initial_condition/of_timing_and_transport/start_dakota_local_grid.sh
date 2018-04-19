## run dakota using a restart file if it exists.
if [ -e dakota.rst ]
then
  dakota -i dakota_grid.in -o dakota_grid.out --read_restart dakota.rst &> dakota.log
else
  dakota -i dakota_grid.in -o dakota_grid.out &> dakota.log
fi

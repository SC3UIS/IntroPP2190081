# Evaluación 2

Se experimentó implementando OpenMP como un acercamiento intuitivo ante los bucles y cálculos repetitivos
que están en el archivo core.c.

Se observó que a pesar de que inicialmente tenía sentido paralelizar operaciones repetitivas, esto 
mostró servir en contra de lo esperado, aumentando el tiempo de ejecución de forma exponencial.

Por lo tanto, se descartó esta opción como viable tras múltiples pruebas en una máquina local.

No obstante, en cambio, se implementaron más banderas a nivel del compilador, precisamente 

    -Os

Debido a que implementa una bandera que optimiza funciones inline, tal como la función idx,
que se llama repetidas veces en el archivo core.c.

De esta misma manera, se implementaron las banderas -march=core-avx2 -msse2, tras haber
ejecutado 
  
    $ cat /proc/cpuinfo 


Para ejecutar el código

    make
    mpirun -np 4 ./heat_mpi



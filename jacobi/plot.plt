unset colorbox
set palette rgb 33,13,10
set size square
plot 'solution.bin' binary array=(size+2,size+2) format="%lf" with image

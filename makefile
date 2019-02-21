ransac:main.cpp  kernel.o  common.h verify.h timer.h
	g++ -pthread -mavx -o ransac main.cpp  kernel.o 
kernel.o:kernel.cpp  partitioner.h kernel.h
	g++  -c -march=native kernel.cpp
clean:
	rm ransac kernel.o
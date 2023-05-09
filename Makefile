CC=gcc
CFLAGS=-O2
all:
	$(CC) $(CFLAGS) csum-test-and-bench.c -o csum-test-and-bench

clean:
	rm -f csum-test-and-bench *~

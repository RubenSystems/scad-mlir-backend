
CC = clang
CFLAGS = -Wall -Wextra -g

STANDALONE_SRCS := $(wildcard standalone/*.scad)
STANDALONE_OBJS := $(STANDALONE_SRCS:.scad=.o)
FINAL_STANDALONE := $(STANDALONE_OBJS:.o=)
STANDALONE_TEST_RESULTS := $(STANDALONE_OBJS:.o=.txt)

HOSTED_SRCS := $(wildcard hosted/*.scad)
HOSTED_C_SRCS := $(wildcard hosted/*.c)
HOSTED_BUILD_RESULTS := $(HOSTED_SRCS:.scad=.scadout)



.PHONY: all clean

all: standalone_tests

%: %.o
	$(CC) $(CFLAGS) $< -o $@

%.o: %.scad
	../build/bin/scadc $< $@

%.c: %.o
	$(CC) -O3 $@ -o $*.scadout $<

hosted: $(HOSTED_C_SRCS)

test_standalone: $(FINAL_STANDALONE)
	@for exe in $(FINAL_STANDALONE); do \
		echo "Testing $$exe"; \
		./$$exe > $$exe.out; \
		if diff -q $$exe.out $$exe.txt > /dev/null; then \
			echo "Test passed!"; \
		else \
			echo "Test failed!"; \
		fi; \
		rm $$exe.out;\
	done


standalone_tests: $(FINAL_STANDALONE)
	@for exe in $(FINAL_STANDALONE); do \
		if [ ! -f $$exe.txt ]; then \
			echo "generating tests for $$exe"; \
			./$$exe > $$exe.txt; \
			echo "created tests for $$exe"; \
		else \
			echo "$$exe.txt already exists, skipping..."; \
		fi \
	done




reset: 
	rm -f $(STANDALONE_TEST_RESULTS) 

clean:
	rm -f $(STANDALONE_OBJS) $(FINAL_STANDALONE) $(HOSTED_BUILD_RESULTS)

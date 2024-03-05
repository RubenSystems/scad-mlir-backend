
CC = clang
CFLAGS = -Wall -Wextra -g

STANDALONE_SRCS := $(wildcard standalone/*.scad)
STANDALONE_OBJS := $(STANDALONE_SRCS:.scad=.o)
FINAL_STANDALONE := $(STANDALONE_OBJS:.o=)

.PHONY: all clean

all: $(FINAL_STANDALONE)

%: %.o
	$(CC) $(CFLAGS) $< -o $@

%.o: %.scad
	../build/bin/scadc $< $@

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
		echo "generating tests for $$exe"; \
		./$$exe > $$exe.txt; \
		echo "created tests for $$exe"; \
	done


clean:
	rm -f $(STANDALONE_OBJS) $(FINAL_STANDALONE) 


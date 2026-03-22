CC = gcc
CFLAGS = -mavx -mavx2 -pthread
LDFLAGS = -lm

SRC_DIR = src

TARGETS = main_float main_double

all: $(TARGETS)

main_float: $(SRC_DIR)/main_float.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main_double: $(SRC_DIR)/main_double.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGETS)

.PHONY: all clean

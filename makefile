TARGET = main
CC = gcc
CFLAGS = -Wall -Wextra -g -std=c11 -mavx -mavx2
SRC_DIR = src

SRC = $(SRC_DIR)/main.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

.PHONY: all clean

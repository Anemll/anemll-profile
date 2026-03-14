PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin

CC = xcrun clang
CFLAGS = -O2 -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML
SRC = anemll_profile.m
BIN = anemll-profile
TEST_SCRIPT = tests/test.sh
TEST_FIXTURE = tests/fixtures/multifunction.mlpackage

.PHONY: all clean install uninstall test

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $<

install: $(BIN)
	install -d $(BINDIR)
	install -m 755 $(BIN) $(BINDIR)/$(BIN)

test: $(BIN)
	$(TEST_SCRIPT) ./$(BIN) $(TEST_FIXTURE)

uninstall:
	rm -f $(BINDIR)/$(BIN)

clean:
	rm -f $(BIN)

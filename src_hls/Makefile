
# A generic Makefile for building any project
# with the following directory structure.
# ./src
# ./obj
# ./bin

SRCDIR  = .
OBJDIR  = .
BINDIR  = .
FILES   := $(wildcard $(SRCDIR)/*.C)
OBJECTS := $(FILES:$(SRCDIR)/%.C=$(OBJDIR)/%.o)

TARGET  = stitch
CC      = g++
CCFLAGS = -std=c++1z -I/usr/include/eigen3/
LDFLAGS = -lm -lpthread -lX11

$(BINDIR)/$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "Linking Complete."
	@echo "To start the file explorer run --> "$(BINDIR)"/"$(TARGET)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.C
	$(CC) $(CCFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully."

clean:
	@echo "Cleaning all the object files and binaries."
	rm -f core $(OBJECTS) $(BINDIR)/$(TARGET)

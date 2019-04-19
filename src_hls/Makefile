
# A generic Makefile for building any project
# with the following directory structure.
# ./src
# ./obj
# ./bin

TARGET  = stitch
CC      = g++
CCFLAGS = -Wall -std=c++1z
SRCDIR  = .
OBJDIR  = .
BINDIR  = .

FILES   := $(wildcard $(SRCDIR)/*.C)
OBJECTS := $(FILES:$(SRCDIR)/%.C=$(OBJDIR)/%.o)

$(BINDIR)/$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@
	@echo "Linking Complete."
	@echo "To start the file explorer run --> "$(BINDIR)"/"$(TARGET)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.C
	$(CC) $(CCFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully."

clean:
	@echo "Cleaning all the object files and binaries."
rm -f core $(OBJECTS) $(BINDIR)/$(TARGET)

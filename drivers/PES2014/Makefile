################################################################################
#
#   This Makefile will compile the potential energy surface module to expose it
#   to Python using f2py.
#
################################################################################

F2PY=f2py

.PHONY: all build clean

all: build

build:
	$(F2PY) -c ch4oh2014.f utility.f ch4oh2014.pyf
	cp *so ../../ipi/engine

clean:
	rm -f *.so

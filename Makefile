default: release

.PHONY: default release debug all clean sonar cppcheck

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Ietl/include -stdlib=libc++

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,bench))

release: release_bench
debug: debug_bench

all: release debug

run: release
	./release/bin/bench

cppcheck:
	cppcheck --enable=all --std=c++11 -I include src

sonar: release
	cppcheck --xml-version=2 --enable=all --std=c++11 -I include src 2> cppcheck_report.xml
	/opt/sonar-runner/bin/sonar-runner

clean: base_clean

include make-utils/cpp-utils-finalize.mk

default: release

.PHONY: default release debug all clean sonar cppcheck

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Ietl/include -Ietl/lib/include

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -stdlib=libc++
endif

# Enable vectorization
CXX_FLAGS += -DETL_VECTORIZE

# Enable BLAS/MKL on demand
ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
endif
endif

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

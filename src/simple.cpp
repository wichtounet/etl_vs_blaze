#include <blaze/Math.h>
#include <eigen3/Eigen/Dense>
#include "etl/etl.hpp"

#define CPM_PARALLEL_RANDOMIZE
#define CPM_FAST_RANDOMIZE

#define CPM_WARMUP 5
#define CPM_REPEAT 20

#define CPM_BENCHMARK "ETL/Blaze/Eigen Benchmark"
#include "cpm/cpm.hpp"

namespace {

template<typename T>
using etl_dyn_vector = etl::dyn_vector<T>;

template<typename T>
using blaze_dyn_vector = blaze::DynamicVector<T>;

template<typename T>
using etl_dyn_matrix = etl::dyn_matrix<T>;

template<typename T>
using blaze_dyn_matrix = blaze::DynamicMatrix<T>;

template<typename T>
using eigen_dyn_vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template<typename T>
using eigen_dyn_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using etl_dvec = etl_dyn_vector<double>;
using blaze_dvec = blaze_dyn_vector<double>;
using eigen_dvec = eigen_dyn_vector<double>;

using etl_dmat = etl_dyn_matrix<double>;
using blaze_dmat = blaze_dyn_matrix<double>;
using eigen_dmat = eigen_dyn_matrix<double>;

CPM_SECTION_P("r *= 3.3", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d)); },
        [](etl_dvec& r){ r *= 3.3; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d)); },
        [](blaze_dvec& r){ r *= 3.3; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d)); },
        [](eigen_dvec& r){ r *= 3.3; }
        );
}

CPM_SECTION_P("add_complex", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& r, etl_dvec& a, etl_dvec& b){ r = a + b + a + b + a + a + b + a + a; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& r, blaze_dvec& a, blaze_dvec& b){ r = a + b + a + b + a + a + b + a + a; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& r, eigen_dvec& a, eigen_dvec& b){ r = a + b + a + b + a + a + b + a + a; }
        );
}

CPM_SECTION_P("mix", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& r, etl_dvec& a, etl_dvec& b){ r = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& r, blaze_dvec& a, blaze_dvec& b){ r = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& r, eigen_dvec& a, eigen_dvec& b){ r = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1; }
        );
}

CPM_SECTION_P("mix_matrix", NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2)); },
        [](etl_dmat& R, etl_dmat& A, etl_dmat& B){ R = A + A * 5.9 + A + B - B / 2.3 - A + B * 1.1; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2)); },
        [](blaze_dmat& R, blaze_dmat& A, blaze_dmat& B){ R = A + A * 5.9 + A + B - B / 2.3 - A + B * 1.1; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2)); },
        [](eigen_dmat& R, eigen_dmat& A, eigen_dmat& B){ R = A + A * 5.9 + A + B - B / 2.3 - A + B * 1.1; }
        );
}

CPM_SECTION_P("r = a + b", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& r, etl_dvec& a, etl_dvec& b){ r = a + b; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& r, blaze_dvec& a, blaze_dvec& b){ r = a + b; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& r, eigen_dvec& a, eigen_dvec& b){ r = a + b; }
        );
}

CPM_SECTION_P("r = a + b + c", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& r, etl_dvec& a, etl_dvec& b, etl_dvec& c){ r = a + b + c; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& r, blaze_dvec& a, blaze_dvec& b, blaze_dvec& c){ r = a + b + c; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& r, eigen_dvec& a, eigen_dvec& b, eigen_dvec& c){ r = a + b + c; }
        );
}

CPM_SECTION_P("r = a + b + c + d", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& r, etl_dvec& a, etl_dvec& b, etl_dvec& c, etl_dvec& d){ r = a + b + c + d; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& r, blaze_dvec& a, blaze_dvec& b, blaze_dvec& c, blaze_dvec& d){ r = a + b + c + d; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& r, eigen_dvec& a, eigen_dvec& b, eigen_dvec& c, eigen_dvec& d){ r = a + b + c + d; }
        );
}

CPM_SECTION_P("R = A'", NARY_POLICY(VALUES_POLICY(64, 64, 128, 256, 256, 256, 300, 512, 512, 1024, 2048, 2048), VALUES_POLICY(64, 128, 128, 128, 256, 384, 500, 512, 1024, 1024, 1024, 2048)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d2,d1)); },
        [](etl_dmat& R, etl_dmat& A){ R = etl::transpose(A); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d1,d2)); },
        [](blaze_dmat& R, blaze_dmat& A){ R = trans(A); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d1,d2)); },
        [](eigen_dmat& R, eigen_dmat& A){ R = A.transpose(); }
        );
}

CPM_SECTION_P("R = R'", NARY_POLICY(VALUES_POLICY(64, 64, 128, 256, 256, 256, 300, 512, 512, 1024, 2048, 2048), VALUES_POLICY(64, 128, 128, 128, 256, 384, 500, 512, 1024, 1024, 1024, 2048)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2)); },
        [](etl_dmat& R){ R.transpose_inplace(); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2)); },
        [](blaze_dmat& R){ R.transpose(); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2)); },
        [](eigen_dmat& R){ R.transposeInPlace(); }
        );
}

CPM_SECTION_P("R = A * B", NARY_POLICY(VALUES_POLICY(128, 128, 256, 256, 300, 512, 768), VALUES_POLICY(32, 128, 128, 256, 200, 512, 768), VALUES_POLICY(64, 128, 256, 256, 400, 512, 768)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d2, d3), etl_dmat(d1,d3)); },
        [](etl_dmat& A, etl_dmat& B, etl_dmat& R){ R = A * B; }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d2,d3), blaze_dmat(d1,d3)); },
        [](blaze_dmat& A, blaze_dmat& B, blaze_dmat& R){ R = A * B; }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d2,d3), eigen_dmat(d1,d3)); },
        [](eigen_dmat& A, eigen_dmat& B, eigen_dmat& R){ R = A * B; }
        );
}

CPM_SECTION_P("R = A * B'", NARY_POLICY(VALUES_POLICY(128, 128, 256, 256, 300, 512, 768), VALUES_POLICY(32, 128, 128, 256, 200, 512, 768), VALUES_POLICY(64, 128, 256, 256, 400, 512, 768)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d3, d2), etl_dmat(d1,d3)); },
        [](etl_dmat& A, etl_dmat& B, etl_dmat& R){ R = A * etl::transpose(B); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d3, d2), blaze_dmat(d1,d3)); },
        [](blaze_dmat& A, blaze_dmat& B, blaze_dmat& R){ R = A * blaze::trans(B); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d3, d2), eigen_dmat(d1,d3)); },
        [](eigen_dmat& A, eigen_dmat& B, eigen_dmat& R){ R = A * B.transpose(); }
        );
}

CPM_SECTION_P("R = A * (B + C)", NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2)); },
        [](etl_dmat& R, etl_dmat& A, etl_dmat& B, etl_dmat& C){ R = A * (B + C); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2)); },
        [](blaze_dmat& R, blaze_dmat& A, blaze_dmat& B, blaze_dmat& C){ R = A * (B + C); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2)); },
        [](eigen_dmat& R, eigen_dmat& A, eigen_dmat& B, eigen_dmat& C){ R = A * (B + C); }
        );
}

CPM_SECTION_P("R = A * (B * C)", NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2)); },
        [](etl_dmat& R, etl_dmat& A, etl_dmat& B, etl_dmat& C){ R = A * (B * C); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2)); },
        [](blaze_dmat& R, blaze_dmat& A, blaze_dmat& B, blaze_dmat& C){ R = A * (B * C); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2)); },
        [](eigen_dmat& R, eigen_dmat& A, eigen_dmat& B, eigen_dmat& C){ R = A * (B * C); }
        );
}

CPM_SECTION_P("R = (A + B) * (C - D)", NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2), etl_dmat(d1,d2)); },
        [](etl_dmat& R, etl_dmat& A, etl_dmat& B, etl_dmat& C, etl_dmat& D){ R = (A + B) * (C - D); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2), blaze_dmat(d1,d2)); },
        [](blaze_dmat& R, blaze_dmat& A, blaze_dmat& B, blaze_dmat& C, blaze_dmat& D){ R = (A + B) * (C - D); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2), eigen_dmat(d1,d2)); },
        [](eigen_dmat& R, eigen_dmat& A, eigen_dmat& B, eigen_dmat& C, eigen_dmat& D){ R = (A + B) * (C - D); }
        );
}

CPM_SECTION_P("dot", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dvec(d), etl_dvec(d), etl_dvec(d)); },
        [](etl_dvec& a, etl_dvec& b, etl_dvec& c){ c *= etl::dot(a, b); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dvec(d), blaze_dvec(d), blaze_dvec(d)); },
        [](blaze_dvec& a, blaze_dvec& b, blaze_dvec& c){ c *= (a, b); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dvec(d), eigen_dvec(d), eigen_dvec(d)); },
        [](eigen_dvec& a, eigen_dvec& b, eigen_dvec& c){ c *= a.dot(b); }
        );
}

} //end of anonymous namespace

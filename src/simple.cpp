#include <iostream>
#include <chrono>
#include <random>
#include <numeric>

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

using timer_clock = std::chrono::high_resolution_clock;
using milliseconds = std::chrono::milliseconds;
using microseconds = std::chrono::microseconds;

template<typename T, std::size_t D>
using etl_static_vector = etl::fast_vector<T, D>;

template<typename T, std::size_t D>
using blaze_static_vector = blaze::StaticVector<T, D>;

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

template<template<typename,int,int,int,int,int> class TT, typename T>
struct is_eigen : std::false_type { };

template<template<typename,int,int,int,int,int> class TT, typename S, int I1,int I2,int I3,int I4,int I5>
struct is_eigen<TT, TT<S,I1,I2,I3,I4,I5>> : std::true_type { };

template<typename T, typename std::enable_if<std::is_same<T, blaze::DynamicMatrix<double>>::value, int>::type = 42>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(std::size_t i=0UL; i<container.rows(); ++i ) {
        for(std::size_t j=0UL; j<container.columns(); ++j ) {
            container(i,j) = generator();
        }
    }
}

template<typename T, typename std::enable_if<std::is_same<T, blaze_dyn_vector<double>>::value, int>::type = 42>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(auto& v : container){
        v = generator();
    }
}

template<typename T, typename std::enable_if<is_eigen<Eigen::Matrix, T>::value, int>::type = 42>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(long i=0; i<container.rows(); ++i ) {
        for(long j=0; j<container.cols(); ++j ) {
            container(i,j) = generator();
        }
    }
}

template<typename T, typename std::enable_if<etl::is_etl_expr<T>::value, int>::type = 42>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(auto& v : container){
        v = generator();
    }
}

void b_randomize(){}

template<typename T1, typename... TT>
void b_randomize(T1& container, TT&... containers){
    randomize_double(container);
    b_randomize(containers...);
}

std::string clean_duration(std::string value){
    while(value.size() > 1 && value.back() == '0'){
        value.pop_back();
    }

    return value;
}

std::string duration_str(std::size_t duration_us){
    if(duration_us > 1000 * 1000){
        return clean_duration(std::to_string(duration_us / 1000.0 / 1000.0)) + "s";
    } else if(duration_us > 1000){
        return clean_duration(std::to_string(duration_us / 1000.0)) + "ms";
    } else {
        return std::to_string(duration_us) + "us";
    }
}

template<typename Functor, typename... T>
auto measure_only(Functor&& functor, T&... references){
    for(std::size_t i = 0; i < 100; ++i){
        b_randomize(references...);
        functor();
    }

    std::size_t duration_acc = 0;

    for(std::size_t i = 0; i < 100; ++i){
        b_randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    return duration_acc;
}

template<typename Functor, typename... T>
void measure(const std::string& title, const std::string& reference, Functor&& functor, T&... references){
    std::cout << title << " took " << duration_str(measure_only(functor, references...)) << " (reference: " << reference << ")\n";
}

template<template<typename, std::size_t> class T, std::size_t D>
struct add_static {
    static auto get(){
        T<double, D> a,b,c;
        return measure_only([&a, &b, &c](){c = a + b;}, a, b);
    }
};

template<template<typename> class T, std::size_t D>
struct add_dynamic {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c = a + b;}, a, b);
    }
};

template<template<typename> class T, std::size_t D>
struct add_three {
    static auto get(){
        T<double> a(D), b(D), c(D), r(D);
        return measure_only([&a, &b, &c, &r](){r = a + b + c;}, a, b, c);
    }
};

template<template<typename> class T, std::size_t D>
struct add_four {
    static auto get(){
        T<double> a(D), b(D), c(D), d(D), r(D);
        return measure_only([&a, &b, &c, &d, &r](){r = a + b + c + d;}, a, b, c, d);
    }
};

template<template<typename, std::size_t> class T, std::size_t D>
struct scale_static {
    static auto get(){
        T<double, D> c;
        return measure_only([&c](){c = 3.3;});
    }
};

template<template<typename> class T, std::size_t D>
struct scale_dynamic {
    static auto get(){
        T<double> c(D);
        return measure_only([&c](){c *= 3.3;});
    }
};

CPM_SECTION_P("dot", VALUES_POLICY(500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000))
    CPM_TWO_PASS_NS("etl",
        [](std::size_t d){ return std::make_tuple(etl_dyn_vector<double>(d), etl_dyn_vector<double>(d), etl_dyn_vector<double>(d)); },
        [](etl_dyn_vector<double>& a, etl_dyn_vector<double>& b, etl_dyn_vector<double>& c){ c *= etl::dot(a, b); }
        );

    CPM_TWO_PASS_NS("blaze",
        [](std::size_t d){ return std::make_tuple(blaze_dyn_vector<double>(d), blaze_dyn_vector<double>(d), blaze_dyn_vector<double>(d)); },
        [](blaze_dyn_vector<double>& a, blaze_dyn_vector<double>& b, blaze_dyn_vector<double>& c){ c *= (a, b); }
        );

    CPM_TWO_PASS_NS("eigen",
        [](std::size_t d){ return std::make_tuple(eigen_dyn_vector<double>(d), eigen_dyn_vector<double>(d), eigen_dyn_vector<double>(d)); },
        [](eigen_dyn_vector<double>& a, eigen_dyn_vector<double>& b, eigen_dyn_vector<double>& c){ c *= a.dot(b); }
        );
}

using etl_dmat = etl_dyn_matrix<double>;
using blaze_dmat = blaze_dyn_matrix<double>;
using eigen_dmat = eigen_dyn_matrix<double>;

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

CPM_SECTION_P("R = A * (B * C)", NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)))
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

template<template<typename> class T, std::size_t D>
struct add_complex {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c = a + b + a + b + a + a + b + a + a;}, a, b);
    }
};

template<template<typename> class T, std::size_t D>
struct mix {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1;}, a, b);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2>
struct mix_matrix {
    static auto get(){
        T<double> a(D1, D2), b(D1, D2), c(D1, D2);
        return measure_only([&](){c = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1;}, a, b);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, typename Enable = void>
struct transpose {
    static auto get(){
        T<double> A(D1, D2), R(D2, D1);
        return measure_only([&](){R = etl::transpose(A);}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2>
struct transpose <T, D1, D2, std::enable_if_t<std::is_same<T<double>, blaze::DynamicMatrix<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2), R(D2, D1);
        return measure_only([&](){R = trans(A);}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2>
struct transpose <T, D1, D2, std::enable_if_t<is_eigen<Eigen::Matrix,T<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2), R(D2, D1);
        return measure_only([&](){R = A.transpose();}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, typename Enable = void>
struct transpose_in {
    static auto get(){
        T<double> A(D1, D2);
        return measure_only([&](){A.transpose_inplace();}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2>
struct transpose_in <T, D1, D2, std::enable_if_t<std::is_same<T<double>, blaze::DynamicMatrix<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2);
        return measure_only([&](){A.transpose();}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2>
struct transpose_in <T, D1, D2, std::enable_if_t<is_eigen<Eigen::Matrix,T<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2);
        return measure_only([&](){A.transposeInPlace();}, A);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, std::size_t D3>
struct mmul {
    static auto get(){
        T<double> a(D1, D2), b(D2, D3), c(D1, D3);
        return measure_only([&a, &b, &c](){c = a * b;}, a, b);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, std::size_t D3, typename Enable = void>
struct mmul_t {
    static auto get(){
        T<double> A(D1, D2), B(D3, D2), R(D1, D3);
        return measure_only([&](){R = A * etl::transpose(B);}, A, B);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, std::size_t D3>
struct mmul_t <T, D1, D2, D3, std::enable_if_t<std::is_same<T<double>, blaze::DynamicMatrix<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2), B(D3, D2), R(D1, D3);
        return measure_only([&](){R = A * trans(B);}, A, B);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, std::size_t D3>
struct mmul_t <T, D1, D2, D3, std::enable_if_t<is_eigen<Eigen::Matrix,T<double>>::value>> {
    static auto get(){
        T<double> A(D1, D2), B(D3, D2), R(D1, D3);
        return measure_only([&](){R = A * B.transpose();}, A, B);
    }
};

std::string format(std::string value, std::size_t max){
    return value + (value.size() < max ? std::string(std::max(0UL, max - value.size()), ' ') : "");
}

template<template<template<typename, std::size_t> class, std::size_t> class T, template<typename, std::size_t> class B, template<typename, std::size_t> class E, std::size_t D>
void bench_static(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D), 29) << " | ";
    std::cout << format(duration_str(T<B,D>::get()), 9) << " | ";
    std::cout << format("", 9) << " | ";
    std::cout << format(duration_str(T<E,D>::get()), 9) << " | ";
    std::cout << std::endl;
}

void display_final(std::size_t d, std::size_t min, std::size_t max){
    if(d == min){
        std::cout << "\033[0;32m";
    } else if(d == max){
        std::cout << "\033[0;31m";
    }

    std::cout << format(duration_str(d), 9);

    std::cout << "" << '\033' << "[" << 0 << ";" << 30 << 47 << "m";

    std::cout << " | ";
}

void display(std::size_t d1, std::size_t d2, std::size_t d3){
    auto max = std::max(d1, std::max(d2, d3));
    auto min = std::min(d1, std::min(d2, d3));

    display_final(d1, min, max);
    display_final(d2, min, max);
    display_final(d3, min, max);
}

template<template<template<typename> class, std::size_t, typename...> class T, template<typename> class B, template<typename> class Eg, template<typename> class E, std::size_t D>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D), 29) << " | ";
    display(T<B,D>::get(), T<Eg,D>::get(), T<E,D>::get());
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t, std::size_t, typename...> class T, template<typename> class B, template<typename> class Eg, template<typename> class E, std::size_t D1, std::size_t D2>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D1) + "x" + std::to_string(D2), 29) << " | ";
    display(T<B,D1,D2>::get(), T<Eg,D1,D2>::get(), T<E,D1,D2>::get());
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t, std::size_t, std::size_t, typename...> class T, template<typename> class B, template<typename> class Eg, template<typename> class E, std::size_t D1, std::size_t D2, std::size_t D3>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D1) + "x" + std::to_string(D2) + "x" + std::to_string(D3), 29) << " | ";
    display(T<B,D1,D2,D3>::get(), T<Eg,D1,D2,D3>::get(), T<E,D1,D2,D3>::get());
    std::cout << std::endl;
}

} //end of anonymous namespace

int old_main(){
    std::cout << "| Name                          | Blaze     | Eigen     |  ETL      |" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;

    //Transposition

    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 64, 64>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 64, 128>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128, 64>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128, 128>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256, 256>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256, 384>("R = A'");
    bench_dyn<transpose, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 512, 512>("R = A'");

    //In place transposition

    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 64, 64>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 64, 128>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128, 64>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128, 128>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256, 256>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256, 384>("A = A'");
    bench_dyn<transpose_in, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 512, 512>("A = A'");

    bench_dyn<add_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 32768>("r = a + b");
    bench_dyn<add_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 32768>("r = a + b");
    bench_dyn<add_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 32768>("r = a + b");
    bench_dyn<add_three, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 32768>("r = a + b + c");
    bench_dyn<add_three, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 32768>("r = a + b + c");
    bench_dyn<add_three, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 32768>("r = a + b + c");
    bench_dyn<add_four, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 32768>("r = a + b + c + d");
    bench_dyn<add_four, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 32768>("r = a + b + c + d");
    bench_dyn<add_four, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 32768>("r = a + b + c + d");
    bench_dyn<scale_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 8 * 1024 * 1024>("r *= 3.3");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128,32,64>("C = A * B");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128,128,128>("C = A * B");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256,128,256>("C = A * B");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256,256,256>("C = A * B");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 300,200,400>("C = A * B");
    bench_dyn<mmul, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 512,512,512>("C = A * B");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128,32,64>("C = A * B'");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 128,128,128>("C = A * B'");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256,128,256>("C = A * B'");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256,256,256>("C = A * B'");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 300,200,400>("C = A * B'");
    bench_dyn<mmul_t, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 512,512,512>("C = A * B'");
    bench_dyn<add_complex, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_add_complex");
    bench_dyn<mix, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_mix");
    bench_dyn<mix, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_mix");
    bench_dyn<mix, blaze_dyn_vector, eigen_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_mix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 256, 256>("dynamic_mix_matrix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 512, 512>("dynamic_mix_matrix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, eigen_dyn_matrix, etl_dyn_matrix, 578, 769>("dynamic_mix_matrix");

    //bench_static<add_static, blaze_static_vector, etl_static_vector, 8192>("static_add");
    //bench_static<scale_static, blaze_static_vector, etl_static_vector, 8192>("static_scale");

    std::cout << "------------------------------------------------------------------" << std::endl;

    return 0;
}

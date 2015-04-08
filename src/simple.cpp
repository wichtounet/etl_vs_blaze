#include <iostream>
#include <chrono>
#include <random>
#include <numeric>

#include <blaze/Math.h>

#include "etl/etl.hpp"

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::microseconds microseconds;

namespace {

template<typename T, std::enable_if_t<std::is_same<T, blaze::DynamicMatrix<double>>::value, int> = 42>
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

template<typename T, std::enable_if_t<!std::is_same<T, blaze::DynamicMatrix<double>>::value, int> = 42>
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
        return clean_duration(std::to_string(duration_us)) + "us";
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

template<template<typename> class T, std::size_t D, typename Enable = void>
struct dot {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c *= etl::dot(a, b);}, a, b);
    }
};

template<template<typename> class T, std::size_t D>
struct dot<T, D, std::enable_if_t<std::is_same<T<double>, blaze::DynamicVector<double>>::value>> {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c *= (a,b);}, a, b);
    }
};

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
        return measure_only([&a, &b, &c](){c = a + a * 5.9 + a + b - b / 2.3 - a + b * 1.1;}, a, b);
    }
};

template<template<typename> class T, std::size_t D>
struct smart_1 {
    static auto get(){
        T<double> A(D, D), B(D, D), C(D, D), R(D, D);
        return measure_only([&A, &B, &C, &R](){R = A * (B + C);}, A, B, C);
    }
};

template<template<typename> class T, std::size_t D>
struct smart_2 {
    static auto get(){
        T<double> A(D, D), B(D, D), C(D, D), R(D, D);
        return measure_only([&A, &B, &C, &R](){R = A * (B * C);}, A, B, C);
    }
};

template<template<typename> class T, std::size_t D>
struct smart_3 {
    static auto get(){
        T<double> A(D, D), B(D, D), C(D, D), DD(D, D), R(D, D);
        return measure_only([&A, &B, &C, &DD, &R](){R = (A + B) * (C - DD);}, A, B, C, DD);
    }
};

template<template<typename> class T, std::size_t D1, std::size_t D2, std::size_t D3>
struct mmul {
    static auto get(){
        T<double> a(D1, D2), b(D2, D3), c(D1, D3);
        return measure_only([&a, &b, &c](){c = a * b;}, a, b);
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
    std::cout << format(duration_str(T<E,D>::get()), 9) << " | ";
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t> class T, template<typename> class B, template<typename> class E, std::size_t D>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D), 29) << " | ";
    std::cout << format(duration_str(T<B,D>::get()), 9) << " | ";
    std::cout << format(duration_str(T<E,D>::get()), 9) << " | ";
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t, typename = void> class T, template<typename> class B, template<typename> class E, std::size_t D>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D), 29) << " | ";
    std::cout << format(duration_str(T<B,D>::get()), 9) << " | "; std::cout << format(duration_str(T<E,D>::get()), 9) << " | ";
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t, std::size_t> class T, template<typename> class B, template<typename> class E, std::size_t D1, std::size_t D2>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D1) + "x" + std::to_string(D2), 29) << " | ";
    std::cout << format(duration_str(T<B,D1,D2>::get()), 9) << " | ";
    std::cout << format(duration_str(T<E,D1,D2>::get()), 9) << " | ";
    std::cout << std::endl;
}

template<template<template<typename> class, std::size_t, std::size_t, std::size_t, typename...> class T, template<typename> class B, template<typename> class E, std::size_t D1, std::size_t D2, std::size_t D3>
void bench_dyn(const std::string& title){
    std::cout << "| ";
    std::cout << format(title + ":" + std::to_string(D1) + "x" + std::to_string(D2) + "x" + std::to_string(D3), 29) << " | ";
    std::cout << format(duration_str(T<B,D1,D2,D3>::get()), 9) << " | ";
    std::cout << format(duration_str(T<E,D1,D2,D3>::get()), 9) << " | ";
    std::cout << std::endl;
}

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

} //end of anonymous namespace

int main(){
    std::cout << "| Name                          | Blaze     |  ETL      |" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    bench_dyn<smart_1, blaze_dyn_matrix, etl_dyn_matrix, 64>("R = A * (B + C)");
    bench_dyn<smart_1, blaze_dyn_matrix, etl_dyn_matrix, 128>("R = A * (B + C)");
    bench_dyn<smart_2, blaze_dyn_matrix, etl_dyn_matrix, 64>("R = A * (B * C)");
    bench_dyn<smart_2, blaze_dyn_matrix, etl_dyn_matrix, 128>("R = A * (B * C)");
    bench_dyn<smart_3, blaze_dyn_matrix, etl_dyn_matrix, 64>("R = (A + B) * (C - D)");
    bench_dyn<smart_3, blaze_dyn_matrix, etl_dyn_matrix, 128>("R = (A + B) * (C - D)");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 1 * 32768>("r = a + b");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 2 * 32768>("r = a + b");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 4 * 32768>("r = a + b");
    bench_dyn<scale_dynamic, blaze_dyn_vector, etl_dyn_vector, 1 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, etl_dyn_vector, 2 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, etl_dyn_vector, 4 * 1024 * 1024>("r *= 3.3");
    bench_dyn<scale_dynamic, blaze_dyn_vector, etl_dyn_vector, 8 * 1024 * 1024>("r *= 3.3");
    bench_dyn<dot, blaze_dyn_vector, etl_dyn_vector, 1024 * 1024>("dot");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_add_complex");
    bench_dyn<mix, blaze_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_mix");
    bench_dyn<mix, blaze_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_mix");
    bench_dyn<mix, blaze_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_mix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, etl_dyn_matrix, 256, 256>("dynamic_mix_matrix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, etl_dyn_matrix, 512, 512>("dynamic_mix_matrix");
    bench_dyn<mix_matrix, blaze_dyn_matrix, etl_dyn_matrix, 578, 769>("dynamic_mix_matrix");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 128,32,64>("dynamic_mmul");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 128,128,128>("dynamic_mmul");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 256,128,256>("dynamic_mmul");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 256,256,256>("dynamic_mmul");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 300,200,400>("dynamic_mmul");
    bench_dyn<mmul, blaze_dyn_matrix, etl_dyn_matrix, 512,512,512>("dynamic_mmul");

    bench_static<add_static, blaze_static_vector, etl_static_vector, 8192>("static_add");
    bench_static<scale_static, blaze_static_vector, etl_static_vector, 8192>("static_scale");

    std::cout << "---------------------------------------------------------" << std::endl;

    return 0;
}

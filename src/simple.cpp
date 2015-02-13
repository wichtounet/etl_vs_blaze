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

template<typename T>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(auto& v : container){
        v = generator();
    }
}

template<typename T1>
void b_randomize(T1& container){
    randomize_double(container);
}

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

template<template<typename> class T, std::size_t D>
struct add_complex {
    static auto get(){
        T<double> a(D), b(D), c(D);
        return measure_only([&a, &b, &c](){c = a + b + a + b + a + a + b + a + a;}, a, b);
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

template<typename T, std::size_t D>
using etl_static_vector = etl::fast_vector<T, D>;

template<typename T, std::size_t D>
using blaze_static_vector = blaze::StaticVector<T, D>;

template<typename T>
using etl_dyn_vector = etl::dyn_vector<T>;

template<typename T>
using blaze_dyn_vector = blaze::DynamicVector<T>;

} //end of anonymous namespace

int main(){
    std::cout << "| Name                          | Blaze     |  ETL      |" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    bench_static<add_static, blaze_static_vector, etl_static_vector, 8192>("static_add");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_add");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_add");
    bench_dyn<add_dynamic, blaze_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_add");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 1 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 2 * 32768>("dynamic_add_complex");
    bench_dyn<add_complex, blaze_dyn_vector, etl_dyn_vector, 4 * 32768>("dynamic_add_complex");

    std::cout << "---------------------------------------------------------" << std::endl;

    return 0;
}

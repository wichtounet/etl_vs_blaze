#include <iostream>
#include <chrono>
#include <random>

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

std::string duration_str(std::size_t duration_us){
    double duration = duration_us;

    if(duration > 1000 * 1000){
        return std::to_string(duration / 1000.0 / 1000.0) + "s";
    } else if(duration > 1000){
        return std::to_string(duration / 1000.0) + "ms";
    } else {
        return std::to_string(duration_us) + "us";
    }
}

template<typename Functor, typename... T>
void measure(const std::string& title, const std::string& reference, Functor&& functor, T&... references){
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

    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")\n";
}

template<std::size_t D>
void bench_blaze_add_static(const std::string& reference){
    blaze::StaticVector<double, D> a;
    blaze::StaticVector<double, D> b;
    blaze::StaticVector<double, D> c;

    measure("blaze_add_static(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b;
    }, a, b);
}

template<std::size_t D>
void bench_etl_add_static(const std::string& reference){
    etl::fast_vector<double, D> a;
    etl::fast_vector<double, D> b;
    etl::fast_vector<double, D> c;

    measure("etl_add_static(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b;
    }, a, b);
}

template<std::size_t D>
void bench_blaze_add_dynamic(const std::string& reference){
    blaze::DynamicVector<double> a(D);
    blaze::DynamicVector<double> b(D);
    blaze::DynamicVector<double> c(D);

    measure("blaze_add_dynamic(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b;
    }, a, b);
}

template<std::size_t D>
void bench_etl_add_dynamic(const std::string& reference){
    etl::dyn_vector<double> a(D);
    etl::dyn_vector<double> b(D);
    etl::dyn_vector<double> c(D);

    measure("etl_add_dynamic(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b;
    }, a, b);
}

template<std::size_t D>
void bench_blaze_add_dynamic_complex(const std::string& reference){
    blaze::DynamicVector<double> a(D);
    blaze::DynamicVector<double> b(D);
    blaze::DynamicVector<double> c(D);

    measure("blaze_add_dynamic_complex(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b + a + b + a + b;
    }, a, b);
}

template<std::size_t D>
void bench_etl_add_dynamic_complex(const std::string& reference){
    etl::dyn_vector<double> a(D);
    etl::dyn_vector<double> b(D);
    etl::dyn_vector<double> c(D);

    measure("etl_add_dynamic_complex(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + b + a + b + a + b;
    }, a, b);
}

} //end of anonymous namespace

int main(){
    bench_blaze_add_static<8192>("0us");
    bench_etl_add_static<8192>("0us");

    bench_blaze_add_dynamic<65536>("4.8ms");
    bench_etl_add_dynamic<65536>("4.9ms");

    bench_blaze_add_dynamic<2*65536>("4.8ms");
    bench_etl_add_dynamic<2*65536>("4.9ms");

    bench_blaze_add_dynamic_complex<65536>("8.2ms");
    bench_etl_add_dynamic_complex<65536>("53ms");

    bench_blaze_add_dynamic_complex<2*65536>("8.2ms");
    bench_etl_add_dynamic_complex<2*65536>("53ms");

    return 0;
}

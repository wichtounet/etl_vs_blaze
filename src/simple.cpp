#include <iostream>
#include <blaze/Math.h>

#include <chrono>
#include <random>

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
void bench_blaze_add(const std::string& reference){
    blaze::StaticVector<double, D> a;
    blaze::StaticVector<double, D> b;
    blaze::StaticVector<double, D> c;

    measure("fast_vector_simple(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = a + a + b + a;
    }, a, b);
}

} //end of anonymous namespace

int main(){
    bench_blaze_add<4096>("11ms");

    return 0;
}

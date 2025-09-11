#include <iostream>
#include <thread>

int main() {
    try {
        unsigned int idleTime = 30;
        std::cout << "Take a break of " << idleTime << " seconds... " << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(idleTime));
        std::cout << "finished!" << std::endl << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
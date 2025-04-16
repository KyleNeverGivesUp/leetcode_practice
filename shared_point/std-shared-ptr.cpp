#include <iostream>
#include <vector>

int modifyByPointer(int p) {
    p = 100;
    return p;
}


void modifyByReference(int& r) {
    r = 200;
}

int main() {
    // Create three shared_ptr instances, each managing a separate int
    std::shared_ptr<int> sp1 = std::make_shared<int>(10);
    std::shared_ptr<int> sp2 = std::make_shared<int>(20);
    std::shared_ptr<int> sp3 = std::make_shared<int>(30);

    // Create a vector that stores regular int values
    std::vector<int> vec;
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    // Modify the values pointed to by the shared_ptr instances
    *sp1 = 100;
    *sp2 = 200;
    *sp3 = 300;

    // Modify the values inside the vector
    vec[0] = 1000;
    vec[1] = 2000;
    vec[2] = 3000;

    // Print the values from shared_ptr instances
    std::cout << "Shared_ptr values:" << std::endl;
    std::cout << "*sp1 = " << *sp1 << std::endl;
    std::cout << "*sp2 = " << *sp2 << std::endl;
    std::cout << "*sp3 = " << *sp3 << std::endl;

    // Print the values from the vector
    std::cout << "\nVector values:" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "vec[" << i << "] = " << vec[i] << std::endl;
    }

    // Print the reference counts of the shared_ptr instances
    std::cout << "\nReference count of sp1 = " << sp1.use_count() << std::endl;
    std::cout << "Reference count of sp2 = " << sp2.use_count() << std::endl;
    std::cout << "Reference count of sp3 = " << sp3.use_count() << std::endl;

        int x = 42;
        int& ref = x; // 把 x 的地址赋值给 raw
        int* raw = &x; // 把 x 的地址赋值给 raw
        std::cout << "x: " << x << std::endl; 
        std::cout << "&x: " << &x << std::endl; 
        std::cout << "raw: " << raw << std::endl; 
        std::cout << "*raw: " << *raw << std::endl; 
        std::cout << "ref: " << ref << std::endl; 
        std::cout << std::string(20,'-') << std::endl;
        ref=40;
        std::cout << "x: " << x << std::endl; 
        std::cout << "&x: " << &x << std::endl; 
        std::cout << "raw: " << raw << std::endl; 
        std::cout << "*raw: " << *raw << std::endl; 
        std::cout << "ref: " << ref << std::endl; 
        std::cout << std::string(20,'-') << std::endl;
        int y=60;
        // auto &ref2 = x;
        ref = y;
        std::cout << "x: " << x << std::endl; 
        std::cout << "&x: " << &x << std::endl; 
        std::cout << "raw: " << raw << std::endl; 
        std::cout << "*raw: " << *raw << std::endl; 
        std::cout << "ref: " << ref << std::endl; 
        std::cout << "&ref: " << &ref << std::endl; 
        // std::cout << "ref2: " << ref2 << std::endl; 
        // std::cout << "&ref2: " << &ref2 << std::endl; 
        raw = &y;

        char* a = "ab";
        std::cout << "a: " << a << std::endl;


    int x = 10;

    std::cout << "Original x: " << x << std::endl;

    x=modifyByPointer(x);  // point
    std::cout << "After modifyByPointer: " << x << std::endl;

    modifyByReference(x); // reference
    std::cout << "After modifyByReference: " << x << std::endl;
        
    
    return 0;
}


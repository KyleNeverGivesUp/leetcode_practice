#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <typeinfo>
#include <regex>

std::vector<std::string> extractOperatorsInReverse(const std::string& input) {
    std::vector<std::string> operators;
std::regex operatorPattern(R"(([a-zA-Z_]+\d+)\()"); // Match patterns like mat_mul1(, mat_add2(, etc.
std::smatch match;

std::string::const_iterator searchStart(input.cbegin());
while (std::regex_search(searchStart, input.cend(), match, operatorPattern)) {
operators.push_back(match[1]); // Extract the operator name
searchStart = match.suffix().first; // Move the search start position
}

// Reverse the order of operators to match actual execution order
std::reverse(operators.begin(), operators.end());
return operators;
}

int main(){
    std::vector<std::string> abc;
    // abc.
    abc.push_back("hello");
    abc.push_back("world");
    
    for ( std::string s : abc) {
        std::cout << s << std::endl;
    }
    abc.resize(1);
    for ( std::string s : abc) {
        std::cout << s << std::endl;
    }

    std::unordered_map<std::string, std::vector<std::vector<std::string>>> v;
    std::cout << typeid(v).name() << std::endl;

    // std::string modelInput = "softmax3(mat_add3(mat_mul3(relu2(mat_add2(mat_mul2(relu1(mat_add1(mat_mul1(features)))))))))";
    std::string modelInput = "softmax4(mat_add4(mat_mul4(features)))";
    std::cout << "Extracting model operators" << std::endl;
    std::vector<std::string> operators = extractOperatorsInReverse(modelInput);
    for ( auto& i : operators){
        std::cout << i << std::endl;
    }
    
}
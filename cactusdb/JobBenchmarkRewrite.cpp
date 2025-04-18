#include <folly/init/Init.h>
//#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cctype>
#include <climits>
#include <unordered_map>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <json/json.h>
#include "velox/type/Type.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include <ctime>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <locale>
#include <regex>
#include <algorithm>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <malloc.h>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

/*
 * The structure to describe the context of a neural network model architecture
 */
struct NNModelContext {
  int inputFeatures;

  int numLayers;

  int hiddenLayerNeurons;

  int outputLayerNeurons;
};

/*
 * The structure to describe the context of a decision tree model architecture
 */

struct DTModelContext {
  int inputFeatures;

  int treeDepth;
};


/*
 * The structure to describe the push down status of a feature
 */
struct FeatureStatus {
    int isFeature;

    int isPushed;

    int vectorSize;

    int featureStartPos;

};


class VectorAddition : public MLFunction {
 public:
  VectorAddition(int inputDims) {
    dims.push_back(inputDims);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    /*auto input_elements1 = args[0]->as<ArrayVector>()->elements();
    float* input1Values = input_elements1->values()->asMutable<float>();

    auto input_elements2 = args[1]->as<ArrayVector>()->elements();
    float* input2Values = input_elements2->values()->asMutable<float>();*/

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = decodedRightArray->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, dims[0]);

    std::vector<std::vector<float>> results;

    /*for (int i = 0; i < numInput; i++) {
      //Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> vSum = input1Matrix.row(i) + input2Matrix.row(i);
      Eigen::VectorXf vSum = input1Matrix.row(i) + input2Matrix.row(i);
      std::vector<float> curVec(vSum.data(), vSum.data() + vSum.size());
      //std::vector<float> std_vector(vSum.data(), vSum.data() + vSum.size());
      results.push_back(curVec);
    }*/

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sumMat  =  input1Matrix + input2Matrix;
    for (int i = 0; i < numInput; i++) {
        std::vector<float> curVec(
            sumMat.row(i).data(),
            sumMat.row(i).data() + sumMat.cols());
        results.push_back(curVec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "vector_addition";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 //private:
 // int inputDims_;
};




class GetFeatureVec : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int64_t vecSizeLarge = 0;
    if (args.size() == 2) {
        // an optional parameter can be passed to enable the GPU for mat_mul
        vecSizeLarge = args[1]->as<ConstantVector<int64_t>>()->valueAt(0);
    }
    int vecSize = static_cast<int>(vecSizeLarge);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> vec;

        for (int j = 0; j < vecSize; j++) {
            if (j % 2 == 0)
                vec.push_back(1.0);
            else
                vec.push_back(0.0);
        }
        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("BIGINT")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_feature_vec";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 //private:
 // int inputDims_;
};



class JobBenchmarkRewrite : HiveConnectorTestBase {
 public:
  JobBenchmarkRewrite() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();
    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();
    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~JobBenchmarkRewrite() {}

  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};

  std::unordered_map<std::string, RowVectorPtr> tableName2RowVector;
  std::unordered_map<std::string, std::vector<std::vector<float>>> operatorParam2Weights;
  std::unordered_map<std::string, std::vector<std::string>> tabel2Columns;
  std::unordered_map<std::string, FeatureStatus> allFeatureStatus;
  std::vector<std::string> modelOperators;
  std::map<std::string, int> factorizationPlans;
  std::map<std::string, int> featureStartPos;
  std::map<std::string, std::string> alisToTable;
  int totalFeatures = 0;

  // Function to check if a string represents a valid integer
  bool isInteger(const std::string& s) {
    if (s.empty() || (s.size() > 1 && s[0] == '0')) {
      return false; // prevent leading zeros for non-zero integers
    }

    for (char c : s) {
      if (!std::isdigit(c)) {
        return false;
      }
    }

    return true;
  }


  int getStringIndex(const std::vector<std::string>& strVec, const std::string& target) {
    auto it = std::find(strVec.begin(), strVec.end(), target);

    if (it != strVec.end()) {
        // Calculate the index
        int index = std::distance(strVec.begin(), it);
        return index;
    } else {
        return -1;
    }

  }


      std::string readQueryDetails(const std::string& filePath) {
            // Open the file
            std::ifstream file(filePath);

            // Check if the file is opened successfully
            if (!file.is_open()) {
                std::cerr << "Error: Could not open the file: " << filePath << std::endl;
                return "1";
            }

            // Read the file content
            std::ostringstream oss;
            oss << file.rdbuf(); // Stream file content to the stringstream
            std::string fileContent = oss.str(); // Convert the stream to a std::string

            // Close the file
            file.close();

            return fileContent;
      }


  void findFactorizationPlans(const std::string& filePath) {
            // Open the file
            std::ifstream inputFile(filePath);
            if (!inputFile.is_open()) {
                std::cerr << "Failed to open the file: " << filePath << std::endl;
                //return 1;
            }

            // Map to store key-value pairs
            std::map<std::string, int> myMap;

            // Read the file line by line
            std::string line;
            while (std::getline(inputFile, line)) {
                // Find the separator " = " to split the key and value
                size_t separatorPos = line.find(" = ");
                if (separatorPos != std::string::npos) {
                    // Extract the key and value
                    std::string key = line.substr(0, separatorPos);
                    int value = std::stoi(line.substr(separatorPos + 3)); // Convert value to int

                    // Store in the map
                    myMap[key] = value;
                }
            }

            // Close the file
            inputFile.close();

            factorizationPlans = myMap;
        }




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




  std::vector<std::vector<float>> loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
            H5::H5File file(filename, H5F_ACC_RDONLY);
            H5::DataSet dataset = file.openDataSet(datasetName);
            H5::DataSpace dataspace = dataset.getSpace();

            // Get the number of dimensions
            int rank = dataspace.getSimpleExtentNdims();
            // std::cout << "Rank: " << rank << std::endl;

            // Allocate space for the dimensions
            std::vector<hsize_t> dims(rank);

            // Get the dataset dimensions
            dataspace.getSimpleExtentDims(dims.data(), nullptr);

            size_t rows;
            size_t cols;

            if (rank == 1) {
                rows = dims[0];
                cols = 1;
            }
            else if (rank == 2) {
                rows = dims[0];
                cols = dims[1];
            } else {
                throw std::runtime_error("Unsupported rank: " + std::to_string(rank));
            }

            // Read data into a 1D vector
            std::vector<float> flatData(rows * cols);
            dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

            // Convert to 2D vector
            std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = flatData[i * cols + j];
                    if (doPrint == 1)
                        std::cout << result[i][j] << ", ";
                }
                if (doPrint == 1)
                    std::cout << std::endl;
            }

            // Close the dataset and file
            dataset.close();
            file.close();

            return result;
        }



  void findWeights(const std::string& modelPath) {
            // read the parameter weights from file
            std::vector<std::vector<float>> w1 = loadHDF5Array(modelPath, "fc1.weight", 0);
            std::vector<std::vector<float>> b1 = loadHDF5Array(modelPath, "fc1.bias", 0);
            std::vector<std::vector<float>> w2 = loadHDF5Array(modelPath, "fc2.weight", 0);
            std::vector<std::vector<float>> b2 = loadHDF5Array(modelPath, "fc2.bias", 0);
            std::vector<std::vector<float>> w3 = loadHDF5Array(modelPath, "fc3.weight", 0);
            std::vector<std::vector<float>> b3 = loadHDF5Array(modelPath, "fc3.bias", 0);

            // store the weights in map with same name as operator
            operatorParam2Weights["mat_mul1"] = w1;
            operatorParam2Weights["mat_add1"] = b1;
            operatorParam2Weights["mat_mul2"] = w2;
            operatorParam2Weights["mat_add2"] = b2;
            operatorParam2Weights["mat_mul3"] = w3;
            operatorParam2Weights["mat_add3"] = b3;

            std::cout << "Shape of mat_mul1 weight: " << w1.size() << ", " << w1[0].size() << std::endl;
            std::cout << "Shape of mat_add1 weight: " << b1.size() << std::endl;
            std::cout << "Shape of mat_mul2 weight: " << w2.size() << ", " << w2[0].size() << std::endl;
            std::cout << "Shape of mat_add2 weight: " << b2.size() << std::endl;
            std::cout << "Shape of mat_mul3 weight: " << w3.size() << ", " << w3[0].size() << std::endl;
            std::cout << "Shape of mat_add3 weight: " << b3.size() << std::endl;
        }


  std::vector<std::vector<float>> extractSubweight(const std::vector<std::vector<float>>& matrix, int start, int n) {
            std::vector<std::vector<float>> result;

            std::cout << "Extracting subweight" << std::endl;
            std::cout << start << ", " << n << std::endl;

            // Ensure that the range [start, start + n) is within bounds
            //int end = std::min(start + n, matrix.size());
            int end = start + n;
            for (int i = start; i < end; ++i) {
                result.push_back(matrix[i]);  // Copy rows within the range
            }

            return result;
        }


  RowVectorPtr getTableFromCSVFile(
      VectorMaker& maker,
      std::string csvFilePath,
      std::string tableName,
      int k) {
    std::ifstream file(csvFilePath.c_str());
    if (file.fail()) {
      std::cerr << "Error in reading data file:" << csvFilePath << std::endl;
      exit(1);
    }

    std::cout << tableName << std::endl;

    std::unordered_map<std::string, std::vector<string>> colName2colHeader;

    std::unordered_map<std::string, std::vector<int>> colName2colType;

    colName2colType["aka_name"] = {0, 0, 1, 1, 1, 1, 1, 1};
    colName2colType["aka_title"] = {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["cast_info"] = {0, 0, 0, 0, 1, 0, 0};
    colName2colType["char_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["comp_cast_type"] = {0, 1};
    colName2colType["company_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["company_type"] = {0, 1};
    colName2colType["complete_cast"] = {0, 0, 0, 0};
    colName2colType["info_type"] = {0, 1};
    colName2colType["keyword"] = {0, 1, 1};
    colName2colType["kind_type"] = {0, 1};
    colName2colType["link_type"] = {0, 1};
    colName2colType["movie_companies"] = {0, 0, 0, 0, 1};
    colName2colType["movie_info_idx"] = {0, 0, 0, 1, 1};
    colName2colType["movie_keyword"] = {0, 0, 0};
    colName2colType["movie_link"] = {0, 0, 0, 0};
    colName2colType["name"] = {0, 1, 1, 0, 1, 1, 1, 1, 1};
    colName2colType["role_type"] = {0, 1};
    colName2colType["title"] = {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["movie_info"] = {0, 0, 0, 1, 1};
    colName2colType["person_info"] = {0, 0, 0, 1, 1};

    std::vector<std::string> aka_name_columns = {
        "id",
        "person_id",
        "name",
        "imdb_index",
        "name_pcode_cf",
        "name_pcode_ndf",
        "surname_pcode",
        "md5sum",
        "an_features"};

    colName2colHeader["aka_name"] = aka_name_columns;

    std::vector<std::string> aka_title_columns = {
        "id",
        "movie_id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "note",
        "md5sum",
        "at_features"};

    colName2colHeader["aka_title"] = aka_title_columns;

    std::vector<std::string> cast_info_columns = {
        "id",
        "person_id",
        "movie_id",
        "person_role_id",
        "note",
        "nr_order",
        "role_id",
        "ci_features"};

    colName2colHeader["cast_info"] = cast_info_columns;

    std::vector<std::string> char_name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "chn_features"};

    colName2colHeader["char_name"] = char_name_columns;

    std::vector<std::string> comp_cast_type_columns = {
        "id", "kind", "cct_features"};

    colName2colHeader["comp_cast_type"] = comp_cast_type_columns;

    std::vector<std::string> company_name_columns = {
        "id",
        "name",
        "country_code",
        "imdb_id",
        "name_pcode_nf",
        "name_pcode_sf",
        "md5sum",
        "cn_features"};

    colName2colHeader["company_name"] = company_name_columns;

    std::vector<std::string> company_type_columns = {
        "id", "kind", "ct_features"};

    colName2colHeader["company_type"] = company_type_columns;

    std::vector<std::string> complete_cast_columns = {
        "id", "movie_id", "subject_id", "status_id", "cc_features"};

    colName2colHeader["complete_cast"] = complete_cast_columns;

    std::vector<std::string> info_type_columns = {"id", "info", "it_features"};

    colName2colHeader["info_type"] = info_type_columns;

    std::vector<std::string> keyword_columns = {
        "id", "keyword", "phonetic_code", "k_features"};

    colName2colHeader["keyword"] = keyword_columns;

    std::vector<std::string> kind_type_columns = {"id", "kind", "kt_features"};

    colName2colHeader["kind_type"] = kind_type_columns;

    std::vector<std::string> link_type_columns = {"id", "link", "lt_features"};

    colName2colHeader["link_type"] = link_type_columns;

    std::vector<std::string> movie_companies_columns = {
        "id",
        "movie_id",
        "company_id",
        "company_type_id",
        "note",
        "mc_features"};

    colName2colHeader["movie_companies"] = movie_companies_columns;

    std::vector<std::string> movie_info_idx_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mii_features"};

    colName2colHeader["movie_info_idx"] = movie_info_idx_columns;

    std::vector<std::string> movie_keyword_columns = {
        "id", "movie_id", "keyword_id", "mk_features"};

    colName2colHeader["movie_keyword"] = movie_keyword_columns;

    std::vector<std::string> movie_link_columns = {
        "id", "movie_id", "linked_movie_id", "link_type_id", "ml_features"};

    colName2colHeader["movie_link"] = movie_link_columns;

    std::vector<std::string> name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "gender",
        "name_pcode_cf",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "n_features"};

    colName2colHeader["name"] = name_columns;

    std::vector<std::string> role_type_columns = {"id", "role", "rt_features"};

    colName2colHeader["role_type"] = role_type_columns;

    std::vector<std::string> title_columns = {
        "id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "imdb_id",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "series_years",
        "md5sum",
        "t_features"};

    colName2colHeader["title"] = title_columns;

    std::vector<std::string> movie_info_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mi_features"};

    colName2colHeader["movie_info"] = movie_info_columns;

    std::vector<std::string> person_info_columns = {
        "id", "person_id", "info_type_id", "info", "note", "pi_features"};

    colName2colHeader["person_info"] = person_info_columns;

    std::string line;

    std::vector<std::vector<int>> intCols;

    std::vector<std::vector<std::string>> stringCols;

    std::vector<int> colTypeIndex = colName2colType[tableName];

    std::vector<int> colIndexInType;

    int colIndex = 0;

    std::string cell;

    int numRows = 0;

    while (std::getline(file, line)) {
      // std::cout << line << std::endl;

      // analyze the first line
      std::stringstream iss(line);

      bool fragmentFlag = false;

      std::string fragmentedStr;

      colIndex = 0;

      // The JOB tables only have two types of columns: integer and string
      while (std::getline(iss, cell, ',')) {
        if ((fragmentFlag == false) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = true;

          fragmentedStr = ",";

          continue;

        } else if (
            (fragmentFlag == true) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = false;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (
            (fragmentFlag == false) && (cell[0] == '"') &&
            ((cell[cell.size() - 1] != '"') ||
             ((cell[cell.size() - 1] == '"') &&
              (cell[cell.size() - 2] == '\\')))) {
          fragmentFlag = true;

          fragmentedStr = cell;

          continue;

        } else if (
            (fragmentFlag == true) && (cell[0] != '"') &&
            (cell[cell.size() - 1] == '"') && (cell[cell.size() - 2] != '\\')) {
          fragmentFlag = false;

          fragmentedStr += cell;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (fragmentFlag == true) {
          fragmentedStr += cell;

          continue;
        }

        // std::cout << colIndex << ":" << cell << std::endl;

        if (!fragmentFlag) {
          if (!colTypeIndex[colIndex]) {
            // this is an integer column

            if (numRows == 0) {
              if (cell == "")

                intCols.push_back(std::vector<int>{INT_MIN});

              else

                intCols.push_back(std::vector<int>{stoi(cell)});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              if (cell == "")

                intCols[vecIndex].push_back(INT_MIN);

              else

                intCols[vecIndex].push_back(stoi(cell));
            }

          } else {
            // this is a string column

            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{cell});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              stringCols[vecIndex].push_back(cell);
            }
          }

          colIndex++;
        }
      }

      if (colIndex < colTypeIndex.size()) {
        // std::cout << "colIndex:" << colIndex << std::endl;
        // std::cout << "colTypeIndex.size():"<< colTypeIndex.size() <<
        // std::endl;

        for (int i = colIndex; i < colTypeIndex.size(); i++) {
          if (!colTypeIndex[i]) {
            if (numRows == 0) {
              intCols.push_back(std::vector<int>{INT_MIN});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              intCols[vecIndex].push_back(INT_MIN);
            }

          } else {
            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{""});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              stringCols[vecIndex].push_back("");
            }
          }
        }
      }

      colIndex = colTypeIndex.size();

      /*if (numRows == 0) {

          for (int i = 0; i < colIndex; i++) {

               std::cout << colTypeIndex[i] << ":" << colIndexInType[i] <<
      std::endl;

          }

      }*/

      numRows++;
    }

    std::vector<VectorPtr> vecs;

    std::cout << "Building RowVector for this table with " << colIndex
              << " columns and " << numRows << " rows." << std::endl;

    for (int i = 0; i < colIndex; i++) {
      int type = colTypeIndex[i];

      int vecIndex = colIndexInType[i];

      // std::cout << i << ":" << type << ":" << vecIndex << std::endl;

      if (!type) {
        auto vec = maker.flatVector<int>(intCols[vecIndex]);

        vecs.push_back(vec);

      } else {
        auto vec = maker.flatVector<std::string>(stringCols[vecIndex]);

        vecs.push_back(vec);
      }
    }

    // to create the last column, which is a feature vector of length k

    if (k < 0)
      k = 8;

    std::vector<std::vector<float>> inputVectors;

    for (int i = 0; i < numRows; i++) {
      std::vector<float> inputVector;

      for (int j = 0; j < k; j++) {
        if (j % 2 == 0)

          inputVector.push_back(1.0);

        else

          inputVector.push_back(0.0);
      }

      inputVectors.push_back(inputVector);
    }

    auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

    vecs.push_back(inputArrayVector);

    RowVectorPtr myRowVector =
        maker.rowVector(colName2colHeader[tableName], vecs);

    return myRowVector;
  }

  int sampleQuery() {
    return 29;
  }

  int sampleModel() {
    return 0;
  }

  void sampleNNModelArch(int numInputFeatures, NNModelContext& nn) {
    nn.inputFeatures = numInputFeatures;
    nn.numLayers = 3;
    nn.hiddenLayerNeurons = 16;
    nn.outputLayerNeurons = 2;
  }

  void sampleDTModelArch(int numInputFeatures, DTModelContext& dt) {
    dt.inputFeatures = numInputFeatures;
    dt.treeDepth = 8;
  }

  bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
      return false;
    str.replace(start_pos, from.length(), to);
    return true;
  }


  void registerNNFunction(const std::string& op_name, const std::vector<std::vector<float>>& weightMatrice, int dim1, int dim2) {

    if (op_name.find("mat_mul") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixMultiply::signatures(),
            std::make_unique<MatrixMultiply>(
                nnWeightVector->elements()->values()->asMutable<float>(), dim1, dim2)
            );
        std::cout << "Registered a mat_mul function of name " << op_name << " with dimension " << dim1 << ", " << dim2 << endl;
    }

    else if (op_name.find("mat_add") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixVectorAddition::signatures(),
            std::make_unique<MatrixVectorAddition>(
                nnWeightVector->elements()->values()->asMutable<float>(), dim1)
            );
        std::cout << "Registered a mat_add function of name " << op_name << " with dimension " << dim1 << endl;
    }

    else if (op_name.find("relu") != std::string::npos) {
        exec::registerVectorFunction(
            op_name, Relu::signatures(), std::make_unique<Relu>(),
            {},
            true);
        std::cout << "Registered a relu function of name " << op_name << endl;
    }

    else if (op_name.find("softmax") != std::string::npos) {
        exec::registerVectorFunction(
            op_name, Softmax::signatures(), std::make_unique<Softmax>());
        std::cout << "Registered a softmax function of name " << op_name << endl;
    }

    else if (op_name.find("vector_addition") != std::string::npos) {
        exec::registerVectorFunction(
            op_name,
            VectorAddition::signatures(),
            std::make_unique<VectorAddition>(dim1)
            );
        std::cout << "Registered a vector_addition function of name " << op_name << " with dimension " << dim1 << endl;
    }


}




  std::unordered_map<std::string, PlanBuilder> getBaseTableSources(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder>
        sources; // with filters and projections pushed down;

    auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_name"]})
                    .project({"person_id as an_person_id", "an_features"});
    //  .capturePlanNodeId(akaNameNodeId1)
    tabel2Columns["an"] = {"an_person_id", "an_features"};

    sources["an"] = an_a;

    auto cc_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["complete_cast"]})
            .project({"movie_id", "subject_id", "status_id", "cc_features"});
    //  .capturePlanNodeId(completeCastNodeId1)
    tabel2Columns["cc"] = {"movie_id", "subject_id", "status_id", "cc_features"};

    sources["cc"] = cc_a;

    auto cct1_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind = 'cast'")
            .project({"id as cct1_id", "cct_features as cct1_features"});
    //  .capturePlanNodeId(completeCastNodeId1)
    tabel2Columns["cct1"] = {"cct1_id", "cct1_features"};

    sources["cct1"] = cct1_a;

    auto cct2_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind ='complete+verified'")
            .project({"id as cct2_id", "cct_features as cct2_features"});
    //  .capturePlanNodeId(compCastTypeNodeId2)
    tabel2Columns["cct2"] = {"cct2_id", "cct2_features"};

    sources["cct2"] = cct2_a;

    auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["char_name"]})
                     .filter("name = 'Queen'")
                     .project({"id as chn_id", "chn_features"});
    //  .capturePlanNodeId(charNameNodeId)
    tabel2Columns["chn"] = {"chn_id", "chn_features"};

    sources["chn"] = chn_a;

    auto ci_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cast_info"]})
            .filter(
                "note = '(voice)' OR note = '(voice) (uncredited)' OR note = '(voice: English version)'")
            .project(
                {"movie_id",
                 "person_id",
                 "role_id",
                 "person_role_id",
                 "ci_features"});
    //  .capturePlanNodeId(castInfoNodeId)
    tabel2Columns["ci"] = {"movie_id",
                 "person_id",
                 "role_id",
                 "person_role_id",
                 "ci_features"};

    sources["ci"] = ci_a;

    auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_name"]})
                    .filter("country_code ='[us]'")
                    .project({"id as cn_id", "cn_features"});
    //  .capturePlanNodeId(companyNameNodeId)
    tabel2Columns["cn"] = {"cn_id", "cn_features"};

    sources["cn"] = cn_a;

    auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["info_type"]})
                    .filter("info = 'release dates'")
                    .project({"id as it_id", "it_features"});
    //  .capturePlanNodeId(infoTypeNodeId1)
    tabel2Columns["it"] = {"it_id", "it_features"};

    sources["it"] = it_a;

    auto it3_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["info_type"]})
                     .filter("info = 'trivia'")
                     .project({"id as it3_id", "it_features as it3_features"});
    //  .capturePlanNodeId(infoTypeNodeId2)
    tabel2Columns["it3"] = {"it3_id", "it3_features"};

    sources["it3"] = it3_a;

    auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["keyword"]})
                   .filter("keyword = 'computer-animation'")
                   .project({"id as k_id", "k_features"});
    //  .capturePlanNodeId(keywordNodeId)
    tabel2Columns["k"] = {"k_id", "k_features"};

    sources["k"] = k_a;

    auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_companies"]})
                    .project({"movie_id", "company_id", "mc_features"});
    //  .capturePlanNodeId(movieCompaniesNodeId)
    tabel2Columns["mc"] = {"movie_id", "company_id", "mc_features"};

    sources["mc"] = mc_a;

    auto mi_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info"]})
            .filter(
                "info IS NOT NULL AND (info LIKE 'Japan:%200%' OR info LIKE 'USA:%200%')")
            .project({"movie_id", "info_type_id", "mi_features"});
    //  .capturePlanNodeId(movieInfoNodeId)
    tabel2Columns["mi"] = {"movie_id", "info_type_id", "mi_features"};

    sources["mi"] = mi_a;

    auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_keyword"]})
                    .project({"movie_id", "keyword_id", "mk_features"});
    //  .capturePlanNodeId(movieKeywordNodeId)
    tabel2Columns["mk"] = {"movie_id", "keyword_id", "mk_features"};

    sources["mk"] = mk_a;

    auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["name"]})
                   .filter("gender ='f' AND name LIKE '%An%'")
                   .project({"id as n_id", "n_features"});
    //  .capturePlanNodeId(nameNodeId)
    tabel2Columns["n"] = {"n_id", "n_features"};

    sources["n"] = n_a;

    auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["person_info"]})
                    .project(
                        {"person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
    // .capturePlanNodeId(personInfoNodeId)
    tabel2Columns["pi"] = {"pi_person_id", "pi_info_type_id", "pi_features"};

    sources["pi"] = pi_a;

    auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["role_type"]})
                    .filter("role ='actress'")
                    .project({"id as rt_id", "rt_features"});
    //  .capturePlanNodeId(roleTypeNodeId)
    tabel2Columns["rt"] = {"rt_id", "rt_features"};

    sources["rt"] = rt_a;

    auto t_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["title"]})
            .filter(
                "title = 'Shrek 2' AND  production_year > 2000 AND production_year < 2010")
            .project({"id", "t_features"});
    //    .capturePlanNodeId(titleNodeId)
    tabel2Columns["t"] = {"id", "t_features"};

    sources["t"] = t_a;

    return sources;
  }


  std::unordered_map<std::string, PlanBuilder> getBaseTableSourcesQ29(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder> sources; // with filters and projections pushed down;

      auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["aka_name"]})
    .project({"person_id as an_person_id","an_features"});
tabel2Columns["an"] = {"an_person_id","an_features"};
sources["an"] = an_a;

auto ci_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["cast_info"]})
    //.filter("note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)')")
    .filter("note = '(voice)' OR note = '(voice) (uncredited)' OR note = '(voice: English version)'")
    .project({"movie_id as ci_movie_id","person_id as ci_person_id","person_role_id as ci_person_role_id","role_id as ci_role_id","ci_features"});
tabel2Columns["ci"] = {"ci_movie_id","ci_person_id","ci_person_role_id","ci_role_id","ci_features"};
sources["ci"] = ci_a;

auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["char_name"]})
    .filter("name = 'Queen'")
    .project({"id as chn_id","name as chn_name","chn_features"});
tabel2Columns["chn"] = {"chn_id","chn_name","chn_features"};
sources["chn"] = chn_a;

auto cct2_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["comp_cast_type"]})
    .filter("kind = 'complete+verified'")
    .project({"id as cct2_id","cct_features as cct2_features"});
tabel2Columns["cct2"] = {"cct2_id","cct2_features"};
sources["cct2"] = cct2_a;

auto cct1_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["comp_cast_type"]})
    .filter("kind = 'cast'")
    .project({"id as cct1_id","cct_features as cct1_features"});
tabel2Columns["cct1"] = {"cct1_id","cct1_features"};
sources["cct1"] = cct1_a;

auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["company_name"]})
    .filter("country_code = '[us]'")
    .project({"id as cn_id","cn_features"});
tabel2Columns["cn"] = {"cn_id","cn_features"};
sources["cn"] = cn_a;

auto cc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["complete_cast"]})
    //.filter("country_code = '[us]'")
    .project({"movie_id as cc_movie_id","status_id as cc_status_id","subject_id as cc_subject_id","cc_features"});
tabel2Columns["cc"] = {"cc_movie_id","cc_status_id","cc_subject_id","cc_features"};
sources["cc"] = cc_a;

auto it3_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["info_type"]})
    .filter("info = 'trivia'")
    .project({"id as it3_id","it_features as it3_features"});
tabel2Columns["it3"] = {"it3_id","it3_features"};
sources["it3"] = it3_a;

auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["info_type"]})
    .filter("info = 'release dates'")
    .project({"id as it_id","it_features"});
tabel2Columns["it"] = {"it_id","it_features"};
sources["it"] = it_a;

auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["keyword"]})
    .filter("keyword = 'computer-animation'")
    .project({"id as k_id","k_features"});
tabel2Columns["k"] = {"k_id","k_features"};
sources["k"] = k_a;

auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_companies"]})
    .project({"company_id as mc_company_id","movie_id as mc_movie_id","mc_features"});
tabel2Columns["mc"] = {"mc_company_id","mc_movie_id","mc_features"};
sources["mc"] = mc_a;

auto mi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_info"]})
    .filter("info IS NOT NULL AND (info LIKE 'Japan:%200%' OR info LIKE 'USA:%200%')")
    .project({"info_type_id as mi_info_type_id","movie_id as mi_movie_id","mi_features"});
tabel2Columns["mi"] = {"mi_info_type_id","mi_movie_id","mi_features"};
sources["mi"] = mi_a;

auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_keyword"]})
    .project({"keyword_id as mk_keyword_id","movie_id as mk_movie_id","mk_features"});
tabel2Columns["mk"] = {"mk_keyword_id","mk_movie_id","mk_features"};
sources["mk"] = mk_a;

auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["name"]})
    .filter("name LIKE '%An%' AND gender = 'f'")
    .project({"id as n_id","name as n_name","n_features"});
tabel2Columns["n"] = {"n_id","n_name","n_features"};
sources["n"] = n_a;

auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["person_info"]})
    .project({"info_type_id as pi_info_type_id","person_id as pi_person_id","pi_features"});
tabel2Columns["pi"] = {"pi_info_type_id","pi_person_id","pi_features"};
sources["pi"] = pi_a;

auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["role_type"]})
    .filter("role = 'actress'")
    .project({"id as rt_id","rt_features"});
tabel2Columns["rt"] = {"rt_id","rt_features"};
sources["rt"] = rt_a;

auto t_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["title"]})
    //.filter("production_year BETWEEN 2000 AND 2010")
    .filter("title = 'Shrek 2' AND  production_year > 2000 AND production_year < 2010")
    .project({"id as t_id","title as t_title","t_features"});
tabel2Columns["t"] = {"t_id","t_title","t_features"};
sources["t"] = t_a;

      return sources;
  }



  std::unordered_map<std::string, PlanBuilder> getBaseTableSourcesQ9a(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder> sources; // with filters and projections pushed down;

      auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["aka_name"]})
    .project({"person_id as an_person_id", "name as an_name", "an_features"});
tabel2Columns["an"] = {"an_person_id", "an_name", "an_features"};
sources["an"] = an_a;

auto ci_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["cast_info"]})
    .filter("note = '(voice)' OR note = '(voice) (uncredited)' OR note = '(voice: English version)'")
    .project({"movie_id as ci_movie_id","person_id as ci_person_id","person_role_id as ci_person_role_id","role_id as ci_role_id","ci_features"});
tabel2Columns["ci"] = {"ci_movie_id","ci_person_id","ci_person_role_id","ci_role_id","ci_features"};
sources["ci"] = ci_a;

auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["char_name"]})
    //.filter("name = 'Queen'")
    .project({"id as chn_id","name as chn_name","chn_features"});
tabel2Columns["chn"] = {"chn_id","chn_name","chn_features"};
sources["chn"] = chn_a;

/*auto cct2_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["comp_cast_type"]})
    .filter("kind = 'complete+verified'")
    .project({"id as cct2_id","cct_features as cct2_features"});
tabel2Columns["cct2"] = {"cct2_id","cct2_features"};
sources["cct2"] = cct2_a;

auto cct1_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["comp_cast_type"]})
    .filter("kind = 'cast'")
    .project({"id as cct1_id","cct_features as cct1_features"});
tabel2Columns["cct1"] = {"cct1_id","cct1_features"};
sources["cct1"] = cct1_a;*/

auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["company_name"]})
    .filter("country_code = '[us]'")
    .project({"id as cn_id","cn_features"});
tabel2Columns["cn"] = {"cn_id","cn_features"};
sources["cn"] = cn_a;

/*auto cc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["complete_cast"]})
    .project({"movie_id as cc_movie_id","status_id as cc_status_id","subject_id as cc_subject_id","cc_features"});
tabel2Columns["cc"] = {"cc_movie_id","cc_status_id","cc_subject_id","cc_features"};
sources["cc"] = cc_a;

auto it3_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["info_type"]})
    .filter("info = 'trivia'")
    .project({"id as it3_id","it_features as it3_features"});
tabel2Columns["it3"] = {"it3_id","it3_features"};
sources["it3"] = it3_a;

auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["info_type"]})
    .filter("info = 'release dates'")
    .project({"id as it_id","it_features"});
tabel2Columns["it"] = {"it_id","it_features"};
sources["it"] = it_a;

auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["keyword"]})
    .filter("keyword = 'computer-animation'")
    .project({"id as k_id","k_features"});
tabel2Columns["k"] = {"k_id","k_features"};
sources["k"] = k_a;*/

auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_companies"]})
    .filter("note LIKE '%(USA)%' OR note LIKE '%(worldwide)%'")
    .project({"company_id as mc_company_id","movie_id as mc_movie_id","mc_features"});
tabel2Columns["mc"] = {"mc_company_id","mc_movie_id","mc_features"};
sources["mc"] = mc_a;

/*auto mi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_info"]})
    .filter("info IS NOT NULL AND (info LIKE 'Japan:%200%' OR info LIKE 'USA:%200%')")
    .project({"info_type_id as mi_info_type_id","movie_id as mi_movie_id","mi_features"});
tabel2Columns["mi"] = {"mi_info_type_id","mi_movie_id","mi_features"};
sources["mi"] = mi_a;

auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["movie_keyword"]})
    .project({"keyword_id as mk_keyword_id","movie_id as mk_movie_id","mk_features"});
tabel2Columns["mk"] = {"mk_keyword_id","mk_movie_id","mk_features"};
sources["mk"] = mk_a;*/

auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["name"]})
    .filter("name LIKE '%An%' AND gender = 'f'")
    .project({"id as n_id","n_features"});
tabel2Columns["n"] = {"n_id","n_features"};
sources["n"] = n_a;

/*auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["person_info"]})
    .project({"info_type_id as pi_info_type_id","person_id as pi_person_id","pi_features"});
tabel2Columns["pi"] = {"pi_info_type_id","pi_person_id","pi_features"};
sources["pi"] = pi_a;*/

auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["role_type"]})
    .filter("role = 'actress'")
    .project({"id as rt_id","rt_features"});
tabel2Columns["rt"] = {"rt_id","rt_features"};
sources["rt"] = rt_a;

auto t_a = PlanBuilder(planNodeIdGenerator, pool_.get())
    .values({tableName2RowVector["title"]})
    //.filter("production_year BETWEEN 2000 AND 2010")
    .filter("production_year > 2000 AND production_year < 2010")
    .project({"id as t_id","title as t_title","t_features"});
tabel2Columns["t"] = {"t_id","t_title","t_features"};
sources["t"] = t_a;

      return sources;
  }



    std::unordered_map<std::string, PlanBuilder> getAllTableSources(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder>
        sources; // with filters and projections pushed down;

    auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_name"]})
                    .project({"person_id as an_person_id", "name as an_name", "imdb_index as an_imdb_index", "an_features"});
    tabel2Columns["an"] = {"an_person_id", "an_name", "an_imdb_index", "an_features"};
    sources["an"] = an_a;

    auto at_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_title"]})
                    .project({"id as at_id", "movie_id as at_movie_id", "title as at_title", "imdb_index as at_imdb_index", "kind_id as at_kind_id", "at_features"});
    tabel2Columns["at"] = {"at_id", "at_movie_id", "at_title", "at_imdb_index", "at_kind_id", "at_features"};
    sources["at"] = at_a;

    auto ci_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cast_info"]})
            .project(
                {"movie_id as ci_movie_id",
                 "person_id as ci_person_id",
                 "role_id as ci_role_id",
                 "person_role_id as ci_person_role_id",
                 "ci_features"})
            .limit(0, 3624434, false);
    tabel2Columns["ci"] = {"ci_movie_id",
                 "ci_person_id",
                 "ci_role_id",
                 "ci_person_role_id",
                 "ci_features"};
    sources["ci"] = ci_a;

    auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["char_name"]})
                     .project({"id as chn_id", "name as chn_name", "imdb_index as chn_imdb_index", "imdb_id as chn_imdb_id", "chn_features"});
    tabel2Columns["chn"] = {"chn_id", "chn_name", "chn_imdb_index", "chn_imdb_id", "chn_features"};
    sources["chn"] = chn_a;

    auto cc_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["complete_cast"]})
            .project({"id as cc_id", "movie_id as cc_movie_id", "subject_id as cc_subject_id", "status_id as cc_status_id", "cc_features"});
    tabel2Columns["cc"] = {"cc_id", "cc_movie_id", "cc_subject_id", "cc_status_id", "cc_features"};
    sources["cc"] = cc_a;

    auto cct_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .project({"id as cct_id", "kind as cct_kind", "cct_features"});
    tabel2Columns["cct"] = {"cct_id", "cct_kind", "cct_features"};
    sources["cct"] = cct_a;


    auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_name"]})
                    .project({"id as cn_id", "name as cn_name", "country_code as cn_country_code", "imdb_id as cn_imdb_id", "cn_features"});
    tabel2Columns["cn"] = {"cn_id", "cn_name", "cn_country_code", "cn_imdb_id", "cn_features"};
    sources["cn"] = cn_a;

    auto ct_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_type"]})
                    .project({"id as ct_id", "kind as ct_kind", "ct_features"});
    tabel2Columns["ct"] = {"ct_id", "ct_kind", "ct_features"};
    sources["ct"] = ct_a;

    auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["info_type"]})
                    .project({"id as it_id", "info as it_info", "it_features"});
    tabel2Columns["it"] = {"it_id", "it_info", "it_features"};
    sources["it"] = it_a;

    auto kt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["kind_type"]})
                    .project({"id as kt_id", "kind as kt_kind", "kt_features"});
    tabel2Columns["kt"] = {"kt_id", "kt_kind", "kt_features"};
    sources["kt"] = kt_a;

    auto lt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["link_type"]})
                    .project({"id as lt_id", "link as lt_link", "lt_features"});
    tabel2Columns["lt"] = {"lt_id", "lt_link", "lt_features"};
    sources["lt"] = lt_a;

    auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["keyword"]})
                   .project({"id as k_id", "keyword as k_keyword", "k_features"});
    tabel2Columns["k"] = {"k_id", "k_keyword", "k_features"};
    sources["k"] = k_a;

    auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_companies"]})
                    .project({"id as mc_id", "movie_id as mc_movie_id", "company_id as mc_company_id", "company_type_id as mc_company_type_id", "mc_features"});
    tabel2Columns["mc"] = {"mc_id", "mc_movie_id", "mc_company_id", "mc_company_type_id", "mc_features"};
    sources["mc"] = mc_a;

    auto mi_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info"]})
            .project({"id as mi_id", "movie_id as mi_movie_id", "info_type_id as mi_info_type_id", "info as mi_info", "mi_features"});
    tabel2Columns["mi"] = {"mi_id", "mi_movie_id", "mi_info_type_id", "mi_info", "mi_features"};
    sources["mi"] = mi_a;

    auto mii_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info_idx"]})
            .project({"id as mii_id", "movie_id as mii_movie_id", "info_type_id as mii_info_type_id", "info as mii_info", "mii_features"});
    tabel2Columns["mii"] = {"mii_id", "mii_movie_id", "mii_info_type_id", "mii_info", "mii_features"};
    sources["mii"] = mii_a;

    auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_keyword"]})
                    .project({"id as mk_id", "movie_id as mk_movie_id", "keyword_id as mk_keyword_id", "mk_features"});
    tabel2Columns["mk"] = {"mk_id", "mk_movie_id", "mk_keyword_id", "mk_features"};
    sources["mk"] = mk_a;

    auto ml_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_link"]})
                    .project({"id as ml_id", "movie_id as ml_movie_id", "linked_movie_id as ml_linked_movie_id", "link_type_id as ml_link_type_id", "ml_features"});
    tabel2Columns["ml"] = {"ml_id", "ml_movie_id", "ml_linked_movie_id", "ml_link_type_id", "ml_features"};
    sources["ml"] = ml_a;

    auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["name"]})
                   .project({"id as n_id", "name as n_name", "imdb_index as n_imdb_index", "imdb_id as n_imdb_id", "n_features"});
    tabel2Columns["n"] = {"n_id", "n_name", "n_imdb_index", "n_imdb_id", "n_features"};
    sources["n"] = n_a;

    auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["person_info"]})
                    .project(
                        {"id as pi_id",
                         "person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
    tabel2Columns["pi"] = {"pi_id", "pi_person_id", "pi_info_type_id", "pi_features"};
    sources["pi"] = pi_a;

    auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["role_type"]})
                    .project({"id as rt_id", "role as rt_role", "rt_features"});
    tabel2Columns["rt"] = {"rt_id", "rt_role", "rt_features"};
    sources["rt"] = rt_a;

    auto t_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["title"]})
            .project({"id as t_id", "title as t_title", "imdb_index as t_imdb_index", "kind_id as t_kind_id", "imdb_id as t_imdb_id", "t_features"});
    tabel2Columns["t"] = {"t_id", "t_title", "t_imdb_index", "t_kind_id", "t_imdb_id", "t_features"};
    sources["t"] = t_a;

    return sources;
  }



  void updateFeatureStatus(const std::string& joinOrderStr) {
      //std::unordered_map<std::string, RowVectorPtr> tableName2RowVector;
      // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
        std::cerr << "Error parsing JSON: " << errors << "\n";
        //return 1;
    }
    //JSON Reading successful

    // Iterate through the array
    for (const auto& item : root) {
        std::string joinId = item["ID"].asString();
        std::string leftTable = item["Left"].asString();
        std::string rightTable = item["Right"].asString();
        std::string probKeys = item["ProbeKeys"].asString();
        std::string buildKeys = item["BuildKeys"].asString();
        int NumDimLeft = item["NumDimLeft"].asInt();
        int NumDimRight = item["NumDimRight"].asInt();

        bool isLeftTableNotLeaf = isInteger(leftTable);
        if (isLeftTableNotLeaf == false) {
            std::string featureCol = leftTable + "_features";
            FeatureStatus fs;
            fs.isFeature = 1;
            fs.isPushed = 0;
            fs.vectorSize = NumDimLeft;
            allFeatureStatus[featureCol] = fs;
        }

        bool isRightTableNotLeaf = isInteger(rightTable);
        if (isRightTableNotLeaf == false) {
            std::string featureCol = rightTable + "_features";
            FeatureStatus fs;
            fs.isFeature = 1;
            fs.isPushed = 0;
            fs.vectorSize = NumDimRight;
            allFeatureStatus[featureCol] = fs;
        }

    }

  }




  float* genWeight(int dim1, int dim2) {
    int total_size = dim1 * dim2;

    // generate weight matrix
    float* weight = new float[total_size];
    for (int i = 0; i < total_size; i++) {
      if (i % 2 == 0) {
        weight[i] = 1.0;
      } else {
        weight[i] = 0.0;
      }
    }
    return weight;
  }


bool addModelInferenceToQueryPlanAfterFactorize(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, const std::string& colFeature) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        //int avoid = 0;
        for (int i = modelOperators.size() - 1; i > 0; i--) {
        //for (int i = 1; i > 0; i--) {
            std::string opName = modelOperators[i];
            if (opName.find("mat_mul") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                int dim2 = param[0].size();
                registerNNFunction(opName, param, dim1, dim2);
            }
            else if (opName.find("mat_add") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                registerNNFunction(opName, param, dim1, -1);
            }
            /*else if (opName.find("relu") != std::string::npos) {
                avoid += 1;
                continue;
            }*/
            else {
                registerNNFunction(opName, emptyMatrix, -1, -1);
            }
            modelProjString += opName + "(";
        }
        modelProjString += colFeature;

        for (int i = modelOperators.size() - 1; i > 0; i--) {
        //for (int i = 1; i > 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;

}

bool addModelInferenceToQueryPlan(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        for (int i = modelOperators.size() - 1; i >= 0; i--) {
            std::string opName = modelOperators[i];
            if (opName.find("mat_mul") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                int dim2 = param[0].size();
                registerNNFunction(opName, param, dim1, dim2);
            }
            else if (opName.find("mat_add") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                registerNNFunction(opName, param, dim1, -1);
            }
            else {
                registerNNFunction(opName, emptyMatrix, -1, -1);
            }
            modelProjString += opName + "(";
        }
        modelProjString += "features";

        for (int i = modelOperators.size() - 1; i >= 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;

}


  std::string rewriteWithFactorization(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string joinOrderStr) {

    // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
        std::cerr << "Error parsing JSON: " << errors << "\n";
        return "1";
    }
    //JSON Reading successful"

    std::unordered_map<std::string, PlanBuilder> sources = getAllTableSources(planNodeIdGenerator); //with filters and projections pushed down;
    std::unordered_map<std::string, std::string> colToMapCol;

    std::string outName;
    std::string leftName;
    std::string rightName;

    std::string firstOpName = modelOperators[0]; // name of operator of split layer
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName]; // weight of the split layer
    int fCurrentTotal = 0;
    int numCols = firstWeight.size(); // number of columns in split layer
    int numNeurons = firstWeight[0].size(); // number of neurons in split layer
    int k = 0;
    int addIdx = 0;
    int appendedCount = 0;

    std::vector<std::vector<float>> emptyMatrix;
    bool isAdditionRegistered = false;

    // Iterate through the array
    for (const auto& item : root) {
        std::cout << "ID: " << item["ID"].asString() << "\n";
        std::cout << "Left: " << item["Left"].asString() << "\n";
        std::cout << "Right: " << item["Right"].asString() << "\n";
        std::cout << "Pred: " << item["Pred"].asString() << "\n";
        std::cout << "ProbeKeys: " << item["ProbeKeys"].asString() << "\n";
        std::cout << "BuildKeys: " << item["BuildKeys"].asString() << "\n";

        std::string joinId = item["ID"].asString();
        std::string leftTable = item["Left"].asString();
        std::string rightTable = item["Right"].asString();
        std::string probKeys = item["ProbeKeys"].asString();
        std::string buildKeys = item["BuildKeys"].asString();
        int NumDimLeft = item["NumDimLeft"].asInt();
        int NumDimRight = item["NumDimRight"].asInt();

        std::string leftFactorizationKey = leftTable + "--->" + joinId;
        std::string rightFactorizationKey = rightTable + "--->" + joinId;

        std::cout << "left factorization key: " << leftFactorizationKey << std::endl;
        std::cout << "right factorization key: " << rightFactorizationKey << std::endl;

        bool isLeftTableNotLeaf = isInteger(leftTable);

        if (isLeftTableNotLeaf == false) {
            // left table is leaf

            featureStartPos[leftTable] = fCurrentTotal;
            fCurrentTotal += NumDimLeft;

            std::vector<std::string> curCols = tabel2Columns[leftTable];
            std::vector<std::string> newProj;
            int projSize = curCols.size();
            for (int i = 0; i < projSize - 1; i++) {
                newProj.push_back(curCols[i]);
            }

            std::string fName = curCols[projSize - 1];
            std::cout << fName << std::endl;

            if (factorizationPlans[leftFactorizationKey] == 1) {
               // Doing factorization of left edge
                std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, featureStartPos[leftTable], NumDimLeft);
                std::string newOpName = firstOpName + "_" + std::to_string(k);
                registerNNFunction(newOpName, subWeight, NumDimLeft, numNeurons);
                std::string fNewName = "factorized_" + std::to_string(k);
                colToMapCol[fName] = fNewName;
                std::string fNewNameFull = newOpName + "(get_feature_vec(" + fName + ", " + std::to_string(NumDimLeft) + ")) AS " + fNewName;
                tabel2Columns[leftTable][projSize - 1] = fNewName;
                k += 1;

                FeatureStatus fs;
                fs.isFeature = 1;
                fs.isPushed = 1;
                fs.vectorSize = numNeurons;
                allFeatureStatus[fNewName] = fs;

                newProj.push_back(fNewNameFull);
                sources[leftTable] = sources[leftTable].project({newProj});
            }
            else {
                // Not Doing factorization of left edge
                std::string fNewName = "mapped_" + fName;
                colToMapCol[fName] = fNewName;
                std::string fNewNameFull = "get_feature_vec(" + fName + ", " + std::to_string(NumDimLeft) + ") AS " + fNewName;
                tabel2Columns[leftTable][projSize - 1] = fNewName;

                FeatureStatus fs = allFeatureStatus[fName];
                fs.featureStartPos = featureStartPos[leftTable];
                allFeatureStatus[fNewName] = fs;

                newProj.push_back(fNewNameFull);
                sources[leftTable] = sources[leftTable].project({newProj});
            }

        }

        bool isRightTableNotLeaf = isInteger(rightTable);
        if (isRightTableNotLeaf == false) {
            // right table is leaf

            featureStartPos[rightTable] = fCurrentTotal;
            fCurrentTotal += NumDimRight;

            std::vector<std::string> curCols = tabel2Columns[rightTable];
            std::vector<std::string> newProj;
            int projSize = curCols.size();
            for (int i = 0; i < projSize - 1; i++) {
                newProj.push_back(curCols[i]);
            }

            std::string fName = curCols[projSize - 1];
            std::cout << fName << std::endl;

            if (factorizationPlans[rightFactorizationKey] == 1) {
                // Doing factorization of right edge
                std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, featureStartPos[rightTable], NumDimRight);
                std::string newOpName = firstOpName + "_" + std::to_string(k);
                registerNNFunction(newOpName, subWeight, NumDimRight, numNeurons);
                std::string fNewName = "factorized_" + std::to_string(k);
                colToMapCol[fName] = fNewName;
                std::string fNewNameFull = "get_feature_vec(" + fName + ", " + std::to_string(NumDimRight) + ") AS " + fNewName;
                tabel2Columns[rightTable][projSize - 1] = fNewName;
                k += 1;

                FeatureStatus fs;
                fs.isFeature = 1;
                fs.isPushed = 1;
                fs.vectorSize = numNeurons;
                allFeatureStatus[fNewName] = fs;

                newProj.push_back(fNewNameFull);
                sources[rightTable] = sources[rightTable].project({newProj});
            }
            else {
                // Not Doing factorization of right edge
                // Not Doing factorization of left edge
                std::string fNewName = "mapped_" + fName;
                colToMapCol[fName] = fNewName;
                std::string fNewNameFull = "get_feature_vec(" + fName + ", " + std::to_string(NumDimRight) + ") AS " + fNewName;
                tabel2Columns[rightTable][projSize - 1] = fNewName;

                FeatureStatus fs = allFeatureStatus[fName];
                fs.featureStartPos = featureStartPos[rightTable];
                allFeatureStatus[fNewName] = fs;

                newProj.push_back(fNewNameFull);
                sources[rightTable] = sources[rightTable].project({newProj});
            }

        }

    featureStartPos[joinId] = featureStartPos[leftTable];


    PlanBuilder left, right, out;

    if (sources.count(leftTable) > 0) {

        left = sources[leftTable];

    }

    if (sources.count(rightTable) > 0) {

        right = sources[rightTable];

    }


    std::unordered_map<std::string, int> actualJoinProj;
    // Access Projection if it exists
    if (item.isMember("Projection")) {
        for (const auto& proj : item["Projection"]) {
             int is_mapped = colToMapCol.count(proj.asString());
             if (is_mapped >= 1) {
                 //std::cout << colToMapCol[proj.asString()] << std::endl;
                 //projections.push_back(colToMapCol[proj.asString()]);
                 actualJoinProj[colToMapCol[proj.asString()]] = 1;
             }
             else {
                 //std::cout << proj.asString() << std::endl;
                 //projections.push_back(proj.asString());
                 actualJoinProj[proj.asString()] = 1;
             }

	    }
    }

    //compose join features
    std::vector<std::string> leftProj;
    std::vector<std::string> leftCols;
    std::vector<std::string> joinCols;
    int totalPushed = 0;
    std::string addedVecPrefix = "added_vec_";
    std::string appendedFeaturePrefix = "appended_feature_";

    if (factorizationPlans[leftFactorizationKey] == 1) {
        // Factorizing all left features that were not pushed earlier
        int newFactorizedCount = 0;
        for (int i = 0; i < tabel2Columns[leftTable].size(); i++) {
                std::string fName = tabel2Columns[leftTable][i];
                std::cout << fName << std::endl;
                if (actualJoinProj.count(fName) <= 0 && fName.substr(0, 10) != addedVecPrefix && fName.substr(0, 17) != appendedFeaturePrefix) {
                    if (fName == probKeys) {
                        leftProj.push_back(fName);
                        leftCols.push_back(fName);
                    }
                    continue;
                }
                if (allFeatureStatus.count(fName) <= 0) {
                // Feature not a feature
                   //std::cout << "Feature status not found for feature " << fName << std::endl;
                   leftProj.push_back(fName);
                   leftCols.push_back(fName);
                   joinCols.push_back(fName);
                   continue;
                }
                FeatureStatus fs = allFeatureStatus[fName];
                if (fs.isPushed == 0) {

                   // Feature not pushed earlier, pushing now
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    colToMapCol[fName] = fNewName;
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    /*int factorizedIndex = getStringIndex(tabel2Columns[leftTable], fName);
                    if (factorizedIndex >= 0) {
                        tabel2Columns[leftTable][factorizedIndex] = fNewName;
                    }*/

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    leftProj.push_back(fNewNameFull);
                    leftCols.push_back(fNewName);
                    joinCols.push_back(fNewName);
                    newFactorizedCount += 1;
                    totalPushed += 1;
                }
                else {
                    // Feature pushed earlier
                    leftProj.push_back(fName);
                    leftCols.push_back(fName);
                    joinCols.push_back(fName);
                    totalPushed += 1;
                }
            }

            if (newFactorizedCount > 0) {
                left = left.project({leftProj});
                sources[leftTable] = left;
                tabel2Columns[leftTable] = leftCols;
            }
    }
    else {
        // Just adding all left columns without factorization
        //bool projChanged = false;
        for (int i = 0; i < tabel2Columns[leftTable].size(); i++) {
            std::string fName = tabel2Columns[leftTable][i];
            if (actualJoinProj.count(fName) <= 0 && fName.substr(0, 10) != addedVecPrefix && fName.substr(0, 17) != appendedFeaturePrefix) {
                //projChanged = true;
                if (fName == probKeys) {
                    leftProj.push_back(fName);
                    leftCols.push_back(fName);
                }
                continue;
            }
            leftProj.push_back(fName);
            leftCols.push_back(fName);
            joinCols.push_back(fName);
        }
        /*if (projChanged == true) {
            left = left.project({leftProj});
        }*/
    }


    std::vector<std::string> rightProj;
    std::vector<std::string> rightCols;

    if (factorizationPlans[rightFactorizationKey] == 1) {
        // Factorizing all right features that were not pushed earlier
        int newFactorizedCount = 0;
        for (int i = 0; i < tabel2Columns[rightTable].size(); i++) {
                std::string fName = tabel2Columns[rightTable][i];
                std::cout << fName << std::endl;
                if (actualJoinProj.count(fName) <= 0 && fName.substr(0, 10) != addedVecPrefix && fName.substr(0, 17) != appendedFeaturePrefix) {
                    if (fName == buildKeys) {
                        rightProj.push_back(fName);
                        rightCols.push_back(fName);
                    }
                    continue;
                }
                if (allFeatureStatus.count(fName) <= 0) {
                    // Feature not a feature
                    //std::cout << "Feature status not found for feature " << fName << std::endl;
                    rightProj.push_back(fName);
                    rightCols.push_back(fName);
                    joinCols.push_back(fName);
                   continue;
                }
                FeatureStatus fs = allFeatureStatus[fName];
                if (fs.isPushed == 0) {

                   // Feature not pushed earlier, pushing now
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    colToMapCol[fName] = fNewName;
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    /*int factorizedIndex = getStringIndex(tabel2Columns[rightTable], fName);
                    if (factorizedIndex >= 0) {
                        tabel2Columns[rightTable][factorizedIndex] = fNewName;
                    }*/

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    rightProj.push_back(fNewNameFull);
                    rightCols.push_back(fNewName);
                    joinCols.push_back(fNewName);
                    newFactorizedCount += 1;
                    totalPushed += 1;
                }
                else {
                    // Feature pushed earlier
                    rightProj.push_back(fName);
                    rightCols.push_back(fName);
                    joinCols.push_back(fName);
                    totalPushed += 1;
                }
            }

            if (newFactorizedCount > 0) {
                right = right.project({rightProj});
                sources[rightTable] = right;
                tabel2Columns[rightTable] = rightCols;
            }
    }
    else {
        // Just adding all right columns without factorization
        //bool projChanged = false;
        for (int i = 0; i < tabel2Columns[rightTable].size(); i++) {
            std::string fName = tabel2Columns[rightTable][i];
            if (actualJoinProj.count(fName) <= 0 && fName.substr(0, 10) != addedVecPrefix && fName.substr(0, 17) != appendedFeaturePrefix) {
                //projChanged = true;
                if (fName == buildKeys) {
                    rightProj.push_back(fName);
                    rightCols.push_back(fName);
                }
                continue;
            }
            rightProj.push_back(fName);
            rightCols.push_back(fName);
            joinCols.push_back(fName);
        }
        /*if (projChanged == true) {
            right = right.project({rightProj});
        }*/
    }

    std:cout << "Join columns for Join ID: " << joinId << std::endl;
    std::cout << joinCols << std::endl;

    tabel2Columns[joinId] = joinCols;
    // Writing join
    out = left.hashJoin(
        {item["ProbeKeys"].asString()},
        {item["BuildKeys"].asString()},
        right.planNode(),
        "",
            {joinCols}
    );

    std::vector<std::string> joinColsNew;
    std::vector<std::string> joinProjNew;

        bool concatPerformed = false;
        int startPos = -1;
        int vecSize = -1;
        bool isAppended = false;
        std::string lastCol = "";
        std::string projString = "";
        for (int i = 0; i < joinCols.size(); i++) {
            std::string fName = joinCols[i];
            int is_feature = allFeatureStatus.count(fName);
            //FeatureStatus fs = allFeatureStatus[fName];
            if (is_feature <= 0 ) {
                joinColsNew.push_back(fName);
                joinProjNew.push_back(fName);
                continue;
            }
            if (allFeatureStatus[fName].isPushed == 1) {
                if (totalPushed <= 1) {
                    joinColsNew.push_back(fName);
                    joinProjNew.push_back(fName);
                }
                continue;
            }

            //colToMapCol[fName] = fNewName;

            FeatureStatus fs = allFeatureStatus[fName];
            if (fs.featureStartPos == startPos + vecSize) {
                projString += "," + fName;
                vecSize += fs.vectorSize;
                isAppended = true;
            }
            else {
                if (isAppended == false) {
                    if (lastCol != "") {
                        joinColsNew.push_back(lastCol);
                        joinProjNew.push_back(lastCol);
                    }
                }
                else {
                    std::string fNewName = "appended_feature_" + std::to_string(appendedCount);
                    std::string fNewNameFull = "concat(" + projString + ") AS " + fNewName;
                    FeatureStatus fsNew;
                    fsNew.isFeature = true;
                    fsNew.isPushed = 0;
                    fsNew.featureStartPos = startPos;
                    fsNew.vectorSize = vecSize;
                    allFeatureStatus[fNewName] = fsNew;
                    joinColsNew.push_back(fNewName);
                    joinProjNew.push_back(fNewNameFull);
                    appendedCount += 1;
                    isAppended = false;
                    concatPerformed = true;
                }

                projString = fName;
                lastCol = fName;
                startPos = fs.featureStartPos;
                vecSize = fs.vectorSize;
            }
        }

        if (isAppended) {
            std::string fNewName = "appended_feature_" + std::to_string(appendedCount);
            std::string fNewNameFull = "concat(" + projString + ") AS " + fNewName;
            FeatureStatus fsNew;
            fsNew.isFeature = true;
            fsNew.isPushed = 0;
            fsNew.featureStartPos = startPos;
            fsNew.vectorSize = vecSize;
            allFeatureStatus[fNewName] = fsNew;
            joinColsNew.push_back(fNewName);
            joinProjNew.push_back(fNewNameFull);
            appendedCount += 1;
            concatPerformed = true;
        }
        else if (lastCol != "") {
            joinColsNew.push_back(lastCol);
            joinProjNew.push_back(lastCol);
        }


    if (totalPushed > 1) {
       // some features were pushed before join, they need to be aggregated
        //std::vector<std::string> joinColsNew;
        //std::vector<std::string> joinProj;

        bool isFirstVec = true;
        std::string projString = "";
        for (int i = 0; i < joinCols.size(); i++) {
            std::string fName = joinCols[i];
            int is_feature = allFeatureStatus.count(fName);
            //FeatureStatus fs = allFeatureStatus[fName];
            if (is_feature <= 0 || allFeatureStatus[fName].isPushed == 0) {
                //joinColsNew.push_back(fName);
                //joinProj.push_back(fName);
                continue;
            }

            //colToMapCol[fName] = fNewName;

            if (isFirstVec == true) {
                projString = fName;
                isFirstVec = false;
                continue;
            }

            std::string newOpName = "vector_addition";
            std::string opName = newOpName + "(" + projString + ", " + fName + ")";
            projString = opName;
            if (isAdditionRegistered == false) {
                registerNNFunction(newOpName, emptyMatrix, numNeurons, -1);
                isAdditionRegistered = true;
            }
        }

        std::string fNewName = "added_vec_" + std::to_string(addIdx);
        addIdx += 1;

        projString += " AS " + fNewName;
        joinColsNew.push_back(fNewName);
        joinProjNew.push_back(projString);

        FeatureStatus fs;
        fs.isFeature = true;
        fs.isPushed = 1;
        fs.vectorSize = numNeurons;
        allFeatureStatus[fNewName] = fs;

        out = out.project({joinProjNew});
        tabel2Columns[joinId] = joinColsNew;
    }
    else if (concatPerformed) {
        out = out.project({joinProjNew});
        tabel2Columns[joinId] = joinColsNew;
    }
    /*if (concatPerformed) {
        out = out.project({joinProjNew});
        tabel2Columns[joinId] = joinColsNew;
    }*/
    // Finished Writing join

    outName = joinId;
    sources[outName] = out;

    }

    planBuilder = sources[outName];

    std::cout << "After the last join" << std::endl;
    std::vector<std::string> joinProj;
    std::vector<std::string> joinCols;
    int newPushedCount = 0;

    // iterate over the final join projection output to compute not pushed feature
    for (int i = 0; i < tabel2Columns[outName].size(); i++) {
            std::string fName = tabel2Columns[outName][i];
            int is_feature = allFeatureStatus.count(fName);
            if (is_feature >= 1) {
                FeatureStatus fs = allFeatureStatus[fName];
                if (fs.isPushed == 0) {

                    // found a feature in the final join output which were not pushed
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    std::cout << fName << ", " << fs.featureStartPos << ", " << fs.vectorSize << ", " << subWeight.size() << ", " << subWeight[0].size() << std::endl;
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    joinProj.push_back(fNewNameFull);
                    joinCols.push_back(fNewName);
                    newPushedCount += 1;
                }
                else {
                    joinProj.push_back(fName);
                    joinCols.push_back(fName);
                }
            }
        }

    if (newPushedCount > 0) {
       // final join output projection changed
        planBuilder = planBuilder.project({joinProj});
    }

    std::string colOutput = "features";

    // perform aggregation of the final join output
        std::string projString = joinCols[0];
        for (int i = 1; i < joinCols.size(); i++) {
            std::string newOpName = "vector_addition";
            std::string opName = newOpName + "(" + projString + ", " + joinCols[i] + ")";
            projString = opName;
            if (isAdditionRegistered == false) {
                registerNNFunction(newOpName, emptyMatrix, numNeurons, -1);
                isAdditionRegistered = true;
            }
        }

        if (projString != joinCols[0]) {
            projString += " AS " + colOutput;
            planBuilder = planBuilder.project({projString});
        }
        else {
            colOutput = projString;
        }

    std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
    return colOutput;

}



bool writeWithoutFactorization(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string joinOrderStr) {

    // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
        std::cerr << "Error parsing JSON: " << errors << "\n";
        return 1;
    }

    std::unordered_map<std::string, PlanBuilder> sources = getAllTableSources(planNodeIdGenerator);
    std::unordered_map<std::string, std::string> colToMapCol;

    std::string outName;
    std::string leftName;
    std::string rightName;
    std::vector<std::string> projections;

    std::string firstOpName = modelOperators[0];
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName];
    int fCurrentTotal = 0;
    int numCols = firstWeight.size();
    int numNeurons = firstWeight[0].size();
    int k = 0;

    // Iterate through the array
    for (const auto& item : root) {
        std::cout << "ID: " << item["ID"].asString() << "\n";
        std::cout << "Left: " << item["Left"].asString() << "\n";
        std::cout << "Right: " << item["Right"].asString() << "\n";
        std::cout << "Pred: " << item["Pred"].asString() << "\n";
        std::cout << "ProbeKeys: " << item["ProbeKeys"].asString() << "\n";
        std::cout << "BuildKeys: " << item["BuildKeys"].asString() << "\n";

        std::string joinId = item["ID"].asString();
        std::string leftTable = item["Left"].asString();
        std::string rightTable = item["Right"].asString();
        std::string probKeys = item["ProbeKeys"].asString();
        std::string buildKeys = item["BuildKeys"].asString();
        int NumDimLeft = item["NumDimLeft"].asInt();
        int NumDimRight = item["NumDimRight"].asInt();


        bool isLeftTableNotLeaf = isInteger(leftTable);
        if (isLeftTableNotLeaf == false) {
            // left table is leaf
            fCurrentTotal += NumDimLeft;

            std::vector<std::string> curCols = tabel2Columns[leftTable];
            std::vector<std::string> newProj;
            int projSize = curCols.size();
            for (int i = 0; i < projSize - 1; i++) {
                newProj.push_back(curCols[i]);
            }

            std::string fName = curCols[projSize - 1];

            std::string fNewName = "mapped_" + fName;
            //std::string fNewName = fName;
            colToMapCol[fName] = fNewName;
            std::string fNewNameFull = "get_feature_vec(" + fName + ", " + std::to_string(NumDimLeft) + ") AS " + fNewName;
            //std::string fNewNameFull = "get_feature_vec(" + fName + ", " + NumDimLeft + ") AS " + fNewName;
            tabel2Columns[leftTable][projSize - 1] = fNewName;

            FeatureStatus fs = allFeatureStatus[fName];
            fs.featureStartPos = featureStartPos[leftTable];
            allFeatureStatus[fNewName] = fs;

            newProj.push_back(fNewNameFull);
            sources[leftTable] = sources[leftTable].project({newProj});

        }

        bool isRightTableNotLeaf = isInteger(rightTable);
        if (isRightTableNotLeaf == false) {
            // right table is leaf
            fCurrentTotal += NumDimRight;

            std::vector<std::string> curCols = tabel2Columns[rightTable];
            std::vector<std::string> newProj;
            int projSize = curCols.size();
            for (int i = 0; i < projSize - 1; i++) {
                newProj.push_back(curCols[i]);
            }

            std::string fName = curCols[projSize - 1];

            std::string fNewName = "mapped_" + fName;
            colToMapCol[fName] = fNewName;
            std::string fNewNameFull = "get_feature_vec(" + fName + ", " + std::to_string(NumDimRight) + ") AS " + fNewName;
            //std::string fNewNameFull = "get_feature_vec(" + fName + ", " + NumDimRight + ") AS " + fNewName;
            tabel2Columns[rightTable][projSize - 1] = fNewName;

            FeatureStatus fs = allFeatureStatus[fName];
            fs.featureStartPos = featureStartPos[rightTable];
            allFeatureStatus[fNewName] = fs;

            newProj.push_back(fNewNameFull);
            sources[rightTable] = sources[rightTable].project({newProj});

        }


    PlanBuilder left, right, out;

    //retrieve the corresponding PlanBuilder
    if (sources.count(leftTable) > 0) {

        left = sources[leftTable];

    }

    if (sources.count(rightTable) > 0) {

        right = sources[rightTable];

    }

    //compose join
    projections.clear();
    // Access Projection if it exists
    if (item.isMember("Projection")) {
        for (const auto& proj : item["Projection"]) {
             int is_mapped = colToMapCol.count(proj.asString());
             if (is_mapped >= 1) {
                 std::cout << colToMapCol[proj.asString()] << std::endl;
                 projections.push_back(colToMapCol[proj.asString()]);
             }
             else {
                 std::cout << proj.asString() << std::endl;
                 projections.push_back(proj.asString());
             }

	    }
    }

    std::cout << "Writing join" << std::endl;
    out = left.hashJoin(
        {item["ProbeKeys"].asString()},
        {item["BuildKeys"].asString()},
        right.planNode(),
        "",
            {projections}
    );
    std::cout << "Finished Writing join" << std::endl;

    outName = joinId;
    sources[outName] = out;

    }
    planBuilder = sources[outName];

    // After the final join
    std::string projString="";
    bool isFirst = true;
    for (std::string proj : projections) {
       int is_feature = allFeatureStatus.count(proj);
       if (is_feature >= 1) {
           //FeatureStatus fs = allFeatureStatus[proj];
           if (isFirst) {
               projString += proj;
	           isFirst = false;
           } else {
               projString += "," + proj;
           }
       }
    }
    projString = "concat(" + projString + ") as features";

    planBuilder = planBuilder.project({projString});
    std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
    return true;

}



  bool createAndExecuteQuery(int queryId, int modelId, const std::string& queryJsonStr, bool withFactorization) {
    PlanBuilder planBuilder{pool_.get()};
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();

    if (withFactorization) {
        // Rewriting with factorization
        std::string colFeature = rewriteWithFactorization(planBuilder, planNodeIdGenerator, queryJsonStr);

        // Step 2. Add model inference to query plan
       // addModelInferenceToQueryPlanAfterFactorize(planBuilder, planNodeIdGenerator, colFeature);
      }
      else {
          // Writing without factorization
          writeWithoutFactorization(planBuilder, planNodeIdGenerator, queryJsonStr);

          // Step 2. Add model inference to query plan
         // addModelInferenceToQueryPlan(planBuilder, planNodeIdGenerator);
      }

    //std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
      auto myPlan = planBuilder.planNode();
      std::cout << myPlan->toString(true, true) << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
      std::cout << "Results Size: " << results->size() << std::endl;
      std::cout << "Results:" << results->toString(0, 5) << std::endl;
      //std::cout << results->toString(0, results->size()) << std::endl;
      return true;
  }
};

int main(int argc, char** argv) {
  //setlocale(LC_TIME, "C");
  //malloc_trim(0);

  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  JobBenchmarkRewrite bench;
  std::cout
      << "[WARNING] the data path is hardcoded and needs to be modified accordingly."
      << std::endl;

  // Load Aka Name table
  RowVectorPtr akaNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_name.csv", "aka_name", 8);

  // Load Aka Title table
  RowVectorPtr akaTitleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_title.csv", "aka_title", 8);

  // Load Cast Info table
  RowVectorPtr castInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/cast_info.csv", "cast_info", 8);

  // Load Char Name table
  RowVectorPtr charNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/char_name.csv", "char_name", 8);

  // Load Comp Cast Type table
  RowVectorPtr compCastTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/comp_cast_type.csv", "comp_cast_type", 8);

  // Load Company Name table
  RowVectorPtr companyNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_name.csv", "company_name", 8);

  // Load Company Type table
  RowVectorPtr companyTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_type.csv", "company_type", 8);

  // Load Complete Cast table
  RowVectorPtr completeCastVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/complete_cast.csv", "complete_cast", 8);

  // Load Info Type table
  RowVectorPtr infoTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/info_type.csv", "info_type", 8);

  // Load Keyword table
  RowVectorPtr keywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/keyword.csv", "keyword", 8);

  // Load Kind Type table
  RowVectorPtr kindTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/kind_type.csv", "kind_type", 8);

  // Load Link Type table
  RowVectorPtr linkTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/link_type.csv", "link_type", 8);

  // Load Movie Companies table
  RowVectorPtr movieCompaniesVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_companies.csv", "movie_companies", 8);

  // Load Movie Info Index table
  RowVectorPtr movieInfoIdxVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info_idx.csv", "movie_info_idx", 8);

  // Load Movie Keyword table
  RowVectorPtr movieKeywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_keyword.csv", "movie_keyword", 8);

  // Load Movie Link table
  RowVectorPtr movieLinkVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_link.csv", "movie_link", 8);

  // Load Name table
  RowVectorPtr nameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/name.csv", "name", 8);

  // Load Role Type table
  RowVectorPtr roleTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/role_type.csv", "role_type", 8);

  // Load Title table
  RowVectorPtr titleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/title.csv", "title", 8);

  // Load Movie Info table
  RowVectorPtr movieInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info.csv", "movie_info", 8);

  // Load Person Info table
  RowVectorPtr personInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb//person_info.csv", "person_info", 8);

  bench.tableName2RowVector["aka_name"] = akaNameVec;

  bench.tableName2RowVector["aka_title"] = akaTitleVec;

  bench.tableName2RowVector["cast_info"] = castInfoVec;

  bench.tableName2RowVector["char_name"] = charNameVec;

  bench.tableName2RowVector["comp_cast_type"] = compCastTypeVec;

  bench.tableName2RowVector["company_name"] = companyNameVec;

  bench.tableName2RowVector["company_type"] = companyTypeVec;

  bench.tableName2RowVector["complete_cast"] = completeCastVec;

  bench.tableName2RowVector["info_type"] = infoTypeVec;

  bench.tableName2RowVector["keyword"] = keywordVec;

  bench.tableName2RowVector["kind_type"] = kindTypeVec;

  bench.tableName2RowVector["link_type"] = linkTypeVec;

  bench.tableName2RowVector["movie_companies"] = movieCompaniesVec;

  bench.tableName2RowVector["movie_info_idx"] = movieInfoIdxVec;

  bench.tableName2RowVector["movie_keyword"] = movieKeywordVec;

  bench.tableName2RowVector["movie_link"] = movieLinkVec;

  bench.tableName2RowVector["name"] = nameVec;

  bench.tableName2RowVector["role_type"] = roleTypeVec;

  bench.tableName2RowVector["title"] = titleVec;

  bench.tableName2RowVector["movie_info"] = movieInfoVec;

  bench.tableName2RowVector["person_info"] = personInfoVec;

  int queryId = bench.sampleQuery();

  int modelId = bench.sampleModel();

  exec::registerVectorFunction(
          "get_feature_vec",
          GetFeatureVec::signatures(),
          std::make_unique<GetFeatureVec>());
  std::cout << "Completed registering function for get_feature_vec" << std::endl;

  // retrieve the weights and set to map
  std::cout << "Reading model parameters" << std::endl;
  bench.findWeights("resources/model/job_any_64.h5");

  // retrieve the model operators from model expression IR
  std::string modelInput = "softmax3(mat_add3(mat_mul3(relu2(mat_add2(mat_mul2(relu1(mat_add1(mat_mul1(features)))))))))";
  std::cout << "Extracting model operators" << std::endl;
  std::vector<std::string> operators = bench.extractOperatorsInReverse(modelInput);
  bench.modelOperators = operators;

  std::cout << "Reading factorization Plan" << std::endl;
  bench.findFactorizationPlans("resources/plans/factorization_job_any_64.txt");

  std::string jsonString = bench.readQueryDetails("resources/plans/job_any.txt");
  bench.updateFeatureStatus(jsonString);

  std::cout << "Performing rewriting" << std::endl;
  bool ret = bench.createAndExecuteQuery(queryId, modelId, jsonString, false);

  return ret;
}


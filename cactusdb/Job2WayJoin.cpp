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

};



class GetZeroVec : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int64_t vecSizeLarge = 0;
    if (args.size() == 1) {
        // an optional parameter can be passed to enable the GPU for mat_mul
        vecSizeLarge = args[0]->as<ConstantVector<int64_t>>()->valueAt(0);
    }
    int vecSize = static_cast<int>(vecSizeLarge);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> vec;

        for (int j = 0; j < vecSize; j++) {
            vec.push_back(0.0);
        }
        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_zero_vec";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class Job2WayJoin : HiveConnectorTestBase {
 public:
  Job2WayJoin() {
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

  ~Job2WayJoin() {}

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
  std::unordered_map<std::string, std::vector<RowVectorPtr>> tableName2RowVectorBatches;
  std::unordered_map<std::string, std::vector<std::vector<float>>> operatorParam2Weights;
  std::unordered_map<std::string, std::vector<std::string>> tabel2Columns;
  std::vector<std::string> modelOperators;
  std::vector<std::string> modelOperators2;
  std::vector<std::string> modelOperators3;

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


  void findWeights2(const std::string& modelPath) {
            // read the parameter weights from file
            std::vector<std::vector<float>> w1 = loadHDF5Array(modelPath, "fc1.weight", 0);
            std::vector<std::vector<float>> b1 = loadHDF5Array(modelPath, "fc1.bias", 0);
            std::vector<std::vector<float>> w2 = loadHDF5Array(modelPath, "fc2.weight", 0);
            std::vector<std::vector<float>> b2 = loadHDF5Array(modelPath, "fc2.bias", 0);
            std::vector<std::vector<float>> w3 = loadHDF5Array(modelPath, "fc3.weight", 0);
            std::vector<std::vector<float>> b3 = loadHDF5Array(modelPath, "fc3.bias", 0);

            // store the weights in map with same name as operator
            operatorParam2Weights["mat_mul5"] = w1;
            operatorParam2Weights["mat_add5"] = b1;
            operatorParam2Weights["mat_mul6"] = w2;
            operatorParam2Weights["mat_add6"] = b2;
            operatorParam2Weights["mat_mul7"] = w3;
            operatorParam2Weights["mat_add7"] = b3;

            std::cout << "Shape of mat_mul5 weight: " << w1.size() << ", " << w1[0].size() << std::endl;
            std::cout << "Shape of mat_add5 weight: " << b1.size() << std::endl;
            std::cout << "Shape of mat_mul6 weight: " << w2.size() << ", " << w2[0].size() << std::endl;
            std::cout << "Shape of mat_add6 weight: " << b2.size() << std::endl;
            std::cout << "Shape of mat_mul7 weight: " << w3.size() << ", " << w3[0].size() << std::endl;
            std::cout << "Shape of mat_add7 weight: " << b3.size() << std::endl;
        }


  void findWeightsSingleLayer(const std::string& modelPath) {
            // read the parameter weights from file
            std::vector<std::vector<float>> w1 = loadHDF5Array(modelPath, "fc1.weight", 0);
            std::vector<std::vector<float>> b1 = loadHDF5Array(modelPath, "fc1.bias", 0);

            // store the weights in map with same name as operator
            operatorParam2Weights["mat_mul4"] = w1;
            operatorParam2Weights["mat_add4"] = b1;

            operatorParam2Weights["mat_mul8"] = w1;
            operatorParam2Weights["mat_add8"] = b1;
            operatorParam2Weights["mat_mul9"] = w1;
            operatorParam2Weights["mat_add9"] = b1;

            std::cout << "Shape of mat_mul4 weight: " << w1.size() << ", " << w1[0].size() << std::endl;
            std::cout << "Shape of mat_add4 weight: " << b1.size() << std::endl;
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




    PlanBuilder getTableBaseSources(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string tableAlias) {
      PlanBuilder basePlan;

      if (tableAlias == "an") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["an"]})
                    .project({"person_id as an_person_id", "name as an_name", "imdb_index as an_imdb_index", "an_features"});
          tabel2Columns["an"] = {"an_person_id", "an_name", "an_imdb_index", "an_features"};
      }
      else if (tableAlias == "at") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["at"]})
                    .project({"id as at_id", "movie_id as at_movie_id", "title as at_title", "imdb_index as at_imdb_index", "kind_id as at_kind_id", "at_features"})
                    .limit(0, 100, false);
          tabel2Columns["at"] = {"at_id", "at_movie_id", "at_title", "at_imdb_index", "at_kind_id", "at_features"};
      }
      else if (tableAlias == "ci") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["ci"]})
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
      }
      else if (tableAlias == "chn") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["chn"]})
                     .project({"id as chn_id", "name as chn_name", "imdb_index as chn_imdb_index", "imdb_id as chn_imdb_id", "chn_features"});
          tabel2Columns["chn"] = {"chn_id", "chn_name", "chn_imdb_index", "chn_imdb_id", "chn_features"};
      }
      else if (tableAlias == "cc") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cc"]})
            .project({"id as cc_id", "movie_id as cc_movie_id", "subject_id as cc_subject_id", "status_id as cc_status_id", "cc_features"});
          tabel2Columns["cc"] = {"cc_id", "cc_movie_id", "cc_subject_id", "cc_status_id", "cc_features"};
      }
      else if (tableAlias == "cct") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cct"]})
            .project({"id as cct_id", "kind as cct_kind", "cct_features"});
          tabel2Columns["cct"] = {"cct_id", "cct_kind", "cct_features"};
      }
      else if (tableAlias == "cn") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["cn"]})
                    .project({"id as cn_id", "name as cn_name", "country_code as cn_country_code", "imdb_id as cn_imdb_id", "cn_features"});
          tabel2Columns["cn"] = {"cn_id", "cn_name", "cn_country_code", "cn_imdb_id", "cn_features"};
      }
      else if (tableAlias == "ct") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["ct"]})
                    .project({"id as ct_id", "kind as ct_kind", "ct_features"});
          tabel2Columns["ct"] = {"ct_id", "ct_kind", "ct_features"};
      }
      else if (tableAlias == "it") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["it"]})
                    .project({"id as it_id", "info as it_info", "it_features"});
          tabel2Columns["it"] = {"it_id", "it_info", "it_features"};
      }
      else if (tableAlias == "kt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["kt"]})
                    .project({"id as kt_id", "kind as kt_kind", "kt_features"});
          tabel2Columns["kt"] = {"kt_id", "kt_kind", "kt_features"};
      }
      else if (tableAlias == "lt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["lt"]})
                    .project({"id as lt_id", "link as lt_link", "lt_features"});
          tabel2Columns["lt"] = {"lt_id", "lt_link", "lt_features"};
      }
      else if (tableAlias == "k") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["k"]})
                   .project({"id as k_id", "keyword as k_keyword", "k_features"});
          tabel2Columns["k"] = {"k_id", "k_keyword", "k_features"};
      }
      else if (tableAlias == "mc") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["mc"]})
                    .project({"id as mc_id", "movie_id as mc_movie_id", "company_id as mc_company_id", "company_type_id as mc_company_type_id", "mc_features"});
                   // .limit(0, 260912, false);
          tabel2Columns["mc"] = {"mc_id", "mc_movie_id", "mc_company_id", "mc_company_type_id", "mc_features"};
      }
      else if (tableAlias == "mi") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["mi"]})
            .localPartition({"info_type_id"})
            .project({"id as mi_id", "movie_id as mi_movie_id", "info_type_id as mi_info_type_id", "info as mi_info", "mi_features"});
            //.limit(0, 1483572, false);
          tabel2Columns["mi"] = {"mi_id", "mi_movie_id", "mi_info_type_id", "mi_info", "mi_features"};
      }
      else if (tableAlias == "mii") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["mii"]})
            .project({"id as mii_id", "movie_id as mii_movie_id", "info_type_id as mii_info_type_id", "info as mii_info", "mii_features"});
          tabel2Columns["mii"] = {"mii_id", "mii_movie_id", "mii_info_type_id", "mii_info", "mii_features"};
      }
      else if (tableAlias == "mk") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["mk"]})
                    .project({"id as mk_id", "movie_id as mk_movie_id", "keyword_id as mk_keyword_id", "mk_features"});
          tabel2Columns["mk"] = {"mk_id", "mk_movie_id", "mk_keyword_id", "mk_features"};
      }
      else if (tableAlias == "ml") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["ml"]})
                    .project({"id as ml_id", "movie_id as ml_movie_id", "linked_movie_id as ml_linked_movie_id", "link_type_id as ml_link_type_id", "ml_features"});
          tabel2Columns["ml"] = {"ml_id", "ml_movie_id", "ml_linked_movie_id", "ml_link_type_id", "ml_features"};
      }
      else if (tableAlias == "n") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["n"]})
                   .project({"id as n_id", "name as n_name", "imdb_index as n_imdb_index", "imdb_id as n_imdb_id", "n_features"});
          tabel2Columns["n"] = {"n_id", "n_name", "n_imdb_index", "n_imdb_id", "n_features"};
      }
      else if (tableAlias == "pi") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["pi"]})
                    .project(
                        {"id as pi_id",
                         "person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
          tabel2Columns["pi"] = {"pi_id", "pi_person_id", "pi_info_type_id", "pi_features"};
      }
      else if (tableAlias == "rt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["rt"]})
                    .project({"id as rt_id", "role as rt_role", "rt_features"});
          tabel2Columns["rt"] = {"rt_id", "rt_role", "rt_features"};
      }
      else if (tableAlias == "t") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["t"]})
            .project({"id as t_id", "title as t_title", "imdb_index as t_imdb_index", "kind_id as t_kind_id", "imdb_id as t_imdb_id", "t_features"});
          tabel2Columns["t"] = {"t_id", "t_title", "t_imdb_index", "t_kind_id", "t_imdb_id", "t_features"};
      }

    return basePlan;
  }


  PlanBuilder getTableBaseSourcesInBatches(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string tableAlias) {
      PlanBuilder basePlan;

      if (tableAlias == "an") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["an"])
                    .project({"person_id as an_person_id", "name as an_name", "imdb_index as an_imdb_index", "an_features"});
          tabel2Columns["an"] = {"an_person_id", "an_name", "an_imdb_index", "an_features"};
      }
      else if (tableAlias == "at") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["at"])
                    .project({"id as at_id", "movie_id as at_movie_id", "title as at_title", "imdb_index as at_imdb_index", "kind_id as at_kind_id", "at_features"})
                    .limit(0, 100, false);
          tabel2Columns["at"] = {"at_id", "at_movie_id", "at_title", "at_imdb_index", "at_kind_id", "at_features"};
      }
      else if (tableAlias == "ci") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["ci"])
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
      }
      else if (tableAlias == "chn") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values(tableName2RowVectorBatches["chn"])
                     .project({"id as chn_id", "name as chn_name", "imdb_index as chn_imdb_index", "imdb_id as chn_imdb_id", "chn_features"});
          tabel2Columns["chn"] = {"chn_id", "chn_name", "chn_imdb_index", "chn_imdb_id", "chn_features"};
      }
      else if (tableAlias == "cc") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["cc"])
            .project({"id as cc_id", "movie_id as cc_movie_id", "subject_id as cc_subject_id", "status_id as cc_status_id", "cc_features"});
          tabel2Columns["cc"] = {"cc_id", "cc_movie_id", "cc_subject_id", "cc_status_id", "cc_features"};
      }
      else if (tableAlias == "cct") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["cct"])
            .project({"id as cct_id", "kind as cct_kind", "cct_features"});
          tabel2Columns["cct"] = {"cct_id", "cct_kind", "cct_features"};
      }
      else if (tableAlias == "cn") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["cn"])
                    .project({"id as cn_id", "name as cn_name", "country_code as cn_country_code", "imdb_id as cn_imdb_id", "cn_features"});
          tabel2Columns["cn"] = {"cn_id", "cn_name", "cn_country_code", "cn_imdb_id", "cn_features"};
      }
      else if (tableAlias == "ct") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["ct"])
                    .project({"id as ct_id", "kind as ct_kind", "ct_features"});
          tabel2Columns["ct"] = {"ct_id", "ct_kind", "ct_features"};
      }
      else if (tableAlias == "it") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["it"])
                    .project({"id as it_id", "info as it_info", "it_features"});
          tabel2Columns["it"] = {"it_id", "it_info", "it_features"};
      }
      else if (tableAlias == "kt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["kt"])
                    .project({"id as kt_id", "kind as kt_kind", "kt_features"});
          tabel2Columns["kt"] = {"kt_id", "kt_kind", "kt_features"};
      }
      else if (tableAlias == "lt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["lt"])
                    .project({"id as lt_id", "link as lt_link", "lt_features"});
          tabel2Columns["lt"] = {"lt_id", "lt_link", "lt_features"};
      }
      else if (tableAlias == "k") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values(tableName2RowVectorBatches["k"])
                   .project({"id as k_id", "keyword as k_keyword", "k_features"});
          tabel2Columns["k"] = {"k_id", "k_keyword", "k_features"};
      }
      else if (tableAlias == "mc") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["mc"])
                    .project({"id as mc_id", "movie_id as mc_movie_id", "company_id as mc_company_id", "company_type_id as mc_company_type_id", "mc_features"});
                    //.limit(0, 260912, false);
          tabel2Columns["mc"] = {"mc_id", "mc_movie_id", "mc_company_id", "mc_company_type_id", "mc_features"};
      }
      else if (tableAlias == "mi") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["mi"])
            .project({"id as mi_id", "movie_id as mi_movie_id", "info_type_id as mi_info_type_id", "info as mi_info", "mi_features"});
            //.limit(0, 1483572, false);
          tabel2Columns["mi"] = {"mi_id", "mi_movie_id", "mi_info_type_id", "mi_info", "mi_features"};
      }
      else if (tableAlias == "mii") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["mii"])
            .project({"id as mii_id", "movie_id as mii_movie_id", "info_type_id as mii_info_type_id", "info as mii_info", "mii_features"});
          tabel2Columns["mii"] = {"mii_id", "mii_movie_id", "mii_info_type_id", "mii_info", "mii_features"};
      }
      else if (tableAlias == "mk") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["mk"])
                    .project({"id as mk_id", "movie_id as mk_movie_id", "keyword_id as mk_keyword_id", "mk_features"});
          tabel2Columns["mk"] = {"mk_id", "mk_movie_id", "mk_keyword_id", "mk_features"};
      }
      else if (tableAlias == "ml") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["ml"])
                    .project({"id as ml_id", "movie_id as ml_movie_id", "linked_movie_id as ml_linked_movie_id", "link_type_id as ml_link_type_id", "ml_features"});
          tabel2Columns["ml"] = {"ml_id", "ml_movie_id", "ml_linked_movie_id", "ml_link_type_id", "ml_features"};
      }
      else if (tableAlias == "n") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values(tableName2RowVectorBatches["n"])
                   .project({"id as n_id", "name as n_name", "imdb_index as n_imdb_index", "imdb_id as n_imdb_id", "n_features"});
          tabel2Columns["n"] = {"n_id", "n_name", "n_imdb_index", "n_imdb_id", "n_features"};
      }
      else if (tableAlias == "pi") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["pi"])
                    .project(
                        {"id as pi_id",
                         "person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
          tabel2Columns["pi"] = {"pi_id", "pi_person_id", "pi_info_type_id", "pi_features"};
      }
      else if (tableAlias == "rt") {
          basePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(tableName2RowVectorBatches["rt"])
                    .project({"id as rt_id", "role as rt_role", "rt_features"});
          tabel2Columns["rt"] = {"rt_id", "rt_role", "rt_features"};
      }
      else if (tableAlias == "t") {
          basePlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values(tableName2RowVectorBatches["t"])
            .project({"id as t_id", "title as t_title", "imdb_index as t_imdb_index", "kind_id as t_kind_id", "imdb_id as t_imdb_id", "t_features"});
          tabel2Columns["t"] = {"t_id", "t_title", "t_imdb_index", "t_kind_id", "t_imdb_id", "t_features"};
      }

    return basePlan;
  }



  std::vector<RowVectorPtr> convertRowVectorToBatches(RowVectorPtr rowVector, int batch_counts) {
     int totalRows = rowVector->size();
     if (batch_counts > totalRows) {
         batch_counts = totalRows;
     }
     int batchSize = totalRows / batch_counts;
     std::vector<RowVectorPtr> batchesVector;

     for (int i = 0; i < batch_counts; ++i) {
         int start = i * batchSize;
         int end = (i == (batch_counts - 1)) ? totalRows : (i + 1) * batchSize;  // Handle remainder for last batch
         batchesVector.push_back(std::dynamic_pointer_cast<RowVector>(rowVector->slice(start, end - start)));
     }

     return batchesVector;
  }


  void loadDataTable(std::string path, std::string table_alias) {
      if (table_alias == "an") {
          // Load Aka Name table
          RowVectorPtr akaNameVec = getTableFromCSVFile(maker, path + "/aka_name.csv", "aka_name", 8);
          tableName2RowVector["an"] = akaNameVec;
      }
      else if (table_alias == "at") {
          // Load Aka Title table
          RowVectorPtr akaTitleVec = getTableFromCSVFile(maker, path + "/aka_title.csv", "aka_title", 8);
          tableName2RowVector["at"] = akaTitleVec;
      }
      else if (table_alias == "ci") {
          // Load Cast Info table
          RowVectorPtr castInfoVec = getTableFromCSVFile(maker, path + "/cast_info.csv", "cast_info", 8);
          tableName2RowVector["ci"] = castInfoVec;
      }
      else if (table_alias == "chn") {
          // Load Char Name table
          RowVectorPtr charNameVec = getTableFromCSVFile(maker, path + "/char_name.csv", "char_name", 8);
          tableName2RowVector["chn"] = charNameVec;
      }
      else if (table_alias == "cct") {
          // Load Comp Cast Type table
          RowVectorPtr compCastTypeVec = getTableFromCSVFile(maker, path + "/comp_cast_type.csv", "comp_cast_type", 8);
          tableName2RowVector["cct"] = compCastTypeVec;
      }
      else if (table_alias == "cn") {
          // Load Company Name table
          RowVectorPtr companyNameVec = getTableFromCSVFile(maker, path + "/company_name.csv", "company_name", 8);
          tableName2RowVector["cn"] = companyNameVec;
      }
      else if (table_alias == "ct") {
          // Load Company Type table
          RowVectorPtr companyTypeVec = getTableFromCSVFile(maker, path + "/company_type.csv", "company_type", 8);
          tableName2RowVector["ct"] = companyTypeVec;
      }
      else if (table_alias == "cc") {
          // Load Complete Cast table
          RowVectorPtr completeCastVec = getTableFromCSVFile(maker, path + "/complete_cast.csv", "complete_cast", 8);
          tableName2RowVector["cc"] = completeCastVec;
      }
      else if (table_alias == "it") {
          // Load Info Type table
          RowVectorPtr infoTypeVec = getTableFromCSVFile(maker, path + "/info_type.csv", "info_type", 8);
          tableName2RowVector["it"] = infoTypeVec;
      }
      else if (table_alias == "k") {
          // Load Keyword table
          RowVectorPtr keywordVec = getTableFromCSVFile(maker, path + "/keyword.csv", "keyword", 8);
          tableName2RowVector["k"] = keywordVec;
      }
      else if (table_alias == "kt") {
          // Load Kind Type table
          RowVectorPtr kindTypeVec = getTableFromCSVFile(maker, path + "/kind_type.csv", "kind_type", 8);
          tableName2RowVector["kt"] = kindTypeVec;
      }
      else if (table_alias == "lt") {
          // Load Link Type table
          RowVectorPtr linkTypeVec = getTableFromCSVFile(maker, path + "/link_type.csv", "link_type", 8);
          tableName2RowVector["lt"] = linkTypeVec;
      }
      else if (table_alias == "mc") {
          // Load Movie Companies table
          RowVectorPtr movieCompaniesVec = getTableFromCSVFile(maker, path + "/movie_companies.csv", "movie_companies", 8);
          tableName2RowVector["mc"] = movieCompaniesVec;
      }
      else if (table_alias == "mii") {
          // Load Movie Info Index table
          RowVectorPtr movieInfoIdxVec = getTableFromCSVFile(maker, path + "/movie_info_idx.csv", "movie_info_idx", 8);
          tableName2RowVector["mii"] = movieInfoIdxVec;
      }
      else if (table_alias == "mk") {
          // Load Movie Keyword table
          RowVectorPtr movieKeywordVec = getTableFromCSVFile(maker, path + "/movie_keyword.csv", "movie_keyword", 8);
          tableName2RowVector["mk"] = movieKeywordVec;
      }
      else if (table_alias == "ml") {
          // Load Movie Link table
          RowVectorPtr movieLinkVec = getTableFromCSVFile(maker, path + "/movie_link.csv", "movie_link", 8);
          tableName2RowVector["ml"] = movieLinkVec;
      }
      else if (table_alias == "n") {
          // Load Name table
          RowVectorPtr nameVec = getTableFromCSVFile(maker, path + "/name.csv", "name", 8);
          tableName2RowVector["n"] = nameVec;
      }
      else if (table_alias == "rt") {
          // Load Role Type table
          RowVectorPtr roleTypeVec = getTableFromCSVFile(maker, path + "/role_type.csv", "role_type", 8);
          tableName2RowVector["rt"] = roleTypeVec;
      }
      else if (table_alias == "t") {
          // Load Title table
          RowVectorPtr titleVec = getTableFromCSVFile(maker, path + "/title.csv", "title", 8);
          tableName2RowVector["t"] = titleVec;
      }
      else if (table_alias == "mi") {
          // Load Movie Info table
          RowVectorPtr movieInfoVec = getTableFromCSVFile(maker, path + "/movie_info.csv", "movie_info", 8);
          tableName2RowVector["mi"] = movieInfoVec;
      }
      else if (table_alias == "pi") {
          // Load Person Info table
          RowVectorPtr personInfoVec = getTableFromCSVFile(maker, path + "/person_info.csv", "person_info", 8);
          tableName2RowVector["pi"] = personInfoVec;
      }
  }



// Function to extract the part of the column name after the first "_"
std::string extractColumnName(const std::string& fullName) {
    size_t pos = fullName.find('_'); // Find first "_"
    return (pos != std::string::npos) ? fullName.substr(pos + 1) : fullName; // Return part after "_"
}



// Function to sort a RowVectorPtr and return a new sorted RowVectorPtr
RowVectorPtr sortRowVectorByColumn(RowVectorPtr rowVector, const std::string& columnName) {
    // Get schema type
    auto rowType = std::dynamic_pointer_cast<const RowType>(rowVector->type());
    if (!rowType) {
        throw std::runtime_error("Invalid RowVector type.");
    }

    // Extract right part of column name after "_"
    std::string targetColumn = extractColumnName(columnName);

    // Find column index by name
    /*auto it = std::find(rowType->names().begin(), rowType->names().end(), targetColumn);
    if (it == rowType->names().end()) {
        throw std::runtime_error("Column name not found: " + targetColumn);
    }
    int columnIndex = std::distance(rowType->names().begin(), it);*/

    int columnIndex = -1;
    std::vector<std::string> allCols;
    // Iterate through column names and indices
    for (size_t i = 0; i < rowType->names().size(); i++) {
        std::cout << "Index: " << i << ", Name: " << rowType->names()[i] << "\n";
        std::string curCol = rowType->names()[i];
        if (curCol == targetColumn) {
            columnIndex = i;
        }
        allCols.push_back(curCol);
    }
    if (columnIndex == -1) {
        throw std::runtime_error("Column name not found: " + targetColumn);
    }

    // Extract column data (Assuming BIGINT type, modify for other types)
    auto column = rowVector->childAt(columnIndex)->asFlatVector<int64_t>();

    // Get number of rows
    size_t numRows = rowVector->size();

    // Create an index vector for sorting
    std::vector<int> indices(numRows);
    for (size_t i = 0; i < numRows; i++) {
        indices[i] = i;
    }

    // Sort indices based on column values
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return column->valueAt(a) < column->valueAt(b);
    });

    // Create a new sorted RowVector
    std::vector<VectorPtr> sortedChildren;
    for (size_t i = 0; i < rowVector->childrenSize(); i++) {
        auto child = rowVector->childAt(i);
        auto sortedChild = BaseVector::create(child->type(), numRows, rowVector->pool());

        for (size_t j = 0; j < numRows; j++) {
            sortedChild->copy(child.get(), j, indices[j], 1);
        }
        sortedChildren.push_back(sortedChild);
    }

    // Return new sorted RowVectorPtr
    return maker.rowVector(allCols, sortedChildren);
    /*return std::make_shared<RowVector>(
        rowVector->pool(),
        rowVector->type(),
        nullptr,
        numRows,
        sortedChildren);*/
}





bool addModelInferenceToQueryPlanAfterFactorize(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, const std::string& colFeature) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        for (int i = modelOperators.size() - 1; i > 0; i--) {
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
        modelProjString += colFeature;

        for (int i = modelOperators.size() - 1; i > 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;
}



bool addModelMultiInferenceToQueryPlanAfterFactorize(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, const std::string& colFeature, const std::string& colFeature2, const std::string& colFeature3) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        for (int i = modelOperators.size() - 1; i > 0; i--) {
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
        modelProjString += colFeature;

        for (int i = modelOperators.size() - 1; i > 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";


        std::string modelProjString2 = "";
        for (int i = modelOperators2.size() - 1; i > 0; i--) {
            std::string opName = modelOperators2[i];
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
            modelProjString2 += opName + "(";
        }
        modelProjString2 += colFeature2;

        for (int i = modelOperators2.size() - 1; i > 0; i--) {
            modelProjString2 += ")";
        }
        modelProjString2 += " AS output2";


        std::string modelProjString3 = "";
        for (int i = modelOperators3.size() - 1; i > 0; i--) {
            std::string opName = modelOperators3[i];
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
            modelProjString3 += opName + "(";
        }
        modelProjString3 += colFeature3;

        for (int i = modelOperators3.size() - 1; i > 0; i--) {
            modelProjString3 += ")";
        }
        modelProjString3 += " AS output3";


        std::cout << "Inference Part: " << modelProjString << endl;
        std::cout << "Inference Part2: " << modelProjString2 << endl;
        std::cout << "Inference Part3: " << modelProjString3 << endl;
        planBuilder.project({modelProjString, modelProjString2, modelProjString3});

      return true;
}



/*
param pushDown: 0 -> no push, 1 -> all push, 2 -> left push, 3 -> right push
*/
bool createAndExecuteQuery(std::string leftTable, std::string rightTable, std::string probKey, std::string buildKey, int dimLeft, int dimRight, bool isBatchExecution, int batchCount, int driverCount, int pushDown) {
    std::string firstOpName = modelOperators[0]; // name of operator of split layer
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName]; // weight of the split layer
    int numCols = firstWeight.size(); // number of columns in split layer
    int numNeurons = firstWeight[0].size(); // number of neurons in split layer

    //registering vector addition operator for later use as aggregation
    std::vector<std::vector<float>> emptyMatrix;
    //registerNNFunction("vector_addition", emptyMatrix, numNeurons, -1);

    if (isBatchExecution) {
        // convert loaded RowVectors to Batches
        tableName2RowVectorBatches[leftTable] = convertRowVectorToBatches(tableName2RowVector[leftTable], batchCount);
        tableName2RowVectorBatches[rightTable] = convertRowVectorToBatches(tableName2RowVector[rightTable], batchCount);
        //tableName2RowVectorBatches[leftTable] = convertRowVectorToBatches(sortRowVectorByColumn(tableName2RowVector[leftTable], probKey), batchCount);
        //tableName2RowVectorBatches[rightTable] = convertRowVectorToBatches(sortRowVectorByColumn(tableName2RowVector[rightTable], buildKey), batchCount);
    }
    /*else {
        tableName2RowVector[leftTable] = sortRowVectorByColumn(tableName2RowVector[leftTable], probKey);
        tableName2RowVector[rightTable] = sortRowVectorByColumn(tableName2RowVector[rightTable], buildKey);
    }*/
    //std::cout << "Sorting Done" << std::endl;

    PlanBuilder planBuilder{pool_.get()};
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // fetch plan builders corresponding to left and right table
    PlanBuilder leftPlan;
    PlanBuilder rightPlan;
    if (isBatchExecution) {
        leftPlan = getTableBaseSourcesInBatches(planNodeIdGenerator, leftTable);
        rightPlan = getTableBaseSourcesInBatches(planNodeIdGenerator, rightTable);
    }
    else {
        leftPlan = getTableBaseSources(planNodeIdGenerator, leftTable);
        rightPlan = getTableBaseSources(planNodeIdGenerator, rightTable);
    }

    // form the left and right feature columns
    std::string lFeatureName = leftTable + "_features";
    std::string rFeatureName = rightTable + "_features";

    std::vector<std::string> projectionsInit;
    projectionsInit.push_back(probKey);
    projectionsInit.push_back(buildKey);

    std::vector<std::string> projections;
    projections.push_back(probKey);
    projections.push_back(buildKey);

    std::string fNewNameInit; //new name after applying operator on feature
    std::string fNewNameFull; // fNewName with its full projection details
    std::string fNewName; //new name after applying coalesce on feature

    // checking if push left feature
    if (pushDown == 1 || pushDown == 2) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        fNewNameInit = "factorized_left_init";
        fNewNameFull = newOpName + "(get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ")) AS " + fNewNameInit;
        fNewName = "coalesce(" + fNewNameInit + ", get_zero_vec(" + std::to_string(numNeurons) + ")) AS factorized_left";
    }
    else {
        fNewNameInit = "mapped_left_init";
        fNewNameFull = "get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ") AS " + fNewNameInit;
        fNewName = "coalesce(" + fNewNameInit + ", get_zero_vec(" + std::to_string(dimLeft) + ")) AS mapped_left";
    }
    projectionsInit.push_back(fNewNameInit);
    projections.push_back(fNewName);
    leftPlan= leftPlan.project({probKey, fNewNameFull});

    // checking if push right feature
    if (pushDown == 1 || pushDown == 3) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        fNewNameInit = "factorized_right_init";
        fNewNameFull = newOpName + "(get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ")) AS " + fNewNameInit;
        fNewName = "coalesce(" + fNewNameInit + ", get_zero_vec(" + std::to_string(numNeurons) + ")) AS factorized_right";
    }
    else {
        fNewNameInit = "mapped_right_init";
        fNewNameFull = "get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ") AS " + fNewNameInit;
        fNewName = "coalesce(" + fNewNameInit + ", get_zero_vec(" + std::to_string(dimRight) + ")) AS mapped_right";
    }
    projectionsInit.push_back(fNewNameInit);
    projections.push_back(fNewName);
    rightPlan= rightPlan.project({buildKey, fNewNameFull});

    // Perform the join
    PlanBuilder out = leftPlan.hashJoin(
        {probKey},
        {buildKey},
        rightPlan.planNode(),
        "",
        {projectionsInit},
        core::JoinType::kFull
    );
    //planBuilder = out.project({projections});
    out = out.project({projections});

    // Apply operators after join
    std::string projString;

    if (pushDown == 0) {
        // no features pushed, so concatenate them and apply first operator in the model
        registerNNFunction(firstOpName, firstWeight, dimLeft + dimRight, numNeurons);
        projString = firstOpName + "(concat(mapped_left, mapped_right)) AS features";
        //projString = "concat(mapped_left, mapped_right) AS features";
    }
    else if (pushDown == 1) {
        // all features pushed, so just perform aggregation
        projString = "vector_addition(factorized_left, factorized_right) AS features";
    }
    else if (pushDown == 2) {
        // only left features pushed, so apply first operator on right feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_right)";

        projString = "vector_addition(factorized_left, " + fNewNameFull + ") AS features";
    }

    else {
        // only right features pushed, so apply first operator on left feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_left)";

        projString = "vector_addition(" + fNewNameFull + ", factorized_right) AS features";
    }

    planBuilder = out.project({projString});
    //addModelInferenceToQueryPlanAfterFactorize(planBuilder, planNodeIdGenerator, "features");

      auto myPlan = planBuilder.planNode();
      std::cout << myPlan->toString(true, true) << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      if (isBatchExecution) {
          auto results = exec::test::AssertQueryBuilder(myPlan).maxDrivers(driverCount).copyResults(pool_.get());
          std::cout << "Results Size: " << results->size() << std::endl;
          std::cout << "Results:" << results->toString(0, 5) << std::endl;
      }
      else {
          auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
          std::cout << "Results Size: " << results->size() << std::endl;
          std::cout << "Results:" << results->toString(0, 5) << std::endl;
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
      return true;
  }



  /*
param pushDown: 0 -> no push, 1 -> all push, 2 -> left push, 3 -> right push
*/
bool createAndExecuteMultiTaskQuery(std::string leftTable, std::string rightTable, std::string probKey, std::string buildKey, int dimLeft, int dimRight, bool isBatchExecution, int batchCount, int driverCount, int pushDown) {
    std::string firstOpName = modelOperators[0]; // name of operator of split layer
    std::cout << "First operator: " << firstOpName << std::endl;
    std::string firstOpName2 = modelOperators2[0]; // name of operator of split layer
    std::cout << "First operator2: " << firstOpName2 << std::endl;
    std::string firstOpName3 = modelOperators3[0]; // name of operator of split layer
    std::cout << "First operator3: " << firstOpName3 << std::endl;
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName]; // weight of the split layer
    std::vector<std::vector<float>> firstWeight2 = operatorParam2Weights[firstOpName2]; // weight of the split layer
    std::vector<std::vector<float>> firstWeight3 = operatorParam2Weights[firstOpName3]; // weight of the split layer
    int numCols = firstWeight.size(); // number of columns in split layer
    int numNeurons = firstWeight[0].size(); // number of neurons in split layer
    int numNeurons2 = firstWeight2[0].size(); // number of neurons in split layer
    int numNeurons3 = firstWeight3[0].size(); // number of neurons in split layer
    std::cout << "Num Neurons: " << numNeurons << std::endl;
    std::cout << "Num Neurons2: " << numNeurons2 << std::endl;
    std::cout << "Num Neurons3: " << numNeurons3 << std::endl;

    //registering vector addition operator for later use as aggregation
    std::vector<std::vector<float>> emptyMatrix;
    registerNNFunction("vector_addition", emptyMatrix, numNeurons, -1);
    registerNNFunction("vector_addition2", emptyMatrix, numNeurons2, -1);
    registerNNFunction("vector_addition3", emptyMatrix, numNeurons3, -1);
    std::cout << "Registered both vector addition functions" << std::endl;

    if (isBatchExecution) {
        // convert loaded RowVectors to Batches
        tableName2RowVectorBatches[leftTable] = convertRowVectorToBatches(tableName2RowVector[leftTable], batchCount);
        tableName2RowVectorBatches[rightTable] = convertRowVectorToBatches(tableName2RowVector[rightTable], batchCount);
        //tableName2RowVectorBatches[leftTable] = convertRowVectorToBatches(sortRowVectorByColumn(tableName2RowVector[leftTable], probKey), batchCount);
        //tableName2RowVectorBatches[rightTable] = convertRowVectorToBatches(sortRowVectorByColumn(tableName2RowVector[rightTable], buildKey), batchCount);
    }
    /*else {
        tableName2RowVector[leftTable] = sortRowVectorByColumn(tableName2RowVector[leftTable], probKey);
        tableName2RowVector[rightTable] = sortRowVectorByColumn(tableName2RowVector[rightTable], buildKey);
    }*/
    //std::cout << "Sorting Done" << std::endl;
    std::cout << "Converted row vectors to batches" << std::endl;

    PlanBuilder planBuilder{pool_.get()};
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // fetch plan builders corresponding to left and right table
    PlanBuilder leftPlan;
    PlanBuilder rightPlan;
    if (isBatchExecution) {
        leftPlan = getTableBaseSourcesInBatches(planNodeIdGenerator, leftTable);
        rightPlan = getTableBaseSourcesInBatches(planNodeIdGenerator, rightTable);
    }
    else {
        leftPlan = getTableBaseSources(planNodeIdGenerator, leftTable);
        rightPlan = getTableBaseSources(planNodeIdGenerator, rightTable);
    }
    std::cout << "Retrieved left and right plans" << std::endl;

    // form the left and right feature columns
    std::string lFeatureName = leftTable + "_features";
    std::string rFeatureName = rightTable + "_features";

    std::vector<std::string> projections;
    projections.push_back(probKey);
    projections.push_back(buildKey);

    std::string fNewName; //new name after applying operator on feature
    std::string fNewNameFull; // fNewName with its full projection details
    std::string fNewName2; //new name after applying operator on feature
    std::string fNewNameFull2; // fNewName with its full projection details
    std::string fNewName3; //new name after applying operator on feature
    std::string fNewNameFull3; // fNewName with its full projection details

    // checking if push left feature
    if (pushDown == 1 || pushDown == 2) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        fNewName = "factorized_left";
        fNewNameFull = newOpName + "(get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ")) AS " + fNewName;

        std::vector<std::vector<float>> subWeight2 = extractSubweight(firstWeight2, 0, dimLeft);
        std::string newOpName2 = firstOpName2 + "_left";
        registerNNFunction(newOpName2, subWeight2, dimLeft, numNeurons2);
        fNewName2 = "factorized_left2";
        fNewNameFull2 = newOpName2 + "(get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ")) AS " + fNewName2;

        std::vector<std::vector<float>> subWeight3 = extractSubweight(firstWeight3, 0, dimLeft);
        std::string newOpName3 = firstOpName3 + "_left";
        registerNNFunction(newOpName3, subWeight3, dimLeft, numNeurons3);
        fNewName3 = "factorized_left3";
        fNewNameFull3 = newOpName3 + "(get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ")) AS " + fNewName3;

        projections.push_back(fNewName);
        projections.push_back(fNewName2);
        projections.push_back(fNewName3);
        leftPlan= leftPlan.project({probKey, fNewNameFull, fNewNameFull2, fNewNameFull3});
    }
    else {
        fNewName = "mapped_left";
        fNewNameFull = "get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ") AS " + fNewName;

        projections.push_back(fNewName);
        leftPlan= leftPlan.project({probKey, fNewNameFull});
    }
    //projections.push_back(fNewName);
    //leftPlan= leftPlan.project({probKey, fNewNameFull});

    // checking if push right feature
    if (pushDown == 1 || pushDown == 3) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        fNewName = "factorized_right";
        fNewNameFull = newOpName + "(get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ")) AS " + fNewName;

        std::vector<std::vector<float>> subWeight2 = extractSubweight(firstWeight2, dimLeft, dimRight);
        std::string newOpName2 = firstOpName2 + "_right";
        registerNNFunction(newOpName2, subWeight2, dimRight, numNeurons2);
        fNewName2 = "factorized_right2";
        fNewNameFull2 = newOpName2 + "(get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ")) AS " + fNewName2;

        std::vector<std::vector<float>> subWeight3 = extractSubweight(firstWeight3, dimLeft, dimRight);
        std::string newOpName3 = firstOpName3 + "_right";
        registerNNFunction(newOpName3, subWeight3, dimRight, numNeurons3);
        fNewName3 = "factorized_right3";
        fNewNameFull3 = newOpName3 + "(get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ")) AS " + fNewName3;

        projections.push_back(fNewName);
        projections.push_back(fNewName2);
        projections.push_back(fNewName3);
        rightPlan= rightPlan.project({buildKey, fNewNameFull, fNewNameFull2, fNewNameFull3});
    }
    else {
        fNewName = "mapped_right";
        fNewNameFull = "get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ") AS " + fNewName;

        projections.push_back(fNewName);
        rightPlan= rightPlan.project({buildKey, fNewNameFull});
    }
    //projections.push_back(fNewName);
    //rightPlan= rightPlan.project({buildKey, fNewNameFull});
    std::cout << "Good before adding joins" << std::endl;

    // Perform the join
    PlanBuilder out = leftPlan.hashJoin(
        {probKey},
        {buildKey},
        rightPlan.planNode(),
        "",
        {projections},
        core::JoinType::kLeft
    );
    std::cout << "Added Joins" << std::endl;

    // Apply operators after join
    std::string projString;
    std::string projString2;
    std::string projString3;

    if (pushDown == 0) {
        // no features pushed, so concatenate them and apply first operator in the model
        registerNNFunction(firstOpName, firstWeight, dimLeft + dimRight, numNeurons);
        projString = firstOpName + "(concat(mapped_left, mapped_right)) AS features";

        registerNNFunction(firstOpName2, firstWeight2, dimLeft + dimRight, numNeurons2);
        projString2 = firstOpName2 + "(concat(mapped_left, mapped_right)) AS features2";

        registerNNFunction(firstOpName3, firstWeight3, dimLeft + dimRight, numNeurons3);
        projString3 = firstOpName3 + "(concat(mapped_left, mapped_right)) AS features3";
        //projString = "concat(mapped_left, mapped_right) AS features";
    }
    else if (pushDown == 1) {
        // all features pushed, so just perform aggregation
        projString = "vector_addition(factorized_left, factorized_right) AS features";
        projString2 = "vector_addition2(factorized_left2, factorized_right2) AS features2";
        projString3 = "vector_addition3(factorized_left3, factorized_right3) AS features3";
    }
    else if (pushDown == 2) {
        // only left features pushed, so apply first operator on right feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_right)";

        std::vector<std::vector<float>> subWeight2 = extractSubweight(firstWeight2, dimLeft, dimRight);
        std::string newOpName2 = firstOpName2 + "_right";
        registerNNFunction(newOpName2, subWeight2, dimRight, numNeurons2);
        std::string fNewNameFull2 = newOpName2 + "(mapped_right)";

        std::vector<std::vector<float>> subWeight3 = extractSubweight(firstWeight3, dimLeft, dimRight);
        std::string newOpName3 = firstOpName3 + "_right";
        registerNNFunction(newOpName3, subWeight3, dimRight, numNeurons3);
        std::string fNewNameFull3 = newOpName3 + "(mapped_right)";

        projString = "vector_addition(factorized_left, " + fNewNameFull + ") AS features";
        projString2 = "vector_addition2(factorized_left2, " + fNewNameFull2 + ") AS features2";
        projString3 = "vector_addition3(factorized_left3, " + fNewNameFull3 + ") AS features3";
    }
    else {
        // only right features pushed, so apply first operator on left feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_left)";

        std::vector<std::vector<float>> subWeight2 = extractSubweight(firstWeight2, 0, dimLeft);
        std::string newOpName2 = firstOpName2 + "_left";
        registerNNFunction(newOpName2, subWeight2, dimLeft, numNeurons2);
        std::string fNewNameFull2 = newOpName2 + "(mapped_left)";

        std::vector<std::vector<float>> subWeight3 = extractSubweight(firstWeight3, 0, dimLeft);
        std::string newOpName3 = firstOpName3 + "_left";
        registerNNFunction(newOpName3, subWeight3, dimLeft, numNeurons3);
        std::string fNewNameFull3 = newOpName3 + "(mapped_left)";

        projString = "vector_addition(" + fNewNameFull + ", factorized_right) AS features";
        projString2 = "vector_addition2(" + fNewNameFull2 + ", factorized_right2) AS features2";
        projString3 = "vector_addition3(" + fNewNameFull3 + ", factorized_right3) AS features3";
    }

    planBuilder = out.project({projString, projString2, projString3});
    std::cout << "Good before adding final inference" << std::endl;
    addModelMultiInferenceToQueryPlanAfterFactorize(planBuilder, planNodeIdGenerator, "features", "features2", "features3");
    std::cout << "Good after adding final inference" << std::endl;

      auto myPlan = planBuilder.planNode();
      std::cout << myPlan->toString(true, true) << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      if (isBatchExecution) {
          auto results = exec::test::AssertQueryBuilder(myPlan).maxDrivers(driverCount).copyResults(pool_.get());
          std::cout << "Results Size: " << results->size() << std::endl;
          std::cout << "Results:" << results->toString(0, 5) << std::endl;
      }
      else {
          auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
          std::cout << "Results Size: " << results->size() << std::endl;
          std::cout << "Results:" << results->toString(0, 5) << std::endl;
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
      return true;
  }



};


int main(int argc, char** argv) {
  setlocale(LC_TIME, "C");
  malloc_trim(0);

  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  Job2WayJoin bench;

  exec::registerVectorFunction(
          "get_feature_vec",
          GetFeatureVec::signatures(),
          std::make_unique<GetFeatureVec>());
  std::cout << "Completed registering function for get_feature_vec" << std::endl;

  exec::registerVectorFunction(
          "get_zero_vec",
          GetZeroVec::signatures(),
          std::make_unique<GetZeroVec>());
  std::cout << "Completed registering function for get_zero_vec" << std::endl;

  // retrieve the weights and set to map
  std::cout << "Reading model parameters" << std::endl;
  bench.findWeights("resources/model/job_any_128_700.h5");
  //bench.findWeightsSingleLayer("resources/model/job_any_3_600.h5");
  //bench.findWeights2("resources/model/job_any_32_600.h5");

  // retrieve the model operators from model expression IR
  std::string modelInput = "softmax3(mat_add3(mat_mul3(relu2(mat_add2(mat_mul2(relu1(mat_add1(mat_mul1(features)))))))))";
  //std::string modelInput = "softmax4(mat_add4(mat_mul4(features)))";
  //std::string modelInput2 = "softmax8(mat_add8(mat_mul8(features2)))";
  //std::string modelInput3 = "softmax9(mat_add9(mat_mul9(features3)))";
  //std::string modelInput3 = "softmax7(mat_add7(mat_mul7(relu6(mat_add6(mat_mul6(relu5(mat_add5(mat_mul5(features3)))))))))";
  std::cout << "Extracting model operators" << std::endl;
  std::vector<std::string> operators = bench.extractOperatorsInReverse(modelInput);
  bench.modelOperators = operators;
  //bench.modelOperators2 = bench.extractOperatorsInReverse(modelInput2);
  //bench.modelOperators3 = bench.extractOperatorsInReverse(modelInput3);

  int leftFeatureSize = 680;
  int rightFeatureSize = 20;
  std::string leftTableAlias = "it";
  std::string rightTableAlias = "mi";
  std::string leftKey = "it_id";
  std::string rightKey = "mi_info_type_id";
  int pushPlan = 0;

  // load necessary tables from csv files
  bench.loadDataTable("resources/data/imdb", leftTableAlias);
  bench.loadDataTable("resources/data/imdb", rightTableAlias);

  std::cout << "Performing Join" << std::endl;
  bool batchExecution = true;
  int batchCount = 64;
  int driverCount = 4;
  //bool ret = bench.createAndExecuteMultiTaskQuery(leftTableAlias, rightTableAlias, leftKey, rightKey, leftFeatureSize, rightFeatureSize, batchExecution, batchCount, driverCount, pushPlan);
  bool ret = bench.createAndExecuteQuery(leftTableAlias, rightTableAlias, leftKey, rightKey, leftFeatureSize, rightFeatureSize, batchExecution, batchCount, driverCount, pushPlan);

  return ret;
}

